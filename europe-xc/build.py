#!/usr/bin/env python3
"""
Europe XC outlook from DWD ICON-EU open data.

Downloads a handful of ICON-EU fields for three hours of each of the next
five days, scores every land cell for cross-country potential, and writes
one small PNG per (day, hour) plus a manifest. The PNGs carry data, not
colours; the map decodes and paints them in the browser.

Each pixel is the mean of 2 x 2 model points (about 14 km), encoded as:
  R  XC score out of 40, times 3 (0..120), plus 128 where storms may build;
     255 means rain around that hour
  G  usable ceiling in metres above sea level / 50 (0..127, up to 6,350 m),
     plus 128 where that ceiling is a cumulus base rather than a blue top
  B  wind at 700 hPa (about 3,000 m): the high 4 bits give the direction it
     blows from in 16 sectors (0 = north, clockwise), the low 4 bits the
     speed in 4 km/h steps (15 = 60 km/h or more)
  A  255 over land, 0 over sea and outside the model

Usage:
  build.py --out DIR [--run YYYYMMDDHH] [--source URL_OR_DIR]

Without --run it takes the latest 00 or 12 UTC run. A 12 UTC run starts
at that day's 15 UTC frame.

The source defaults to https://opendata.dwd.de/weather/nwp/icon-eu/grib.
A local directory with the same layout works too (used by the tests).

Data: Deutscher Wetterdienst (DWD), ICON-EU, CC BY 4.0.
"""

import argparse
import bz2
import datetime as dt
import io
import json
import os
import sys
import urllib.request

import numpy as np
import eccodes
from PIL import Image

DWD_SOURCE = 'https://opendata.dwd.de/weather/nwp/icon-eu/grib'
DAYS = 5
# Frames in UTC: 11, 14 and 17 h in Swiss summer time, 10, 13 and 16 h in winter.
# ICON-EU writes every hour to +78 h and every third hour after, so these
# hours exist on all five days of the 00 and 12 UTC runs.
FRAME_HOURS_UTC = [9, 12, 15]
# Rain rate is the precipitation of the three hours up to the frame.
RAIN_HOURS = 3
SINGLE = ['HTOP_DC', 'HBAS_CON', 'CAPE_ML', 'CLCT', 'TOT_PREC']
RAIN_MM_PER_H = 0.3
STORM_CAPE = 1200.0
DECIMATE = 2


# ---------------------------------------------------------------- scoring --

def score_cells(htop, hbas, cape, clct, rain_mm, wind_kmh, hsurf):
    """Score every cell for XC, out of 40. Arrays of equal shape; NaN = missing.

    Returns (score out of 40, usable ceiling in m ASL, rain flag, storm flag, cumulus flag).
    """
    has_base = np.isfinite(hbas) & (hbas > hsurf) & (hbas < htop)
    ceiling = np.where(has_base, hbas, htop)
    usable = np.clip(ceiling - hsurf, 0, None)
    # About 500 m of usable height is where local soaring starts, 2,000 m is a big XC day.
    lift = np.clip((usable - 500.0) / 1500.0, 0.0, 1.0)
    sun = 1.0 - 0.7 * np.clip((clct - 30.0) / 70.0, 0.0, 1.0)
    wind = np.where(wind_kmh <= 20.0, 1.0, np.clip(1.0 - (wind_kmh - 20.0) / 25.0 * 0.75, 0.15, 1.0))
    score = 40.0 * lift * sun * wind
    # Where storms may build, the day is at best Marginal (under 20), as on the site pages.
    storm = np.nan_to_num(cape) >= STORM_CAPE
    score = np.where(storm, np.minimum(score, 19.0), score)
    rain = rain_mm >= RAIN_MM_PER_H
    score = np.where(rain, 0.0, score)
    score = np.where(np.isfinite(score), score, 0.0)
    return score, np.where(np.isfinite(ceiling), ceiling, 0.0), rain, storm, has_base


def block_mean(values, weights):
    """Weighted mean of each DECIMATE x DECIMATE block. A last partial row or column is dropped."""
    d = DECIMATE
    h, w = values.shape[0] // d * d, values.shape[1] // d * d
    v = np.nan_to_num(values[:h, :w].astype(np.float64)).reshape(h // d, d, w // d, d)
    wt = weights[:h, :w].astype(np.float64).reshape(h // d, d, w // d, d)
    total = wt.sum(axis=(1, 3))
    return (v * wt).sum(axis=(1, 3)) / np.where(total > 0, total, 1.0)


def encode_frame(score40, ceiling, rain, storm, cumulus, u, v, land):
    """Pack one frame into RGBA bytes (north up). u and v in m/s."""
    # Thirds of a point keep the band edges smooth where the score changes slowly.
    r = np.where(rain, 255, np.round(np.clip(score40, 0, 40) * 3) + np.where(storm, 128, 0)).astype(np.uint8)
    g = (np.clip(np.round(ceiling / 50.0), 0, 127) + np.where(cumulus, 128, 0)).astype(np.uint8)
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    speed = np.clip(np.round(np.hypot(u, v) * 3.6 / 4.0), 0, 15)
    blows_from = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    sector = np.round(blows_from / 22.5) % 16
    b = (sector * 16 + speed).astype(np.uint8)
    a = np.where(land, 255, 0).astype(np.uint8)
    return np.dstack([r, g, b, a])


# ------------------------------------------------------------------- GRIB --

class Grid:
    def __init__(self, ni, nj, lat_first, lat_last, lon_first, lon_last, j_positive):
        self.ni, self.nj = ni, nj
        self.lat_south, self.lat_north = min(lat_first, lat_last), max(lat_first, lat_last)
        self.lon_west, self.lon_east = lon_first, lon_last
        self.j_positive = j_positive

    def key(self):
        return (self.ni, self.nj, round(self.lat_south, 4), round(self.lon_west, 4))


def read_grib(data):
    """First message of a GRIB2 file as a north-up 2D array (NaN for missing) and its grid."""
    gid = eccodes.codes_new_from_message(data)
    try:
        ni = eccodes.codes_get(gid, 'Ni')
        nj = eccodes.codes_get(gid, 'Nj')
        missing = eccodes.codes_get(gid, 'missingValue')
        # Points masked by the bitmap come back as missingValue.
        values = eccodes.codes_get_values(gid).astype(np.float64)
        values[values == missing] = np.nan
        grid = Grid(
            ni, nj,
            eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees'),
            eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees'),
            eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees'),
            eccodes.codes_get(gid, 'longitudeOfLastGridPointInDegrees'),
            eccodes.codes_get(gid, 'jScansPositively') == 1,
        )
    finally:
        eccodes.codes_release(gid)
    field = values.reshape(nj, ni)
    if grid.j_positive:
        field = field[::-1, :]
    if grid.lon_west > 180:
        grid.lon_west -= 360
    if grid.lon_east > 180:
        grid.lon_east -= 360
    return field, grid


class Source:
    def __init__(self, base, run):
        self.base = base.rstrip('/')
        self.run = run
        self.hour = run[-2:]
        # Total precipitation by step: one frame's total is the next frame's start.
        self._rain = {}

    def _get(self, rel):
        if self.base.startswith('http'):
            url = f'{self.base}/{rel}'
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        with open(os.path.join(self.base, rel), 'rb') as handle:
            return handle.read()

    def single(self, var, step):
        if var != 'TOT_PREC':
            return self._single(var, step)
        if step not in self._rain:
            self._rain[step] = self._single(var, step)
        return self._rain[step]

    def _single(self, var, step):
        name = f'icon-eu_europe_regular-lat-lon_single-level_{self.run}_{step:03d}_{var}.grib2.bz2'
        return read_grib(bz2.decompress(self._get(f'{self.hour}/{var.lower()}/{name}')))

    def pressure(self, var, level, step):
        name = f'icon-eu_europe_regular-lat-lon_pressure-level_{self.run}_{step:03d}_{level}_{var}.grib2.bz2'
        return read_grib(bz2.decompress(self._get(f'{self.hour}/{var.lower()}/{name}')))

    def invariant(self, var):
        name = f'icon-eu_europe_regular-lat-lon_time-invariant_{self.run}_{var}.grib2.bz2'
        return read_grib(bz2.decompress(self._get(f'{self.hour}/{var.lower()}/{name}')))


# ------------------------------------------------------------------- main --

def latest_run(now):
    """The most recent 00 or 12 UTC run old enough to be complete on the server (about 3.5 h)."""
    ready = now - dt.timedelta(hours=4)
    return ready.strftime('%Y%m%d') + ('12' if ready.hour >= 12 else '00')


def build(source, out_dir, now):
    run_time = dt.datetime.strptime(source.run, '%Y%m%d%H').replace(tzinfo=dt.timezone.utc)
    hsurf, grid = source.invariant('HSURF')
    fr_land, _ = source.invariant('FR_LAND')
    land = (np.nan_to_num(fr_land) >= 0.5).astype(np.float64)
    everywhere = np.ones_like(land)
    os.makedirs(os.path.join(out_dir, source.run), exist_ok=True)

    frames = []
    # A 12 UTC run ends at midday of a sixth calendar day.
    for day in range(DAYS + 1):
        for hour in FRAME_HOURS_UTC:
            valid = run_time.replace(hour=0) + dt.timedelta(days=day, hours=hour)
            step = int((valid - run_time).total_seconds() // 3600)
            if step < RAIN_HOURS or step > 120:
                continue
            fields = {}
            for var in SINGLE:
                field, g = source.single(var, step)
                if g.key() != grid.key():
                    raise SystemExit(f'{var} at +{step} h is on another grid')
                fields[var] = field
            previous, _ = source.single('TOT_PREC', step - RAIN_HOURS)
            rain_mm = np.clip(fields['TOT_PREC'] - previous, 0, None) / RAIN_HOURS
            u, _ = source.pressure('U', 700, step)
            v, _ = source.pressure('V', 700, step)
            wind_kmh = np.hypot(u, v) * 3.6
            score40, ceiling, rain, storm, cumulus = score_cells(
                fields['HTOP_DC'], fields['HBAS_CON'], fields['CAPE_ML'], fields['CLCT'], rain_mm, wind_kmh, hsurf,
            )
            storm = block_mean(storm, land) >= 0.5
            # A block that counts as stormy keeps the Marginal cap after averaging.
            score40 = np.where(storm, np.minimum(block_mean(score40, land), 19.0), block_mean(score40, land))
            rain = block_mean(rain, land) >= 0.5
            rgba = encode_frame(
                score40,
                block_mean(ceiling, land),
                rain,
                storm & ~rain,
                block_mean(cumulus, land) >= 0.5,
                block_mean(u, everywhere),
                block_mean(v, everywhere),
                block_mean(land, everywhere) >= 0.5,
            )
            name = f'{source.run}/{valid.strftime("%Y%m%dT%H")}.png'
            Image.fromarray(rgba, 'RGBA').save(os.path.join(out_dir, name), optimize=True)
            frames.append({'time': valid.strftime('%Y-%m-%dT%H:00:00Z'), 'file': name})
            print(f'{name}: best score {round(float(score40.max()))} of 40', flush=True)

    if not frames:
        raise SystemExit(f'run {source.run} has no frames')
    height, width = rgba.shape[:2]
    lat_step = (grid.lat_north - grid.lat_south) / (grid.nj - 1)
    lon_step = (grid.lon_east - grid.lon_west) / (grid.ni - 1)
    west = grid.lon_west - lon_step / 2
    north = grid.lat_north + lat_step / 2
    manifest = {
        'model': 'ICON-EU',
        'run': source.run,
        'generatedAt': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
        # Outer edges of the pixels: west, south, east, north.
        'bounds': [
            round(west, 5),
            round(north - lat_step * DECIMATE * height, 5),
            round(west + lon_step * DECIMATE * width, 5),
            round(north, 5),
        ],
        'width': width,
        'height': height,
        'encoding': {
            'score': '(R & 127) / 3, R & 128 storms, 255 rain',
            'ceiling': '(G & 127) * 50 m, G & 128 cumulus base',
            'wind700': '(B >> 4) * 22.5 deg from, (B & 15) * 4 km/h',
            'land': 'A = 255',
        },
        'frames': frames,
        'attribution': 'Deutscher Wetterdienst (DWD), ICON-EU, CC BY 4.0',
    }
    with open(os.path.join(out_dir, 'latest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--run')
    parser.add_argument('--source', default=DWD_SOURCE)
    args = parser.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    run = args.run or latest_run(now)
    manifest = build(Source(args.source, run), args.out, now)
    print(f'{len(manifest["frames"])} frames for run {run}')


if __name__ == '__main__':
    sys.exit(main())
