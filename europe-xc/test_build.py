#!/usr/bin/env python3
"""
Builds the outlook from a tiny synthetic ICON-EU run laid out like the DWD
server (bz2 GRIB2, south-to-north rows, longitudes 0..360) and checks the
pixels and the manifest.

Run with: python europe-xc/test_build.py
"""

import bz2
import datetime as dt
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import eccodes
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import build  # noqa: E402

RUN = '2026062100'
# 12 x 6 points, 1 degree apart: lon 355..366 (-5..6 E), lat 40..45 N.
NI, NJ = 12, 6
LON0, LAT0 = 355.0, 40.0

# One scenario per pair of columns (each output pixel is a 2 x 2 block).
#            hsurf  htop  hbas   cape  clct  rain mm/3h  u  v   land
SCENARIOS = [
    (1000, 3500, 3000, 100, 20, 0.0, 2.4, 3.2, 1),       # base 2,000 m over the ground, 14 km/h from the south-west: 40
    (1000, 3500, 3000, 1500, 20, 0.0, 2.4, 3.2, 1),      # the same, but storms may build: 19
    (1000, 3500, 3000, 100, 20, 1.5, 2.4, 3.2, 1),       # 0.5 mm/h: rain
    (1000, 3500, 3000, 100, 100, 0.0, 2.4, 3.2, 1),      # overcast: 40 x 0.3 = 12
    (1000, 2000, None, 100, 20, 0.0, 2.4, 3.2, 1),       # blue, 1,000 m usable: 13
    (0, 1500, None, 100, 20, 0.0, 10, 0, 0),         # sea
]


def field(index):
    """North-up field of one scenario column per pair; the north row pair is sea for FR_LAND."""
    out = np.zeros((NJ, NI))
    for c, scenario in enumerate(SCENARIOS):
        value = scenario[index]
        out[:, 2 * c:2 * c + 2] = np.nan if value is None else value
    return out


def grib(values_north_up):
    gid = eccodes.codes_grib_new_from_samples('regular_ll_sfc_grib2')
    try:
        eccodes.codes_set(gid, 'Ni', NI)
        eccodes.codes_set(gid, 'Nj', NJ)
        eccodes.codes_set(gid, 'jScansPositively', 1)
        eccodes.codes_set(gid, 'latitudeOfFirstGridPointInDegrees', LAT0)
        eccodes.codes_set(gid, 'latitudeOfLastGridPointInDegrees', LAT0 + NJ - 1)
        eccodes.codes_set(gid, 'longitudeOfFirstGridPointInDegrees', LON0)
        eccodes.codes_set(gid, 'longitudeOfLastGridPointInDegrees', LON0 + NI - 1)
        eccodes.codes_set(gid, 'iDirectionIncrementInDegrees', 1.0)
        eccodes.codes_set(gid, 'jDirectionIncrementInDegrees', 1.0)
        eccodes.codes_set(gid, 'bitmapPresent', 1)
        south_up = values_north_up[::-1, :].copy()
        missing = eccodes.codes_get(gid, 'missingValue')
        south_up[~np.isfinite(south_up)] = missing
        eccodes.codes_set_values(gid, south_up.ravel())
        return eccodes.codes_get_message(gid)
    finally:
        eccodes.codes_release(gid)


def write(root, var, name, values):
    folder = os.path.join(root, RUN[-2:], var.lower())
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), 'wb') as handle:
        handle.write(bz2.compress(grib(values)))


def fake_run(root):
    land = field(8)
    land[0:2, :] = 0  # the northern row pair is sea
    write(root, 'HSURF', f'icon-eu_europe_regular-lat-lon_time-invariant_{RUN}_HSURF.grib2.bz2', field(0))
    write(root, 'FR_LAND', f'icon-eu_europe_regular-lat-lon_time-invariant_{RUN}_FR_LAND.grib2.bz2', land)
    steps = set()
    for day in range(build.DAYS):
        for hour in build.FRAME_HOURS_UTC:
            steps.add(24 * day + hour)
            steps.add(24 * day + hour - build.RAIN_HOURS)
    for step in sorted(steps):
        single = f'icon-eu_europe_regular-lat-lon_single-level_{RUN}_{step:03d}'
        write(root, 'HTOP_DC', f'{single}_HTOP_DC.grib2.bz2', field(1))
        write(root, 'HBAS_CON', f'{single}_HBAS_CON.grib2.bz2', field(2))
        write(root, 'CAPE_ML', f'{single}_CAPE_ML.grib2.bz2', field(3))
        write(root, 'CLCT', f'{single}_CLCT.grib2.bz2', field(4))
        pressure = f'icon-eu_europe_regular-lat-lon_pressure-level_{RUN}_{step:03d}_700'
        write(root, 'U', f'{pressure}_U.grib2.bz2', field(6))
        write(root, 'V', f'{pressure}_V.grib2.bz2', field(7))
    # Totals grow steadily at the scenario's rain; hourly to +78 h, then every third hour, as DWD writes them.
    for step in list(range(1, build.HOURLY_UNTIL_STEP + 1)) + list(range(81, build.LAST_STEP + 1, 3)):
        single = f'icon-eu_europe_regular-lat-lon_single-level_{RUN}_{step:03d}'
        write(root, 'TOT_PREC', f'{single}_TOT_PREC.grib2.bz2', field(5) / build.RAIN_HOURS * step)


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        source_dir = os.path.join(cls.tmp.name, 'dwd')
        cls.out = os.path.join(cls.tmp.name, 'out')
        fake_run(source_dir)
        now = dt.datetime(2026, 6, 21, 4, 10, tzinfo=dt.timezone.utc)
        cls.manifest = build.build(build.Source(source_dir, RUN), cls.out, now)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def frame(self, index):
        path = os.path.join(self.out, self.manifest['frames'][index]['file'])
        return np.asarray(Image.open(path).convert('RGBA'))

    def test_manifest(self):
        with open(os.path.join(self.out, 'latest.json')) as handle:
            self.assertEqual(json.load(handle), self.manifest)
        self.assertEqual(self.manifest['run'], RUN)
        self.assertEqual((self.manifest['width'], self.manifest['height']), (6, 3))
        # Blocks of points at -5..6 E and 40..45 N, half a degree past the outer points.
        self.assertEqual(self.manifest['bounds'], [-5.5, 39.5, 6.5, 45.5])
        times = [f['time'] for f in self.manifest['frames']]
        self.assertEqual(len(times), build.DAYS * len(build.FRAME_HOURS_UTC))
        self.assertEqual(times[:3], ['2026-06-21T09:00:00Z', '2026-06-21T12:00:00Z', '2026-06-21T15:00:00Z'])
        self.assertEqual(times[-1], '2026-06-25T15:00:00Z')
        self.assertIn('DWD', self.manifest['attribution'])

    def test_pixels(self):
        px = self.frame(0)
        self.assertEqual(px.shape, (3, 6, 4))
        row = px[2]  # the southern row, all land but the sea column
        # Wind from 217 degrees at 14.4 km/h: sector 10 (SW), 4 steps of 4 km/h.
        self.assertEqual(row[0].tolist(), [120, 128 + 60, 10 * 16 + 4, 255], 'score 40, cumulus base 3,000 m, SW 16 km/h')
        self.assertEqual(row[1][0], 128 + 19 * 3, 'storms cap the score under Flyable and set the storm bit')
        self.assertEqual(row[2][0], 255, 'rain')
        self.assertEqual(row[3][0], 12 * 3, 'overcast')
        self.assertEqual(row[4][:2].tolist(), [40, 40], 'blue thermals to 2,000 m: 13.3 of 40 in thirds, no cumulus bit')
        self.assertEqual(row[5][3], 0, 'sea is transparent')
        self.assertEqual(px[0][0][3], 0, 'north-up: the first row is the sea row')
        self.assertEqual(px[1][0][3], 255)

    def test_rain(self):
        rain = self.manifest['rain']
        self.assertEqual((rain['width'], rain['height']), (NI, NJ))
        self.assertEqual(rain['bounds'], [-5.5, 39.5, 6.5, 45.5])
        times = [f['time'] for f in rain['frames']]
        # 05..18 UTC hourly to +78 h (day 3 06 UTC), then 06, 09, 12, 15, 18 UTC.
        self.assertEqual(times[:2], ['2026-06-21T05:00:00Z', '2026-06-21T06:00:00Z'])
        self.assertIn('2026-06-24T06:00:00Z', times)
        self.assertNotIn('2026-06-24T07:00:00Z', times)
        self.assertIn('2026-06-24T09:00:00Z', times)
        self.assertEqual(times[-1], '2026-06-25T18:00:00Z')
        spans = {f['time']: f['hours'] for f in rain['frames']}
        self.assertEqual((spans['2026-06-24T06:00:00Z'], spans['2026-06-24T09:00:00Z']), (1, 3))
        for frame in (rain['frames'][0], rain['frames'][-1]):
            px = np.asarray(Image.open(os.path.join(self.out, frame['file'])))
            self.assertEqual(px.shape, (NJ, NI))
            # Scenario 3 rains 1.5 mm in 3 hours: 0.5 mm/h is 2 + 25 x log2(5) = 60. The rest is dry.
            self.assertEqual(px[3][4:6].tolist(), [60, 60], frame['file'])
            self.assertEqual(px[3][0], 1)
        self.assertEqual(build.encode_rain(np.array([np.nan, 0.0, 0.05, 0.1, 1000.0])).tolist(), [0, 1, 1, 2, 255])

    def test_latest_run(self):
        self.assertEqual(build.latest_run(dt.datetime(2026, 6, 21, 3, 0, tzinfo=dt.timezone.utc)), '2026062012')
        self.assertEqual(build.latest_run(dt.datetime(2026, 6, 21, 5, 0, tzinfo=dt.timezone.utc)), '2026062100')
        self.assertEqual(build.latest_run(dt.datetime(2026, 6, 21, 16, 40, tzinfo=dt.timezone.utc)), '2026062112')


if __name__ == '__main__':
    unittest.main()
