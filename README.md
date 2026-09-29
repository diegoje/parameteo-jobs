# ParaMeteo scheduled jobs

The two scheduled jobs behind [ParaMeteo](https://parameteo.app), kept in a
public repository because GitHub Actions minutes are free here. The app
itself lives in a private repository; nothing in this one is secret.

| Workflow | When (UTC) | What it does |
| --- | --- | --- |
| `europe-xc.yml` | 04:40 and 16:40 | Downloads DWD's ICON-EU run from opendata.dwd.de, scores every 7 km cell for cross-country flying (`europe-xc/build.py`) and uploads 15 small PNG frames, about 70 hourly rain frames for the map's time rail, then `latest.json`, to the app. |
| `site-forecast.yml` | 04:10 | Asks the app to score every listed launch for the morning, in batches, inside a fixed share of the Open-Meteo budget. |

Both call the app's internal endpoints with a shared secret. Set these
under Settings → Secrets and variables → Actions:

- `APP_URL`: the app's address, e.g. `https://parameteo.app`
- `INTERNAL_CRON_SECRET`: the same value as in the app's Vercel settings

To try a job straight away: Actions tab → pick the workflow → Run workflow.

## The Europe XC builder

```
pip install -r europe-xc/requirements.txt
python europe-xc/test_build.py        # the encoding and scoring tests
python europe-xc/build.py --out out   # the latest run, into ./out
```

Forecast data: Deutscher Wetterdienst, ICON-EU, CC BY 4.0.
