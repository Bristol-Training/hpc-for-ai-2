#!/usr/bin/env python3
"""
Collect UK front-page headlines from The Hear API (https://www.thehear.org/api)
into a nested dict:

    data[paper][year][title] = {
        "subtitle":   str,
        "link":       str,
        "first_seen": ISO timestamp (UTC),
        "last_seen":  ISO timestamp (UTC),
        "snapshots":  int   # how many sampled snapshots it was the lead story in
    }

ABOUT THE DATA
- The Hear records each outlet's *main front-page headline* (the lead story),
  not every article published. A historical snapshot gives the lead story of
  each of the ~21 UK sources at one moment.
- There is no endpoint for "all headlines in a date range", so this script
  samples snapshots every N hours from the archive start (2024-09-05) to now
  and merges the results. Smaller intervals catch more lead stories but take
  longer. (Leads refresh roughly every 15 minutes on The Hear.)
- Raw responses are cached in ./hear_cache/ so you can stop and resume, and
  re-running only fetches timestamps you don't have yet.

USAGE
    python hear_uk_headlines.py                       # every 6 hours, full archive
    python hear_uk_headlines.py --hours 2             # denser sampling
    python hear_uk_headlines.py --start 2025-01-01 --end 2025-12-31

    # or from Python:
    from hear_uk_headlines import collect, load
    data = collect()                  # fetch (uses cache)
    data = load("uk_headlines.json")  # reload a saved run

Requires only the Python standard library.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_URL = "https://www.thehear.org/api/country-view/{country}"
ARCHIVE_START = {"uk": "2024-09-05"}
USER_AGENT = "uk-headline-research-script/1.0 (personal research)"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_snapshot(country: str, at: str, cache_dir: Path,
                   retries: int = 3, timeout: int = 30) -> dict | None:
    """Fetch one historical snapshot (cached on disk). Returns parsed JSON or None."""
    cache_file = cache_dir / f"{country}_{at.replace(':', '-')}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache_file.unlink()  # corrupt cache, refetch

    url = BASE_URL.format(country=country) + "?" + urllib.parse.urlencode({"at": at})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
        except urllib.error.HTTPError as e:
            if e.code == 404:          # no data for that moment
                return None
            if e.code == 429:          # rate limited: back off harder
                time.sleep(30 * attempt)
                continue
            time.sleep(2 * attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2 * attempt)
    print(f"  ! giving up on {at}")
    return None


def sample_times(start: str, end: str | None, every_hours: float):
    t = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    stop = (datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(days=1)
            if end else datetime.now(timezone.utc))
    step = timedelta(hours=every_hours)
    while t < stop:
        yield iso_z(t)
        t += step


# ---------------------------------------------------------------------------
# Building the nested dict
# ---------------------------------------------------------------------------

def normalise_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def add_snapshot(data: dict, snapshot: dict) -> int:
    """Merge one API response into data[paper][year][title]. Returns # new titles."""
    new = 0
    for h in snapshot.get("headlines", []) or []:
        paper = (h.get("sourceLabel") or "Unknown").strip()
        title = normalise_title(h.get("headline"))
        if not title:
            continue
        seen = h.get("capturedAt") or snapshot.get("asOfUtc")
        year = seen[:4] if seen else "unknown"

        entry = data[paper][year].get(title)
        if entry is None:
            data[paper][year][title] = {
                "subtitle": normalise_title(h.get("subtitle")),
                "link": h.get("link", ""),
                "first_seen": seen,
                "last_seen": seen,
                "snapshots": 1,
            }
            new += 1
        else:
            entry["snapshots"] += 1
            if seen and seen < entry["first_seen"]:
                entry["first_seen"] = seen
            if seen and seen > entry["last_seen"]:
                entry["last_seen"] = seen
    return new


def collect(country="uk", start=None, end=None, every_hours=6.0,
            delay=1.0, cache_dir="hear_cache") -> dict:
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    start = start or ARCHIVE_START.get(country, "2024-09-05")

    data = defaultdict(lambda: defaultdict(dict))
    times = list(sample_times(start, end, every_hours))
    print(f"Sampling {len(times)} snapshots for {country} from {start} "
          f"every {every_hours}h ...")

    for i, at in enumerate(times, 1):
        cached = (cache / f"{country}_{at.replace(':', '-')}.json").exists()
        snap = fetch_snapshot(country, at, cache)
        if snap:
            add_snapshot(data, snap)
        if not cached:
            time.sleep(delay)          # be polite to a free nonprofit API
        if i % 50 == 0 or i == len(times):
            n = sum(len(t) for p in data.values() for t in p.values())
            print(f"  {i}/{len(times)}  ({at})  unique headlines so far: {n}")

    # convert defaultdicts to plain dicts, sorted for readability
    return {p: {y: dict(sorted(t.items())) for y, t in sorted(yrs.items())}
            for p, yrs in sorted(data.items())}


def load(path="uk_headlines.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="uk")
    ap.add_argument("--start", help="YYYY-MM-DD (default: archive start)")
    ap.add_argument("--end", help="YYYY-MM-DD (default: now)")
    ap.add_argument("--hours", type=float, default=6.0, help="sampling interval in hours")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--out", default="uk_headlines.json")
    args = ap.parse_args()

    data = collect(args.country, args.start, args.end, args.hours, args.delay)
    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(len(t) for p in data.values() for t in p.values())
    print(f"\nSaved {total} unique headlines from {len(data)} sources to {args.out}")
    for paper, years in data.items():
        print(f"  {paper[:30]:30} " + "  ".join(f"{y}: {len(t)}" for y, t in years.items()))


if __name__ == "__main__":
    main()
