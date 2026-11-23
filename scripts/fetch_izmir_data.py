"""Fetch İzmir pharmacy locations and published duty schedules.

Sources
-------
* İzmir Büyükşehir Belediyesi open API (no key):
  https://openapi.izmir.bel.tr/api/ibb/eczaneler  -> every pharmacy in the province
  with coordinates and duty region (Bolge). NOTE: the API's ``LokasyonX`` is the
  latitude and ``LokasyonY`` the longitude.
* İzmir Eczacı Odası duty search (POST form):
  https://www.izmireczaciodasi.org.tr/nobetci-eczaneler
  Only a sliding window is served: roughly yesterday .. end of next month. Past
  schedules disappear, so re-run this script (at least monthly) to accumulate history;
  new rows are merged into the existing CSV, never overwritten.

Usage
-----
    python scripts/fetch_izmir_data.py pharmacies
    python scripts/fetch_izmir_data.py duty [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Outputs go to data/raw/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
IBB_URL = "https://openapi.izmir.bel.tr/api/ibb/eczaneler"
IEO_URL = "https://www.izmireczaciodasi.org.tr/nobetci-eczaneler"
DUTY_CSV = RAW / "izmir_duty.csv"
DUTY_FIELDS = ["date", "region", "name", "address", "phone", "lat", "lon", "note"]


def _get(url: str, data: bytes | None = None) -> str:
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/json",
            "Accept-Language": "tr-TR,tr;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def fetch_pharmacies() -> Path:
    rows = json.loads(_get(IBB_URL))
    out = RAW / f"izmir_pharmacies_{dt.date.today():%Y%m%d}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pharmacy_id", "name", "region_id", "region", "district_id",
                    "lat", "lon", "address", "phone"])
        for r in rows:
            w.writerow([r["EczaneId"], r["Adi"].strip(), r["BolgeId"], r["Bolge"].strip(),
                        r["IlceId"], r["LokasyonX"], r["LokasyonY"],
                        r["Adres"].strip(), r["Telefon"]])
    print(f"{len(rows)} pharmacies -> {out}")
    return out


_BLOCK = re.compile(r'<h4 class="red">(.*?)nobet-karti', re.S)
_TITLE = re.compile(r"<strong>(.*?) - (\d{2}\.\d{2}\.\d{4}) / (.*?)</strong>", re.S)
_ADDR = re.compile(r"<i class='fa fa-home main-color'></i>(.*?)<br", re.S)
_TEL = re.compile(r'href="tel:([^"]*)"')
_GEO = re.compile(r"maps\?q=([-\d.]+),([-\d.]+)")
_TAGS = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAGS.sub(" ", s))).strip()


def parse_duty_page(page: str) -> list[dict]:
    rows = []
    for block in _BLOCK.findall(page):
        m = _TITLE.search(block)
        if not m:
            continue
        name, date, region = (_clean(x) for x in m.groups())
        head, _, body = block.partition("<i class='fa fa-home main-color'>")
        body = "<i class='fa fa-home main-color'>" + body
        note = " | ".join(x for x in (_clean(p) for p in
                          head.split("</h4>", 1)[-1].split("</p>")) if x)
        addr = _ADDR.search(body)
        geo = _GEO.search(body)
        rows.append({
            "date": dt.datetime.strptime(date, "%d.%m.%Y").date().isoformat(),
            "region": region,
            "name": name,
            "address": _clean(addr.group(1)) if addr else "",
            "phone": ";".join(_TEL.findall(body)),
            "lat": geo.group(1) if geo else "",
            "lon": geo.group(2) if geo else "",
            "note": note,
        })
    return rows


def fetch_duty(start: dt.date, end: dt.date, pause: float = 1.0) -> Path:
    existing: dict[tuple, dict] = {}
    if DUTY_CSV.exists():
        with DUTY_CSV.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[(r["date"], r["region"], r["name"])] = r
    n_before = len(existing)
    day = start
    while day <= end:
        body = urllib.parse.urlencode(
            {"tarih1": day.isoformat(), "ilce": "", "gnr": " Kayıt Ara"}).encode()
        rows = parse_duty_page(_get(IEO_URL, body))
        print(f"{day}: {len(rows)} on duty")
        if not rows and day > dt.date.today():
            print("no data beyond this date (outside published window); stopping")
            break
        for r in rows:
            existing[(r["date"], r["region"], r["name"])] = r
        day += dt.timedelta(days=1)
        time.sleep(pause)
    with DUTY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DUTY_FIELDS)
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])
    print(f"{len(existing) - n_before} new rows, {len(existing)} total -> {DUTY_CSV}")
    return DUTY_CSV


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pharmacies")
    d = sub.add_parser("duty")
    today = dt.date.today()
    d.add_argument("--start", type=dt.date.fromisoformat, default=today - dt.timedelta(days=1))
    d.add_argument("--end", type=dt.date.fromisoformat, default=today + dt.timedelta(days=70))
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.cmd == "pharmacies":
        fetch_pharmacies()
    else:
        fetch_duty(a.start, a.end)


if __name__ == "__main__":
    main()
