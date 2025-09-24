#!/usr/bin/env python3
"""
Enhanced dummy CDR server with two modes:
 1) Synthetic generator (original behavior)
 2) Replay / sample mode that reads a CSV/JSONL of real CDRs and emits them (with optional perturbation/mixing)

Features:
 - --replay / --file to point to a CSV or JSONL of CDRs
 - automatic column detection for phone/duration/imsi/imei/isFraud where possible
 - --rate records per second (default: 0.5 rps -> 2s interval like original)
 - --shuffle to shuffle input records
 - --loop to repeat the file indefinitely
 - --mix-real fraction (0-1): fraction of emitted records that are from the real file. When <1.0, synthetic records are mixed in.
 - --preserve-timestamp: if your CDRs have timestamps and you want them preserved; otherwise timestamps may be shifted to "now" when perturbing
 - phone masking before sending (optionally)

Usage examples:
  python dummy_server.py                      # original synthetic generator (1 record every 2s)
  python dummy_server.py --replay cdrs.csv --rate 20 --shuffle --loop
  python dummy_server.py --replay cdrs.csv --rate 10 --mix-real 0.7

This script attempts to be drop-in compatible with the original FRAUD_API_URL constant
"""

import argparse
import csv
import json
import time
import uuid
import random
import requests
import os
from datetime import datetime, timedelta

FRAUD_API_URL = "http://localhost:8001/process_cdr"

# --- synthetic generator (kept from original) ---

def make_synthetic_cdr(now_ts=None):
    """Return a synthetic CDR similar to the original simple generator."""
    return {
        "call_id": str(uuid.uuid4()),
        "caller": f"+233{random.randint(200000000,999999999)}",
        "callee": f"+233{random.randint(200000000,999999999)}",
        "duration": random.randint(1, 600),
        "roaming": random.choice([0, 1]),
        "call_type": random.choice(["MO", "MT"]),
        "sim": f"SIM{random.randint(10000,99999)}",
        "imei": f"IMEI{random.randint(1000000,9999999)}",
        "imsi": f"IMSI{random.randint(1000000,9999999)}",
        "timestamp": (now_ts or datetime.utcnow()).isoformat()
    }

# --- helpers for replay mode ---

def detect_columns(columns):
    """Heuristic detection of common column names.
    Returns a mapping dict with keys: caller, callee, duration, imsi, imei, timestamp, isFraud
    """
    lc = [c.lower() for c in columns]
    mapping = {}
    # phone-like
    for name in ['caller', 'callee', 'phone number', 'msisdn', 'number']:
        for c in columns:
            if name in c.lower():
                mapping.setdefault('caller', c)
                break
        if 'caller' in mapping:
            break
    # if no explicit callee, we'll reuse caller or leave None
    for name in ['callee', 'destination', 'called number']:
        for c in columns:
            if name in c.lower():
                mapping.setdefault('callee', c)
                break
        if 'callee' in mapping:
            break
    # duration
    for name in ['duration', 'call duration', 'secs', 'seconds']:
        for c in columns:
            if name in c.lower():
                mapping.setdefault('duration', c)
                break
        if 'duration' in mapping:
            break
    # imsi/imei
    for name in ['imsi']:
        for c in columns:
            if name in c.lower():
                mapping.setdefault('imsi', c)
                break
    for name in ['imei']:
        for c in columns:
            if name in c.lower():
                mapping.setdefault('imei', c)
                break
    # timestamp
    for name in ['timestamp', 'start_time', 'start', 'time', 'date']:
        for c in columns:
            if name in c.lower():
                mapping.setdefault('timestamp', c)
                break
        if 'timestamp' in mapping:
            break
    # isFraud
    for name in ['isfraud', 'fraud', 'label']:
        for c in columns:
            if name in c.lower():
                mapping.setdefault('isFraud', c)
                break
        if 'isFraud' in mapping:
            break
    return mapping


def read_csv_or_jsonl(path):
    """Read CSV or JSONL into list of dicts."""
    rows = []
    _, ext = os.path.splitext(path.lower())
    if ext == '.jsonl' or ext == '.ndjson':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        # try csv
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def mask_phone(number, keep_prefix=3, keep_suffix=2):
    s = str(number)
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) <= keep_prefix + keep_suffix:
        return 'X'*len(digits)
    return digits[:keep_prefix] + 'X'*(len(digits)-keep_prefix-keep_suffix) + digits[-keep_suffix:]


def to_cdr_from_row(row, mapping, now_ts=None, preserve_timestamp=False):
    """Map an input row (dict) to a CDR dict used by the FRAUD API.
    Uses mapping detected earlier; falls back to sensible defaults.
    """
    cdr = {}
    # call id
    cdr['call_id'] = str(uuid.uuid4())
    # caller/callee
    caller_col = mapping.get('caller')
    callee_col = mapping.get('callee')
    if caller_col and row.get(caller_col):
        cdr['caller'] = str(row.get(caller_col))
    else:
        cdr['caller'] = row.get(callee_col) if callee_col and row.get(callee_col) else f"+233{random.randint(200000000,999999999)}"
    if callee_col and row.get(callee_col):
        cdr['callee'] = str(row.get(callee_col))
    else:
        cdr['callee'] = row.get(caller_col) if caller_col and row.get(caller_col) else f"+233{random.randint(200000000,999999999)}"
    # duration
    dur_col = mapping.get('duration')
    try:
        if dur_col and row.get(dur_col) not in [None, '']:
            cdr['duration'] = int(float(row.get(dur_col)))
        else:
            cdr['duration'] = random.randint(1, 600)
    except Exception:
        cdr['duration'] = random.randint(1, 600)
    # roaming - try to infer or default
    cdr['roaming'] = int(row.get(mapping.get('roaming'), 0)) if mapping.get('roaming') and row.get(mapping.get('roaming')) is not None else random.choice([0,1])
    # call_type
    cdr['call_type'] = row.get(mapping.get('call_type')) or random.choice(['MO','MT'])
    # sim/imsi/imei
    cdr['sim'] = row.get(mapping.get('sim')) or f"SIM{random.randint(10000,99999)}"
    cdr['imei'] = row.get(mapping.get('imei')) or f"IMEI{random.randint(1000000,9999999)}"
    cdr['imsi'] = row.get(mapping.get('imsi')) or f"IMSI{random.randint(1000000,9999999)}"
    # timestamp
    ts_col = mapping.get('timestamp')
    if ts_col and row.get(ts_col) and preserve_timestamp:
        cdr['timestamp'] = str(row.get(ts_col))
    else:
        # set to now +/- small jitter
        jitter = random.randint(-30,30)
        cdr['timestamp'] = (now_ts or datetime.utcnow() + timedelta(seconds=jitter)).isoformat()
    # isFraud label if available
    if mapping.get('isFraud') and row.get(mapping.get('isFraud')) not in [None, '']:
        # try to coerce to int
        val = row.get(mapping.get('isFraud'))
        try:
            cdr['isFraud'] = int(float(val))
        except Exception:
            # allow strings 'yes'/'no'
            v = str(val).lower()
            cdr['isFraud'] = 1 if v in ('1','true','yes','y') else 0
    return cdr


def perturb_cdr(cdr, jitter_seconds=5, duration_jitter_pct=0.1, mask_phones=True):
    # clone
    out = dict(cdr)
    # shift timestamp slightly
    try:
        dt = datetime.fromisoformat(out.get('timestamp'))
        dt = dt + timedelta(seconds=random.randint(-jitter_seconds, jitter_seconds))
        out['timestamp'] = dt.isoformat()
    except Exception:
        out['timestamp'] = datetime.utcnow().isoformat()
    # jitter duration by +-duration_jitter_pct
    try:
        d = int(out.get('duration', 1))
        factor = 1 + ((random.random() - 0.5) * 2 * duration_jitter_pct)
        out['duration'] = max(1, int(d * factor))
    except Exception:
        out['duration'] = int(random.randint(1,600))
    # mask phones
    if mask_phones:
        try:
            out['caller'] = mask_phone(out.get('caller'))
        except Exception:
            pass
        try:
            out['callee'] = mask_phone(out.get('callee'))
        except Exception:
            pass
    return out


def send_cdr(cdr, url, timeout=5):
    try:
        r = requests.post(url, json=cdr, timeout=timeout)
        # try to parse JSON safely
        try:
            resp = r.json()
        except Exception:
            resp = r.text
        return True, r.status_code, resp
    except Exception as e:
        return False, None, str(e)


# --- main ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--replay', dest='replay', help='Path to CSV or JSONL of CDRs (if not set, run synthetic generator)')
    p.add_argument('--rate', type=float, default=0.5, help='Records per second (default 0.5 -> one record every 2s)')
    p.add_argument('--shuffle', action='store_true', help='Shuffle replay records')
    p.add_argument('--loop', action='store_true', help='Loop the file indefinitely')
    p.add_argument('--mix-real', type=float, default=1.0, help='Fraction [0..1] of records emitted that come from the real file. Remaining are synthetic. Default 1.0 (only real).')
    p.add_argument('--preserve-timestamp', action='store_true', help='Preserve timestamp column values from the CSV (if present)')
    p.add_argument('--mask-phones', action='store_true', help='Mask phone numbers before sending (recommended for testing)')
    p.add_argument('--no-perturb', action='store_true', help='Disable perturbation of real records')
    p.add_argument('--url', type=str, default=FRAUD_API_URL, help='FRAUD API URL')
    p.add_argument('--timeout', type=float, default=5.0, help='HTTP timeout in seconds')

    args = p.parse_args()

    interval = 1.0 / args.rate if args.rate > 0 else 0

    if not args.replay:
        # original synthetic loop
        print('Starting synthetic generator (original behavior)')
        try:
            while True:
                cdr = make_synthetic_cdr()
                ok, code, resp = send_cdr(cdr, args.url, timeout=args.timeout)
                print('sent', cdr['call_id'], 'ok' if ok else 'err', code, resp)
                time.sleep(interval)
        except KeyboardInterrupt:
            print('Stopped by user')
        return

    # Replay mode
    print(f'Loading file: {args.replay}')
    rows = read_csv_or_jsonl(args.replay)
    if not rows:
        print('No rows found in file. Exiting.')
        return
    mapping = detect_columns(rows[0].keys())
    print('Detected mapping:', mapping)

    # create list of cdrs from rows
    now_ts = datetime.utcnow()
    base_cdrs = []
    for r in rows:
        try:
            cdr = to_cdr_from_row(r, mapping, now_ts=now_ts, preserve_timestamp=args.preserve_timestamp)
            base_cdrs.append(cdr)
        except Exception as e:
            print('Failed mapping row, skipping. Err:', e)

    if not base_cdrs:
        print('No valid CDRs after mapping. Exiting.')
        return

    print(f'Prepared {len(base_cdrs)} CDR(s) for replay.')

    # loop/emit
    try:
        while True:
            pool = list(base_cdrs)
            if args.shuffle:
                random.shuffle(pool)
            for cdr in pool:
                # decide whether to emit a real or a synthetic record
                if args.mix_real >= 1.0 or random.random() < args.mix_real:
                    out = dict(cdr)
                    if not args.no_perturb:
                        out = perturb_cdr(out, mask_phones=args.mask_phones)
                    else:
                        if args.mask_phones:
                            out['caller'] = mask_phone(out.get('caller'))
                            out['callee'] = mask_phone(out.get('callee'))
                else:
                    out = make_synthetic_cdr(now_ts=now_ts)
                    if args.mask_phones:
                        out['caller'] = mask_phone(out.get('caller'))
                        out['callee'] = mask_phone(out.get('callee'))

                ok, code, resp = send_cdr(out, args.url, timeout=args.timeout)
                print('sent', out.get('call_id'), 'ok' if ok else 'err', code, resp)
                time.sleep(interval)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print('Stopped by user')


if __name__ == '__main__':
    main()
