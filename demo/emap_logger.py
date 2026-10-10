"""Logger for the live eMap feed (vadimklimenko.com/map/statuses.json): raion-level alerts with alert_level red / yellow.

Polite by design: one request per minute at most (the interval cannot be set below 60 s), conditional requests
(If-None-Match / If-Modified-Since, a 304 costs almost nothing), an identifying User-Agent, and a growing pause after errors.

What the feed gives and what it does not (checked on 2026-10-10):
- enabled entries carry enabled_at (exact start, ms) and alert_level ("red" or "yellow"; oblast-wide permanent ones may have null);
- disabled entries carry no timestamps at all, so an END is only known to lie between two polls: it is logged as
  ended_between = [previous poll, this poll];
- a change of level during one alert (yellow -> red) is seen at the poll after it happened.

Output (demo/emap_log/, not committed):
- events.jsonl: one line per change (start, end, level change) with the poll times around it;
- polls.csv: one line per poll (time, HTTP status, counts of red / yellow raions);
- snapshot_<time>.json: the full feed at the first poll, so the state at the start of the log is known.

    python demo/emap_logger.py                 # runs until stopped (Ctrl+C)
    python demo/emap_logger.py --max-polls 3   # short test
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://vadimklimenko.com/map/statuses.json"
USER_AGENT = "alerts-forecast research logger, 1 request/min (github.com/Ddrakkx/alerts-forecast)"
MIN_INTERVAL_S = 60
MAX_BACKOFF_S = 600
OUT = Path(__file__).resolve().parent / "emap_log"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def flatten(feed: dict) -> dict:
    """{(oblast, raion or ''): {'enabled', 'enabled_at', 'alert_level'}} for every oblast and raion in the feed."""
    out = {}
    for oblast, s in feed.get("states", {}).items():
        out[(oblast, "")] = {"enabled": bool(s.get("enabled")), "enabled_at": s.get("enabled_at"), "alert_level": s.get("alert_level")}
        for raion, r in (s.get("districts") or {}).items():
            out[(oblast, raion)] = {"enabled": bool(r.get("enabled")), "enabled_at": r.get("enabled_at"), "alert_level": r.get("alert_level")}
    return out


def diff(prev: dict, cur: dict, prev_poll: str, poll: str) -> list:
    """Events between two flattened states. A key that disappears is treated as disabled, a new key as previously disabled."""
    events = []
    off = {"enabled": False, "enabled_at": None, "alert_level": None}
    for key in sorted(set(prev) | set(cur)):
        a, b = prev.get(key, off), cur.get(key, off)
        base = {"oblast": key[0], "raion": key[1]}
        if not a["enabled"] and b["enabled"]:
            events.append({**base, "event": "start", "started_at": b["enabled_at"], "alert_level": b["alert_level"],
                           "seen_between": [prev_poll, poll]})
        elif a["enabled"] and not b["enabled"]:
            events.append({**base, "event": "end", "started_at": a["enabled_at"], "alert_level": a["alert_level"],
                           "ended_between": [prev_poll, poll]})
        elif a["enabled"] and b["enabled"]:
            if a["enabled_at"] != b["enabled_at"]:  # ended and started again between two polls
                events.append({**base, "event": "end", "started_at": a["enabled_at"], "alert_level": a["alert_level"],
                               "ended_between": [prev_poll, poll]})
                events.append({**base, "event": "start", "started_at": b["enabled_at"], "alert_level": b["alert_level"],
                               "seen_between": [prev_poll, poll]})
            elif a["alert_level"] != b["alert_level"]:
                events.append({**base, "event": "level", "started_at": b["enabled_at"], "from": a["alert_level"],
                               "to": b["alert_level"], "changed_between": [prev_poll, poll]})
    return events


def counts(state: dict) -> dict:
    raions = [v for (o, r), v in state.items() if r and v["enabled"]]
    return {"red": sum(v["alert_level"] == "red" for v in raions), "yellow": sum(v["alert_level"] == "yellow" for v in raions),
            "other": sum(v["alert_level"] not in ("red", "yellow") for v in raions),
            "oblasts_on": sum(1 for (o, r), v in state.items() if not r and v["enabled"])}


class Fetcher:
    """Conditional GET: remembers ETag and Last-Modified, a 304 means "unchanged"."""

    def __init__(self, opener=urllib.request.urlopen):
        self.opener, self.etag, self.last_modified = opener, None, None

    def get(self):
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.etag:
            headers["If-None-Match"] = self.etag
        if self.last_modified:
            headers["If-Modified-Since"] = self.last_modified
        try:
            with self.opener(urllib.request.Request(URL, headers=headers), timeout=20) as resp:
                self.etag = resp.headers.get("ETag") or self.etag
                self.last_modified = resp.headers.get("Last-Modified") or self.last_modified
                return resp.status, json.loads(resp.read().decode("utf-8")), resp.headers.get("Last-Modified")
        except urllib.error.HTTPError as e:
            if e.code == 304:
                return 304, None, self.last_modified
            raise


def seconds_to_wait(polls_csv: Path, interval: int, now: datetime) -> float:
    """The minute also holds across restarts: wait until `interval` seconds have passed since the last poll in polls.csv."""
    if not polls_csv.exists():
        return 0.0
    lines = polls_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return 0.0
    try:
        last = datetime.fromisoformat(lines[-1].split(",")[0])
    except ValueError:
        return float(interval)  # unreadable: be safe
    return max(0.0, interval - (now - last).total_seconds())


def run(interval: int = MIN_INTERVAL_S, max_polls=None, out: Path = OUT, fetcher=None, sleep=time.sleep, clock=time.monotonic,
        now=lambda: datetime.now(timezone.utc)) -> None:
    if interval < MIN_INTERVAL_S:
        raise SystemExit(f"interval {interval} s is below the agreed minimum of {MIN_INTERVAL_S} s")
    out.mkdir(parents=True, exist_ok=True)
    wait = seconds_to_wait(out / "polls.csv", interval, now())
    if wait > 0:
        sleep(wait)
    fetcher = fetcher or Fetcher()
    events_f = open(out / "events.jsonl", "a", encoding="utf-8")
    polls_new = not (out / "polls.csv").exists()
    polls_f = open(out / "polls.csv", "a", encoding="utf-8", newline="")
    polls = csv.writer(polls_f)
    if polls_new:
        polls.writerow(["polled_at", "status", "last_modified", "red", "yellow", "other", "oblasts_on", "events"])
    state, prev_poll, n, backoff = None, None, 0, interval
    try:
        while max_polls is None or n < max_polls:
            started = clock()
            poll = now_utc()
            try:
                status, feed, last_mod = fetcher.get()
                backoff = interval
                if feed is not None:
                    cur = flatten(feed)
                    if state is None:
                        (out / f"snapshot_{poll.replace(':', '')}.json").write_text(json.dumps(feed, ensure_ascii=False), encoding="utf-8")
                        evs = []
                    else:
                        evs = diff(state, cur, prev_poll, poll)
                    for ev in evs:
                        events_f.write(json.dumps({"logged_at": poll, **ev}, ensure_ascii=False) + "\n")
                    state = cur
                else:
                    evs = []
                c = counts(state) if state else {"red": "", "yellow": "", "other": "", "oblasts_on": ""}
                polls.writerow([poll, status, last_mod or "", c["red"], c["yellow"], c["other"], c["oblasts_on"], len(evs)])
                prev_poll = poll
            except Exception as e:  # network trouble: log it and wait longer, never shorter than the interval
                polls.writerow([poll, f"error: {type(e).__name__}: {e}"[:200], "", "", "", "", "", 0])
                backoff = min(MAX_BACKOFF_S, backoff * 2)
            events_f.flush()
            polls_f.flush()
            n += 1
            if max_polls is not None and n >= max_polls:
                break
            sleep(max(0.0, backoff - (clock() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        events_f.close()
        polls_f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=MIN_INTERVAL_S, help="seconds between requests, at least 60")
    ap.add_argument("--max-polls", type=int, default=None)
    args = ap.parse_args()
    print(f"logging {URL} every {args.interval} s into {OUT}", flush=True)
    sys.exit(run(args.interval, args.max_polls))
