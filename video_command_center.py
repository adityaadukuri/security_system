"""
Video Command Center
====================
Processes a real video file through the security analysis pipeline and renders
a live command center UI with bounding boxes, dwell timers, zone overlays, and
a scrolling alert log.

Threats detected
----------------
  Loitering   : 20 s dwell in the same zone  (configurable via LOITERING_THRESHOLD_S)
  Brawl       : ≥2 persons within 150 px for ≥2 s
  Breaking-in : person enters a restricted zone (immediate alert)

Usage
-----
  python video_command_center.py ~/Downloads/loitering2.mov
  python video_command_center.py video.mp4 --display
  python video_command_center.py video.mp4 --zones "0.0,0.0,0.35,0.30;0.65,0.0,1.0,0.30"
  python video_command_center.py video.mp4 --skip 2   # process every Nth frame
"""

import argparse
import math
import subprocess
import sys
import time as _time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from unittest.mock import patch as _patch

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent))
from inference_engine import AlertDispatcher, Incident, LoiteringTracker

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
LOITERING_THRESHOLD_S  = 20       # flag after 20 s in same zone
LOITERING_ZONE_R_PX    = 80       # zone radius in display pixels
BRAWL_MIN_DURATION_S   = 2        # flag after 2 s of proximity
BRAWL_PROXIMITY_PX     = 150      # centroid distance (display pixels)
CONF_THRESHOLD         = 0.20
ALERT_COOLDOWN_S       = 30       # per-track per-threat (video time)
VIDEO_BASE_T           = 1_000_000.0  # base offset so rate limiter never false-blocks

# ── DISPLAY LAYOUT ────────────────────────────────────────────────────────────
TARGET_H  = 540       # resize video to this height for display
PANEL_W   = 380       # right-side panel width
STATUS_H  = 68        # bottom status bar height
MAX_LOG   = 12        # max alert entries visible in panel

# ── COLOURS (BGR) ─────────────────────────────────────────────────────────────
C: Dict[str, Tuple] = {
    "bg":       (20,  20,  30),
    "panel":    (12,  12,  22),
    "border":   (60,  60,  80),
    "header":   (30,  180, 220),
    "text":     (210, 210, 210),
    "dim":      (110, 110, 110),
    "ok":       (50,  200, 50),
    "loiter":   (0,   200, 255),
    "brawl":    (30,  60,  255),
    "breakin":  (120, 0,   255),
    "zone":     (100, 40,  200),
}


# ── BRAWL TRACKER ─────────────────────────────────────────────────────────────

class BrawlTracker:
    """Fires when ≥2 persons are within BRAWL_PROXIMITY_PX for ≥BRAWL_MIN_DURATION_S."""

    def __init__(self) -> None:
        self._since: Optional[float] = None

    def update(
        self,
        centroids: List[Tuple[float, float]],
        video_t: float,
    ) -> float:
        """Returns seconds the brawl cluster has been active (0.0 if none)."""
        active = False
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                if math.sqrt(dx * dx + dy * dy) < BRAWL_PROXIMITY_PX:
                    active = True
                    break
            if active:
                break

        if not active:
            self._since = None
            return 0.0

        if self._since is None:
            self._since = video_t
        return video_t - self._since


# ── BREAKING-IN TRACKER ───────────────────────────────────────────────────────

class BreakingInTracker:
    """Fires once per track_id the first time it enters any restricted zone."""

    def __init__(self, zones: List[Tuple[float, float, float, float]]) -> None:
        self._zones = zones          # absolute pixels (x1,y1,x2,y2)
        self._entered: Set[int] = set()

    def check(self, track_id: int, cx: float, cy: float) -> bool:
        """Returns True the first time this track enters a zone."""
        inside = any(
            z[0] <= cx <= z[2] and z[1] <= cy <= z[3]
            for z in self._zones
        )
        if inside and track_id not in self._entered:
            self._entered.add(track_id)
            return True
        if not inside:
            self._entered.discard(track_id)
        return False

    @property
    def zones(self) -> List[Tuple[float, float, float, float]]:
        return self._zones


# ── ALERT RATE LIMITER ────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, cooldown_s: float = ALERT_COOLDOWN_S) -> None:
        self._last: Dict[str, float] = {}
        self._cooldown = cooldown_s

    def allow(self, key: str, now: float) -> bool:
        if now - self._last.get(key, 0.0) >= self._cooldown:
            self._last[key] = now
            return True
        return False


# ── UI HELPERS ────────────────────────────────────────────────────────────────

def _put(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Optional[Tuple] = None,
    scale: float = 0.45,
    thickness: int = 1,
) -> None:
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        color or C["text"], thickness, cv2.LINE_AA,
    )


def _sep(panel: np.ndarray, y: int) -> None:
    cv2.line(panel, (8, y), (PANEL_W - 8, y), C["border"], 1)


def _fmt_t(secs: float) -> str:
    return str(timedelta(seconds=int(max(secs, 0))))


def _alert_color(threat: str) -> Tuple:
    return {"loitering": C["loiter"], "brawl": C["brawl"],
            "breaking_in": C["breakin"]}.get(threat, C["text"])


# ── PANEL RENDERER ────────────────────────────────────────────────────────────

def _render_panel(
    h: int,
    video_name: str,
    video_t: float,
    total_t: float,
    active_tracks: int,
    loiter_total: int,
    brawl_total: int,
    breakin_total: int,
    alert_log: deque,
) -> np.ndarray:
    panel = np.full((h, PANEL_W, 3), C["panel"], dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, h), C["border"], 2)

    y = 18
    _put(panel, "COMMAND CENTER", 10, y, C["header"], 0.58, 2)
    y += 8;  _sep(panel, y);  y += 14

    # Source + time
    name = Path(video_name).name
    _put(panel, f"Source : {name[:26]}", 10, y, C["dim"], 0.38);  y += 16
    _put(panel, f"Time   : {_fmt_t(video_t)} / {_fmt_t(total_t)}", 10, y, C["text"], 0.42); y += 14

    # Progress bar
    if total_t > 0:
        pw = PANEL_W - 20
        prog = int(pw * min(video_t / total_t, 1.0))
        cv2.rectangle(panel, (10, y), (10 + pw, y + 7), C["border"], -1)
        cv2.rectangle(panel, (10, y), (10 + prog, y + 7), C["header"], -1)
    y += 16;  _sep(panel, y);  y += 12

    # Active tracks
    tc = C["ok"] if active_tracks else C["dim"]
    _put(panel, f"Active Tracks : {active_tracks}", 10, y, tc, 0.45); y += 16
    _sep(panel, y);  y += 12

    # Threat counters
    _put(panel, "THREAT COUNTS", 10, y, C["header"], 0.44, 1);  y += 18
    for label, count, color in (
        ("Loitering", loiter_total,  C["loiter"]),
        ("Brawl",     brawl_total,   C["brawl"]),
        ("Break-in",  breakin_total, C["breakin"]),
    ):
        dot_c = color if count else C["dim"]
        cv2.circle(panel, (18, y - 4), 5, dot_c, -1)
        _put(panel, f"{label:<12} {count:>3}", 28, y, dot_c, 0.42)
        y += 18
    _sep(panel, y);  y += 12

    # Alert log
    _put(panel, "RECENT ALERTS", 10, y, C["header"], 0.44, 1);  y += 16
    max_entries = (h - y - 8) // 17
    for entry in list(alert_log)[-max_entries:]:
        color = _alert_color(entry["threat"])
        ts    = _fmt_t(entry["video_t"])
        label = f"[{ts}] {entry['threat'].upper()[:10]}"
        if entry["track_id"] is not None:
            label += f" ID:{entry['track_id']}"
        _put(panel, label[:36], 10, y, color, 0.38)
        y += 17
        if y > h - 8:
            break

    return panel


# ── STATUS BAR ────────────────────────────────────────────────────────────────

def _render_status(
    w: int,
    loiter_total: int,
    brawl_total: int,
    breakin_total: int,
    fps_actual: float,
) -> np.ndarray:
    bar = np.full((STATUS_H, w, 3), C["bg"], dtype=np.uint8)
    cv2.line(bar, (0, 0), (w, 0), C["border"], 1)

    y = 22
    _put(bar, "STATUS:", 10, y, C["dim"], 0.44)
    x = 80
    for label, count, color in (
        ("LOITERING", loiter_total,  C["loiter"]),
        ("BRAWL",     brawl_total,   C["brawl"]),
        ("BREAK-IN",  breakin_total, C["breakin"]),
    ):
        c = color if count else C["dim"]
        cv2.rectangle(bar, (x, y - 14), (x + 150, y + 6), (30, 30, 45), -1)
        cv2.rectangle(bar, (x, y - 14), (x + 150, y + 6), c, 1)
        _put(bar, f"{label}: {count}", x + 6, y, c, 0.40)
        x += 160
    _put(bar, f"FPS: {fps_actual:.1f}", x + 20, y, C["dim"], 0.40)

    y = 50
    thr = (f"Loitering ≥{LOITERING_THRESHOLD_S}s  |  "
           f"Brawl ≥{BRAWL_MIN_DURATION_S}s within {BRAWL_PROXIMITY_PX}px  |  "
           f"Break-in: immediate")
    _put(bar, thr, 10, y, C["dim"], 0.36)
    return bar


# ── VIDEO FRAME ANNOTATIONS ───────────────────────────────────────────────────

def _annotate_video(
    frame: np.ndarray,
    persons: List[dict],
    loiter_map: Dict[int, float],
    brawl_s: float,
    zones: List[Tuple],
    fired_threats: Set[str],
) -> None:
    # Restricted zone overlays
    for z in zones:
        x1, y1, x2, y2 = int(z[0]), int(z[1]), int(z[2]), int(z[3])
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), C["zone"], -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C["zone"], 2)
        _put(frame, "RESTRICTED ZONE", x1 + 4, y1 + 16, C["zone"], 0.42)

    for p in persons:
        x1, y1, x2, y2 = (int(v) for v in p["bbox"])
        tid   = p["track_id"]
        conf  = p["conf"]
        dwell = loiter_map.get(tid, 0.0)

        if dwell >= LOITERING_THRESHOLD_S:
            color = C["loiter"]
        elif p.get("breakin"):
            color = C["breakin"]
        elif brawl_s >= BRAWL_MIN_DURATION_S:
            color = C["brawl"]
        else:
            color = C["ok"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        lbl = f"ID:{tid}  {conf:.2f}"
        if dwell > 0:
            lbl += f"  {dwell:.0f}s"
        _put(frame, lbl, x1, max(y1 - 6, 12), color, 0.40)

        # Dwell bar
        if 0 < dwell:
            bw     = max(x2 - x1, 1)
            filled = int(bw * min(dwell / LOITERING_THRESHOLD_S, 1.0))
            cv2.rectangle(frame, (x1, y2 + 2), (x2,       y2 + 8), (50, 50, 50), -1)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 8), C["loiter"], -1)

    # Brawl banner
    if brawl_s >= BRAWL_MIN_DURATION_S:
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (0, fh - 34), (fw, fh), (0, 0, 170), -1)
        _put(frame, f"!! BRAWL DETECTED — {brawl_s:.1f}s !!",
             10, fh - 10, (255, 255, 255), 0.60, 2)

    # Alert banners at top
    top_y = 0
    for threat, color, label in (
        ("loitering",   C["loiter"],  "LOITERING ALERT"),
        ("breaking_in", C["breakin"], "BREAKING-IN ALERT"),
        ("brawl",       C["brawl"],   "BRAWL ALERT"),
    ):
        if threat in fired_threats:
            fw = frame.shape[1]
            cv2.rectangle(frame, (0, top_y), (fw, top_y + 24), color, -1)
            _put(frame, f"{label} DISPATCHED",
                 8, top_y + 17, (0, 0, 0), 0.50, 2)
            top_y += 26


# ── ZONE PARSER ───────────────────────────────────────────────────────────────

def _parse_zones(zones_str: str, w: int, h: int) -> List[Tuple]:
    """Parse 'x1%,y1%,x2%,y2%;...' fractions into absolute pixel tuples."""
    result = []
    for seg in zones_str.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        parts = [float(v) for v in seg.split(",")]
        if len(parts) != 4:
            print(f"  [WARN] Skipping malformed zone: '{seg}'")
            continue
        result.append((
            parts[0] * w, parts[1] * h,
            parts[2] * w, parts[3] * h,
        ))
    return result


# ── INCIDENT BUILDER ─────────────────────────────────────────────────────────

def _make_incident(
    camera_id: str,
    threat: str,
    severity: str,
    conf: float,
    video_t: float,
    bbox: List[float],
    track_id: Optional[int],
) -> Incident:
    ts = _fmt_t(video_t)
    return Incident(
        id          = f"{camera_id}_{threat}_{int(VIDEO_BASE_T + video_t)}",
        camera_id   = camera_id,
        threat_type = threat,
        severity    = severity,
        confidence  = conf,
        timestamp   = f"video+{ts}",
        bbox        = bbox,
        track_id    = track_id,
    )


# ── PER-FRAME HELPERS (extracted to keep run() complexity low) ────────────────

def _extract_persons(results) -> Tuple[List[dict], Set[int]]:
    """Pull person detections from YOLO results."""
    persons: List[dict] = []
    active_ids: Set[int] = set()
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            if int(box.cls[0]) != 0:          # person class only
                continue
            tid  = int(box.id[0]) if box.id is not None else None
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            persons.append({"bbox": bbox, "track_id": tid,
                             "conf": conf, "breakin": False})
            if tid is not None:
                active_ids.add(tid)
    return persons, active_ids


def _centroid(p: dict) -> Tuple[float, float]:
    return (p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2


def _run_loitering(
    persons: List[dict],
    loitering: LoiteringTracker,
    sim_t: float,
    video_t: float,
    rate_limiter: RateLimiter,
    dispatcher: AlertDispatcher,
    alert_log: deque,
    active_ids: Set[int],
) -> Tuple[Dict[int, float], int]:
    new_map: Dict[int, float] = {}
    count = 0
    for p in persons:
        tid = p["track_id"]
        if tid is None:
            continue
        cx, cy = _centroid(p)
        with _patch("inference_engine.time.time", return_value=sim_t):
            dwell = loitering.update(tid, cx, cy)
        new_map[tid] = dwell
        if dwell >= LOITERING_THRESHOLD_S and rate_limiter.allow(f"{tid}:loitering", video_t):
            inc = _make_incident("CAM-01", "loitering", "medium",
                                  p["conf"], video_t, p["bbox"], tid)
            dispatcher.dispatch(inc)
            alert_log.append({"video_t": video_t, "threat": "loitering", "track_id": tid})
            count += 1
    loitering.evict_stale(active_ids)
    return new_map, count


def _run_brawl(
    persons: List[dict],
    brawl_tracker: BrawlTracker,
    video_t: float,
    rate_limiter: RateLimiter,
    dispatcher: AlertDispatcher,
    alert_log: deque,
    disp_w: int,
    disp_h: int,
) -> Tuple[float, int]:
    centroids = [_centroid(p) for p in persons if p["track_id"] is not None]
    brawl_s   = brawl_tracker.update(centroids, video_t)
    count     = 0
    if brawl_s >= BRAWL_MIN_DURATION_S and rate_limiter.allow("cluster:brawl", video_t):
        conf = max((p["conf"] for p in persons), default=0.5)
        inc  = _make_incident("CAM-01", "brawl", "high", conf,
                               video_t, [0, 0, disp_w, disp_h], None)
        dispatcher.dispatch(inc)
        alert_log.append({"video_t": video_t, "threat": "brawl", "track_id": None})
        count += 1
    return brawl_s, count


def _run_breaking_in(
    persons: List[dict],
    breakin_tracker: BreakingInTracker,
    video_t: float,
    rate_limiter: RateLimiter,
    dispatcher: AlertDispatcher,
    alert_log: deque,
) -> int:
    count = 0
    for p in persons:
        tid = p["track_id"]
        if tid is None:
            continue
        cx, cy = _centroid(p)
        if breakin_tracker.check(tid, cx, cy) and rate_limiter.allow(f"{tid}:breaking_in", video_t):
            inc = _make_incident("CAM-01", "breaking_in", "critical",
                                  p["conf"], video_t, p["bbox"], tid)
            dispatcher.dispatch(inc)
            alert_log.append({"video_t": video_t, "threat": "breaking_in", "track_id": tid})
            p["breakin"] = True
            count += 1
    return count


def _active_threats(persons: List[dict], loiter_map: Dict[int, float],
                    brawl_s: float) -> Set[str]:
    fired: Set[str] = set()
    if any(loiter_map.get(p["track_id"], 0) >= LOITERING_THRESHOLD_S
           for p in persons if p["track_id"] is not None):
        fired.add("loitering")
    if brawl_s >= BRAWL_MIN_DURATION_S:
        fired.add("brawl")
    if any(p.get("breakin") for p in persons):
        fired.add("breaking_in")
    return fired


def _print_summary(out_path: Path, proc_count: int, total_frames: int,
                   elapsed: float, loiter_total: int, brawl_total: int,
                   breakin_total: int, alert_log: deque) -> None:
    fps_out = proc_count / elapsed if elapsed > 0 else 0.0
    print("\n" + "=" * 64)
    print("  PROCESSING COMPLETE")
    print("=" * 64)
    print(f"  Frames processed  : {proc_count} / {total_frames}")
    print(f"  Processing speed  : {fps_out:.1f} fps")
    print(f"  Output video      : {out_path}")
    print()
    print(f"  Loitering alerts  : {loiter_total}")
    print(f"  Brawl alerts      : {brawl_total}")
    print(f"  Breaking-in alerts: {breakin_total}")
    print(f"  Total incidents   : {loiter_total + brawl_total + breakin_total}")
    if alert_log:
        print()
        print("  Alert timeline:")
        for entry in alert_log:
            ts      = _fmt_t(entry["video_t"])
            tid_str = f" ID:{entry['track_id']}" if entry["track_id"] else ""
            print(f"    [{ts}] {entry['threat'].upper()}{tid_str}")
    print()
    print(f"  Opening {out_path} ...")
    subprocess.Popen(["open", str(out_path)])


# ── MAIN PROCESSING LOOP ──────────────────────────────────────────────────────

def run(video_path: str, zones_str: str, display: bool, skip: int) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        sys.exit(1)

    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_t      = total_frames / src_fps
    disp_w       = int(src_w * TARGET_H / src_h)
    canvas_w     = disp_w + PANEL_W
    canvas_h     = TARGET_H + STATUS_H

    # Default zone: left wall where building storefront/window typically sits
    zones = _parse_zones(zones_str, disp_w, TARGET_H) if zones_str else [
        (disp_w * 0.08, 0, disp_w * 0.38, TARGET_H * 1.0),
    ]

    print("\n" + "=" * 64)
    print("  VIDEO COMMAND CENTER")
    print("=" * 64)
    print(f"  Source    : {video_path}")
    print(f"  Resolution: {src_w}x{src_h}  →  display {disp_w}x{TARGET_H}")
    print(f"  Duration  : {_fmt_t(total_t)}  ({total_frames} frames @ {src_fps:.1f}fps)")
    print(f"  Zones     : {len(zones)} restricted zone(s)")
    for i, z in enumerate(zones):
        print(f"    Zone {i+1}: ({z[0]:.0f},{z[1]:.0f}) → ({z[2]:.0f},{z[3]:.0f}) px")
    print()

    print("Loading YOLOv8n ...")
    model = YOLO("yolov8n.pt")
    print("Model ready.\n")

    Path("demo_output").mkdir(exist_ok=True)
    Path("incidents").mkdir(exist_ok=True)
    out_path = Path("demo_output") / "command_center_output.mp4"
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        min(src_fps / max(skip, 1), 30.0), (canvas_w, canvas_h),
    )

    loitering       = LoiteringTracker()
    brawl_tracker   = BrawlTracker()
    breakin_tracker = BreakingInTracker(zones)
    dispatcher      = AlertDispatcher()
    rate_limiter    = RateLimiter(ALERT_COOLDOWN_S)
    alert_log: deque = deque(maxlen=50)

    loiter_map: Dict[int, float] = {}
    loiter_total = brawl_total = breakin_total = 0
    frame_idx = proc_count = 0
    t0         = _time.time()
    fps_actual = 0.0

    print(f"{'Frame':>7}  {'Time':>8}  {'Tracks':>7}  {'Dwell(max)':>11}  "
          f"{'BrawlS':>7}  {'Alerts':>8}")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % skip != 0:
            continue
        proc_count += 1

        video_t = frame_idx / src_fps
        sim_t   = VIDEO_BASE_T + video_t
        small   = cv2.resize(frame, (disp_w, TARGET_H), interpolation=cv2.INTER_AREA)

        results = model.track(small, persist=True, conf=CONF_THRESHOLD,
                               verbose=False, tracker="bytetrack.yaml")
        persons, active_ids = _extract_persons(results)

        new_map, lc  = _run_loitering(persons, loitering, sim_t, video_t,
                                       rate_limiter, dispatcher, alert_log, active_ids)
        loiter_map    = new_map
        loiter_total += lc

        brawl_s, bc  = _run_brawl(persons, brawl_tracker, video_t,
                                   rate_limiter, dispatcher, alert_log, disp_w, TARGET_H)
        brawl_total += bc

        bic           = _run_breaking_in(persons, breakin_tracker, video_t,
                                          rate_limiter, dispatcher, alert_log)
        breakin_total += bic

        fired_threats = _active_threats(persons, loiter_map, brawl_s)
        _annotate_video(small, persons, loiter_map, brawl_s, zones, fired_threats)

        panel  = _render_panel(TARGET_H, video_path, video_t, total_t, len(active_ids),
                                loiter_total, brawl_total, breakin_total, alert_log)
        status = _render_status(canvas_w, loiter_total, brawl_total, breakin_total, fps_actual)
        canvas = np.full((canvas_h, canvas_w, 3), C["bg"], dtype=np.uint8)
        canvas[:TARGET_H, :disp_w]           = small
        canvas[:TARGET_H, disp_w:]           = panel
        canvas[TARGET_H:TARGET_H + STATUS_H, :] = status
        writer.write(canvas)

        if display:
            cv2.imshow("Command Center", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[User quit]")
                break

        elapsed    = _time.time() - t0
        fps_actual = proc_count / elapsed if elapsed > 0 else 0.0
        if proc_count % 50 == 1 or proc_count == 1:
            max_dwell = max(loiter_map.values(), default=0.0)
            total_alerts = loiter_total + brawl_total + breakin_total
            print(f"{frame_idx:>7}  t={video_t:>6.1f}s  tracks={len(active_ids):>3}  "
                  f"dwell={max_dwell:>7.1f}s  brawl={brawl_s:>5.1f}s  "
                  f"alerts={total_alerts:>4}")

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    _print_summary(out_path, proc_count, total_frames, _time.time() - t0,
                   loiter_total, brawl_total, breakin_total, alert_log)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Command Center")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--display", action="store_true",
                        help="Show live OpenCV window while processing")
    parser.add_argument(
        "--zones", default="",
        help=("Semicolon-separated restricted zones as normalised fractions "
              "'x1,y1,x2,y2' e.g. '0.0,0.0,0.35,1.0;0.65,0.0,1.0,1.0'. "
              "Default: left-wall zone (storefront/window area)."),
    )
    parser.add_argument("--skip", type=int, default=2,
                        help="Process every Nth frame (default 2)")
    args = parser.parse_args()
    run(args.video, args.zones, args.display, max(args.skip, 1))
