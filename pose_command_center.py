"""
Pose-Based Security Command Center
====================================
Uses YOLOv8n-pose to extract 17-point body skeletons, then applies
rule-based activity classification to detect threats — no predefined
zone boxes required.

How it works
------------
  1. YOLOv8n-pose detects each person and outputs 17 body keypoints.
  2. PoseActivityClassifier analyses posture + motion history per track:
       - torso orientation  → climbing / upright
       - wrist velocity     → striking / fighting
       - body velocity      → running / idle
       - limb compression   → crouching
  3. Activities are mapped to threats:
       climbing / striking  →  BREAKING-IN  (critical, immediate)
       fighting             →  BRAWL        (high, ≥ 2 s)
       standing idle ≥ 20 s →  LOITERING    (medium)
       crouching            →  SUSPICIOUS   (low)

Model: yolov8n-pose.pt  (auto-downloads ~6 MB from Ultralytics Hub)

Usage
-----
  python pose_command_center.py ~/Downloads/loitering2.mov
  python pose_command_center.py video.mp4 --display
  python pose_command_center.py video.mp4 --skip 2
"""

import argparse
import math
import subprocess
import sys
import time as _time
from collections import deque
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, List, Optional, Set, Tuple
from unittest.mock import patch as _patch

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent))
from inference_engine import AlertDispatcher, Incident, LoiteringTracker


# ── COCO-17 KEYPOINT INDICES ──────────────────────────────────────────────────
KP = SimpleNamespace(
    NOSE=0,
    L_EYE=1,  R_EYE=2,
    L_EAR=3,  R_EAR=4,
    L_SHOULDER=5,  R_SHOULDER=6,
    L_ELBOW=7,     R_ELBOW=8,
    L_WRIST=9,     R_WRIST=10,
    L_HIP=11,      R_HIP=12,
    L_KNEE=13,     R_KNEE=14,
    L_ANKLE=15,    R_ANKLE=16,
)

SKELETON_PAIRS: List[Tuple[int, int]] = [
    (KP.NOSE, KP.L_SHOULDER),   (KP.NOSE, KP.R_SHOULDER),
    (KP.L_SHOULDER, KP.R_SHOULDER),
    (KP.L_SHOULDER, KP.L_ELBOW), (KP.L_ELBOW, KP.L_WRIST),
    (KP.R_SHOULDER, KP.R_ELBOW), (KP.R_ELBOW, KP.R_WRIST),
    (KP.L_SHOULDER, KP.L_HIP),   (KP.R_SHOULDER, KP.R_HIP),
    (KP.L_HIP, KP.R_HIP),
    (KP.L_HIP, KP.L_KNEE),       (KP.L_KNEE, KP.L_ANKLE),
    (KP.R_HIP, KP.R_KNEE),       (KP.R_KNEE, KP.R_ANKLE),
]


# ── THRESHOLDS ────────────────────────────────────────────────────────────────
LOITERING_THRESHOLD_S = 20       # seconds idle before loitering alert
BRAWL_MIN_S           = 2        # seconds of fighting before brawl alert
CONF_THRESHOLD        = 0.20     # YOLO detection confidence
KP_CONF_THRESH        = 0.30     # min keypoint confidence to use
POSE_HISTORY_LEN      = 20       # frames of pose history per track
TORSO_CLIMB_THRESH    = 0.38     # torso vert-ratio below this → climbing
WRIST_FIGHT_VEL       = 6.0      # px / frame — fighting threshold
WRIST_STRIKE_VEL      = 10.0     # px / frame — striking / smashing threshold
BODY_RUN_VEL          = 10.0     # px / frame — running threshold
BODY_IDLE_VEL         = 2.0      # px / frame — standing / idle threshold
ALERT_COOLDOWN_S      = 30       # rate-limit per track per threat (video time)
VIDEO_BASE_T          = 1_000_000.0  # base offset keeps rate limiter from false-blocking

# ── DISPLAY LAYOUT ────────────────────────────────────────────────────────────
TARGET_H = 540
PANEL_W  = 400
STATUS_H = 72
MAX_LOG  = 12

# ── COLOURS (BGR) ─────────────────────────────────────────────────────────────
C: Dict[str, Tuple] = {
    "bg":          (20,  20,  30),
    "panel":       (12,  12,  22),
    "border":      (60,  60,  80),
    "header":      (30,  180, 220),
    "text":        (210, 210, 210),
    "dim":         (110, 110, 110),
    # activity colours
    "walking":     (50,  200, 50),
    "standing":    (80,  200, 80),
    "running":     (0,   210, 210),
    "crouching":   (0,   165, 255),
    "fighting":    (30,  60,  255),
    "striking":    (0,   0,   220),
    "climbing":    (200, 0,   255),
    # threat colours
    "loitering":   (0,   200, 255),
    "brawl":       (30,  60,  255),
    "breaking_in": (200, 0,   255),
    "suspicious":  (0,   165, 255),
}

# Activity → (threat_type, severity) — None = no immediate alert
ACTIVITY_THREAT: Dict[str, Optional[Tuple[str, str]]] = {
    "climbing":  ("breaking_in", "critical"),
    "striking":  ("breaking_in", "critical"),
    "fighting":  ("brawl",       "high"),
    "crouching": ("suspicious",  "low"),
    "standing":  (None,          None),   # handled by LoiteringTracker
    "running":   (None,          None),
    "walking":   (None,          None),
}

ACTIVITY_LABEL = {
    "walking":   "Walking",
    "standing":  "Standing",
    "running":   "Running",
    "crouching": "Crouching",
    "fighting":  "FIGHTING",
    "striking":  "STRIKING",
    "climbing":  "CLIMBING",
}


# ── ACTIVITY RESULT ───────────────────────────────────────────────────────────

class ActivityResult:
    __slots__ = ("activity", "confidence", "signals")

    def __init__(self, activity: str, confidence: float, signals: dict) -> None:
        self.activity   = activity
        self.confidence = confidence
        self.signals    = signals


# ── KEYPOINT HELPERS ──────────────────────────────────────────────────────────

def _kp(xy: np.ndarray, conf: Optional[np.ndarray], idx: int
        ) -> Optional[Tuple[float, float]]:
    """Return (x, y) if keypoint confidence ≥ threshold, else None."""
    if conf is not None and float(conf[idx]) < KP_CONF_THRESH:
        return None
    return float(xy[idx, 0]), float(xy[idx, 1])


def _mid(a: Optional[Tuple], b: Optional[Tuple]) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return None
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


# ── POSE ACTIVITY CLASSIFIER ──────────────────────────────────────────────────

class PoseActivityClassifier:
    """
    Rule-based activity classifier from COCO-17 skeleton keypoints.

    Signals computed per frame
    --------------------------
    torso_vert   0..1   Ratio of vertical to total torso extent.
                        1.0 = fully upright, 0.0 = horizontal (climbing).
    arms_raised  bool   Average wrist Y above average shoulder Y.
    crouching    bool   Hip-to-ankle distance compressed vs torso height.
    wrist_vel    float  Max wrist speed over last 5 frames (px/frame).
    body_vel     float  Hip centroid speed over last 5 frames (px/frame).

    Classification priority
    -----------------------
    climbing  > striking  > fighting  > running  > crouching  > standing  > walking
    """

    def __init__(self) -> None:
        self._hist: Dict[int, Deque[Tuple[np.ndarray, Optional[np.ndarray]]]] = {}

    def update(
        self,
        tid: int,
        kps_xy: np.ndarray,          # (17, 2)
        kps_conf: Optional[np.ndarray],  # (17,) or None
    ) -> ActivityResult:
        if tid not in self._hist:
            self._hist[tid] = deque(maxlen=POSE_HISTORY_LEN)
        snap_conf = kps_conf.copy() if kps_conf is not None else None
        self._hist[tid].append((kps_xy.copy(), snap_conf))
        signals = self._compute_signals(tid, kps_xy, kps_conf)
        return self._classify(signals)

    # ── signal helpers ────────────────────────────────────────────────────────

    def _compute_signals(
        self, tid: int, xy: np.ndarray, conf: Optional[np.ndarray]
    ) -> dict:
        return {
            "torso_vert":  self._torso_vert(xy, conf),
            "arms_raised": self._arms_raised(xy, conf),
            "crouching":   self._is_crouching(xy, conf),
            "wrist_vel":   self._wrist_vel(tid),
            "body_vel":    self._body_vel(tid),
        }

    def _torso_vert(self, xy: np.ndarray, conf: Optional[np.ndarray]
                    ) -> Optional[float]:
        sh = _mid(_kp(xy, conf, KP.L_SHOULDER), _kp(xy, conf, KP.R_SHOULDER))
        hi = _mid(_kp(xy, conf, KP.L_HIP),      _kp(xy, conf, KP.R_HIP))
        if sh is None or hi is None:
            return None
        dx = abs(sh[0] - hi[0])
        dy = abs(sh[1] - hi[1])
        return dy / (dx + dy + 1e-6)

    def _arms_raised(self, xy: np.ndarray, conf: Optional[np.ndarray]) -> bool:
        sh = _mid(_kp(xy, conf, KP.L_SHOULDER), _kp(xy, conf, KP.R_SHOULDER))
        wr = _mid(_kp(xy, conf, KP.L_WRIST),    _kp(xy, conf, KP.R_WRIST))
        if sh is None or wr is None:
            return False
        return wr[1] < sh[1]   # y↓ in image: smaller y = higher in frame

    def _is_crouching(self, xy: np.ndarray, conf: Optional[np.ndarray]) -> bool:
        sh = _mid(_kp(xy, conf, KP.L_SHOULDER), _kp(xy, conf, KP.R_SHOULDER))
        hi = _mid(_kp(xy, conf, KP.L_HIP),      _kp(xy, conf, KP.R_HIP))
        an = _mid(_kp(xy, conf, KP.L_ANKLE),    _kp(xy, conf, KP.R_ANKLE))
        if sh is None or hi is None or an is None:
            return False
        torso_h = abs(sh[1] - hi[1]) + 1e-6
        leg_h   = abs(hi[1] - an[1])
        return (leg_h / torso_h) < 0.85   # legs compressed vs torso

    def _wrist_vel(self, tid: int) -> float:
        hist = list(self._hist.get(tid, []))
        if len(hist) < 5:
            return 0.0
        old_xy, old_c = hist[-5]
        new_xy, new_c = hist[-1]
        vels = []
        for idx in (KP.L_WRIST, KP.R_WRIST):
            p0 = _kp(old_xy, old_c, idx)
            p1 = _kp(new_xy, new_c, idx)
            if p0 and p1:
                vels.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 4)
        return max(vels, default=0.0)

    def _body_vel(self, tid: int) -> float:
        hist = list(self._hist.get(tid, []))
        if len(hist) < 5:
            return 0.0
        old_xy, old_c = hist[-5]
        new_xy, new_c = hist[-1]
        p0 = _mid(_kp(old_xy, old_c, KP.L_HIP), _kp(old_xy, old_c, KP.R_HIP))
        p1 = _mid(_kp(new_xy, new_c, KP.L_HIP), _kp(new_xy, new_c, KP.R_HIP))
        if p0 is None or p1 is None:
            return 0.0
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 4

    # ── classifier ────────────────────────────────────────────────────────────

    def _classify(self, s: dict) -> ActivityResult:
        tv = s.get("torso_vert")
        ar = s.get("arms_raised", False)
        wv = s.get("wrist_vel",   0.0)
        bv = s.get("body_vel",    0.0)
        cr = s.get("crouching",   False)

        if tv is not None and tv < TORSO_CLIMB_THRESH:
            return ActivityResult("climbing",  0.82, s)
        if wv >= WRIST_STRIKE_VEL:
            return ActivityResult("striking",  0.78, s)
        if wv >= WRIST_FIGHT_VEL and ar:
            return ActivityResult("fighting",  0.75, s)
        if bv >= BODY_RUN_VEL:
            return ActivityResult("running",   0.72, s)
        if cr:
            return ActivityResult("crouching", 0.68, s)
        if bv < BODY_IDLE_VEL:
            return ActivityResult("standing",  0.82, s)
        return ActivityResult("walking",   0.65, s)

    def evict_stale(self, active_ids: Set[int]) -> None:
        for tid in list(self._hist.keys()):
            if tid not in active_ids:
                del self._hist[tid]


# ── SKELETON RENDERER ─────────────────────────────────────────────────────────

def _draw_skeleton(
    frame: np.ndarray,
    xy: np.ndarray,
    conf: Optional[np.ndarray],
    color: Tuple,
) -> None:
    for a, b in SKELETON_PAIRS:
        pa = _kp(xy, conf, a)
        pb = _kp(xy, conf, b)
        if pa and pb:
            cv2.line(frame,
                     (int(pa[0]), int(pa[1])),
                     (int(pb[0]), int(pb[1])),
                     color, 2, cv2.LINE_AA)
    for idx in range(17):
        p = _kp(xy, conf, idx)
        if p:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (240, 240, 240), -1)
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, color, 1)


# ── RATE LIMITER ──────────────────────────────────────────────────────────────

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

def _put(img, text, x, y, color=None, scale=0.45, thickness=1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color or C["text"], thickness, cv2.LINE_AA)


def _sep(panel: np.ndarray, y: int) -> None:
    cv2.line(panel, (8, y), (PANEL_W - 8, y), C["border"], 1)


def _fmt_t(secs: float) -> str:
    return str(timedelta(seconds=int(max(secs, 0))))


def _render_panel(
    h: int,
    video_name: str,
    video_t: float,
    total_t: float,
    active_tracks: int,
    activity_counts: Dict[str, int],
    loiter_total: int,
    brawl_total: int,
    breakin_total: int,
    suspicious_total: int,
    alert_log: deque,
) -> np.ndarray:
    panel = np.full((h, PANEL_W, 3), C["panel"], dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, h), C["border"], 2)

    y = 18
    _put(panel, "POSE COMMAND CENTER", 10, y, C["header"], 0.55, 2)
    y += 8;  _sep(panel, y);  y += 14

    _put(panel, f"Source : {Path(video_name).name[:26]}", 10, y, C["dim"], 0.38); y += 16
    _put(panel, f"Time   : {_fmt_t(video_t)} / {_fmt_t(total_t)}", 10, y, C["text"], 0.42); y += 14

    if total_t > 0:
        pw   = PANEL_W - 20
        prog = int(pw * min(video_t / total_t, 1.0))
        cv2.rectangle(panel, (10, y), (10 + pw, y + 7), C["border"], -1)
        cv2.rectangle(panel, (10, y), (10 + prog, y + 7), C["header"], -1)
    y += 16;  _sep(panel, y);  y += 12

    tc = C["walking"] if active_tracks else C["dim"]
    _put(panel, f"Active Tracks : {active_tracks}", 10, y, tc, 0.45);  y += 16
    _sep(panel, y);  y += 12

    # Activity breakdown
    _put(panel, "ACTIVITY BREAKDOWN", 10, y, C["header"], 0.44, 1);  y += 18
    for act in ("walking", "standing", "running", "crouching", "fighting",
                "striking", "climbing"):
        cnt = activity_counts.get(act, 0)
        col = C.get(act, C["text"]) if cnt else C["dim"]
        cv2.circle(panel, (18, y - 4), 4, col, -1)
        _put(panel, f"{ACTIVITY_LABEL.get(act, act):<14} {cnt:>2}", 28, y, col, 0.40)
        y += 16
    _sep(panel, y);  y += 12

    # Threat counts
    _put(panel, "THREATS DETECTED", 10, y, C["header"], 0.44, 1);  y += 18
    for label, count, tkey in (
        ("Breaking-in", breakin_total,  "breaking_in"),
        ("Brawl",       brawl_total,    "brawl"),
        ("Loitering",   loiter_total,   "loitering"),
        ("Suspicious",  suspicious_total, "suspicious"),
    ):
        col = C.get(tkey, C["text"]) if count else C["dim"]
        cv2.circle(panel, (18, y - 4), 5, col, -1)
        _put(panel, f"{label:<14} {count:>3}", 28, y, col, 0.42)
        y += 18
    _sep(panel, y);  y += 12

    # Alert log
    _put(panel, "RECENT ALERTS", 10, y, C["header"], 0.44, 1);  y += 16
    max_entries = max((h - y - 8) // 17, 0)
    for entry in list(alert_log)[-max_entries:]:
        col   = C.get(entry["threat"], C["text"])
        ts    = _fmt_t(entry["video_t"])
        act   = entry.get("activity", "")
        tidst = f" ID:{entry['track_id']}" if entry["track_id"] is not None else ""
        line  = f"[{ts}] {entry['threat'].upper()[:10]}{tidst} ({act})"
        _put(panel, line[:38], 10, y, col, 0.36)
        y += 17
        if y > h - 8:
            break

    return panel


def _render_status(
    w: int,
    loiter_total: int,
    brawl_total: int,
    breakin_total: int,
    suspicious_total: int,
    fps_actual: float,
) -> np.ndarray:
    bar = np.full((STATUS_H, w, 3), C["bg"], dtype=np.uint8)
    cv2.line(bar, (0, 0), (w, 0), C["border"], 1)
    y = 22
    _put(bar, "STATUS:", 10, y, C["dim"], 0.44)
    x = 80
    for label, count, tkey in (
        ("BREAK-IN",  breakin_total,   "breaking_in"),
        ("BRAWL",     brawl_total,     "brawl"),
        ("LOITERING", loiter_total,    "loitering"),
        ("SUSPICIOUS", suspicious_total, "suspicious"),
    ):
        col = C.get(tkey, C["dim"]) if count else C["dim"]
        cv2.rectangle(bar, (x, y - 14), (x + 148, y + 6), (30, 30, 45), -1)
        cv2.rectangle(bar, (x, y - 14), (x + 148, y + 6), col, 1)
        _put(bar, f"{label}: {count}", x + 5, y, col, 0.38)
        x += 154
    _put(bar, f"FPS: {fps_actual:.1f}   Model: yolov8n-pose", x + 10, y, C["dim"], 0.38)
    y = 50
    _put(bar, "Activity-based detection — no predefined zones required", 10, y, C["dim"], 0.36)
    return bar


# ── FRAME ANNOTATION ─────────────────────────────────────────────────────────

def _annotate_frame(
    frame: np.ndarray,
    persons: List[dict],
    loiter_map: Dict[int, float],
    brawl_s: float,
    alert_threats: Set[str],
) -> None:
    for p in persons:
        act    = p.get("activity_result")
        color  = C.get(act.activity if act else "walking", C["walking"])
        x1, y1, x2, y2 = (int(v) for v in p["bbox"])
        tid    = p["track_id"]
        dwell  = loiter_map.get(tid, 0.0) if tid is not None else 0.0

        # Skeleton
        if p.get("kp_xy") is not None:
            _draw_skeleton(frame, p["kp_xy"], p.get("kp_conf"), color)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: activity + dwell
        act_name = act.activity if act else "?"
        lbl = f"ID:{tid}  {ACTIVITY_LABEL.get(act_name, act_name)}"
        if dwell > 0:
            lbl += f"  {dwell:.0f}s"
        _put(frame, lbl, x1, max(y1 - 6, 14), color, 0.40)

        # Loitering dwell bar
        if dwell > 0:
            bw     = max(x2 - x1, 1)
            filled = int(bw * min(dwell / LOITERING_THRESHOLD_S, 1.0))
            cv2.rectangle(frame, (x1, y2 + 2), (x2,          y2 + 8), (50, 50, 50), -1)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 8), C["loitering"], -1)

    # Alert banners
    fh, fw = frame.shape[:2]
    if brawl_s >= BRAWL_MIN_S:
        cv2.rectangle(frame, (0, fh - 34), (fw, fh), (0, 0, 170), -1)
        _put(frame, f"!! BRAWL DETECTED — {brawl_s:.1f}s !!",
             10, fh - 10, (255, 255, 255), 0.58, 2)

    top_y = 0
    for threat, color, label in (
        ("breaking_in", C["breaking_in"], "BREAKING-IN ALERT"),
        ("brawl",       C["brawl"],       "BRAWL ALERT"),
        ("loitering",   C["loitering"],   "LOITERING ALERT"),
    ):
        if threat in alert_threats:
            cv2.rectangle(frame, (0, top_y), (fw, top_y + 24), color, -1)
            _put(frame, f"{label} — POSE-BASED DETECTION",
                 8, top_y + 17, (0, 0, 0), 0.48, 2)
            top_y += 26


# ── INCIDENT BUILDER ──────────────────────────────────────────────────────────

def _make_incident(
    threat: str, severity: str, conf: float,
    video_t: float, bbox: List[float], tid: Optional[int],
    activity: str,
) -> Incident:
    return Incident(
        id          = f"CAM-01_{threat}_{int(VIDEO_BASE_T + video_t)}",
        camera_id   = "CAM-01",
        threat_type = threat,
        severity    = severity,
        confidence  = conf,
        timestamp   = f"video+{_fmt_t(video_t)} (activity={activity})",
        bbox        = bbox,
        track_id    = tid,
    )


# ── PER-FRAME PROCESSING HELPERS ─────────────────────────────────────────────

def _extract_persons(results) -> Tuple[List[dict], Set[int]]:
    persons: List[dict] = []
    active_ids: Set[int] = set()
    for r in results:
        if r.boxes is None:
            continue
        kps = r.keypoints
        for i, box in enumerate(r.boxes):
            if int(box.cls[0]) != 0:
                continue
            tid  = int(box.id[0]) if box.id is not None else None
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            kp_xy   = kps.xy[i].cpu().numpy()   if kps is not None else None
            kp_conf = (kps.conf[i].cpu().numpy()
                       if kps is not None and kps.conf is not None else None)
            persons.append({
                "bbox": bbox, "track_id": tid, "conf": conf,
                "kp_xy": kp_xy, "kp_conf": kp_conf,
                "activity_result": None,
            })
            if tid is not None:
                active_ids.add(tid)
    return persons, active_ids


def _classify_activities(
    persons: List[dict],
    classifier: PoseActivityClassifier,
    active_ids: Set[int],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in persons:
        tid = p["track_id"]
        if tid is None or p["kp_xy"] is None:
            continue
        result = classifier.update(tid, p["kp_xy"], p["kp_conf"])
        p["activity_result"] = result
        counts[result.activity] = counts.get(result.activity, 0) + 1
    classifier.evict_stale(active_ids)
    return counts


def _run_loitering(
    persons: List[dict], loitering: LoiteringTracker,
    sim_t: float, video_t: float,
    rate_limiter: RateLimiter, dispatcher: AlertDispatcher,
    alert_log: deque, active_ids: Set[int],
) -> Tuple[Dict[int, float], int]:
    new_map: Dict[int, float] = {}
    count = 0
    for p in persons:
        tid = p["track_id"]
        if tid is None:
            continue
        cx = (p["bbox"][0] + p["bbox"][2]) / 2
        cy = (p["bbox"][1] + p["bbox"][3]) / 2
        with _patch("inference_engine.time.time", return_value=sim_t):
            dwell = loitering.update(tid, cx, cy)
        new_map[tid] = dwell
        if dwell >= LOITERING_THRESHOLD_S and rate_limiter.allow(f"{tid}:loitering", video_t):
            inc = _make_incident("loitering", "medium", p["conf"],
                                  video_t, p["bbox"], tid, "standing")
            dispatcher.dispatch(inc)
            alert_log.append({"video_t": video_t, "threat": "loitering",
                               "track_id": tid, "activity": "standing"})
            count += 1
    loitering.evict_stale(active_ids)
    return new_map, count


def _run_brawl(
    persons: List[dict], brawl_s_prev: float,
    video_t: float, rate_limiter: RateLimiter,
    dispatcher: AlertDispatcher, alert_log: deque,
    disp_w: int,
) -> Tuple[float, int]:
    fighters = [p for p in persons
                if p.get("activity_result") and
                   p["activity_result"].activity in ("fighting", "striking")]
    count = 0

    if len(fighters) >= 2:
        brawl_s = brawl_s_prev + (1.0 / 20.0)   # approx increment per frame at 20fps
    elif len(fighters) == 1:
        # single person striking — still a potential brawl
        brawl_s = brawl_s_prev + (0.5 / 20.0)
    else:
        brawl_s = 0.0

    if brawl_s >= BRAWL_MIN_S and rate_limiter.allow("cluster:brawl", video_t):
        conf = max((p["conf"] for p in fighters), default=0.5)
        inc  = _make_incident("brawl", "high", conf,
                               video_t, [0, 0, disp_w, TARGET_H], None, "fighting")
        dispatcher.dispatch(inc)
        alert_log.append({"video_t": video_t, "threat": "brawl",
                           "track_id": None, "activity": "fighting"})
        count += 1
    return brawl_s, count


def _run_activity_threats(
    persons: List[dict], video_t: float,
    rate_limiter: RateLimiter, dispatcher: AlertDispatcher,
    alert_log: deque,
) -> int:
    """Fire breaking_in / suspicious alerts from activity classification."""
    count = 0
    for p in persons:
        result = p.get("activity_result")
        if result is None:
            continue
        threat_info = ACTIVITY_THREAT.get(result.activity)
        if not threat_info or threat_info[0] is None:
            continue
        threat, severity = threat_info
        if threat in ("brawl",):   # brawl handled separately
            continue
        tid = p["track_id"]
        key = f"{tid}:{threat}"
        if rate_limiter.allow(key, video_t):
            inc = _make_incident(threat, severity, p["conf"],
                                  video_t, p["bbox"], tid, result.activity)
            dispatcher.dispatch(inc)
            alert_log.append({"video_t": video_t, "threat": threat,
                               "track_id": tid, "activity": result.activity})
            count += 1
    return count


def _active_alert_threats(
    persons: List[dict], loiter_map: Dict[int, float], brawl_s: float,
) -> Set[str]:
    fired: Set[str] = set()
    for p in persons:
        result = p.get("activity_result")
        if result and result.activity in ("climbing", "striking"):
            fired.add("breaking_in")
        if result and result.activity in ("fighting",):
            fired.add("brawl")
        tid = p["track_id"]
        if tid is not None and loiter_map.get(tid, 0) >= LOITERING_THRESHOLD_S:
            fired.add("loitering")
    if brawl_s >= BRAWL_MIN_S:
        fired.add("brawl")
    return fired


def _print_summary(
    out_path: Path, proc_count: int, total_frames: int, elapsed: float,
    loiter_total: int, brawl_total: int, breakin_total: int,
    suspicious_total: int, alert_log: deque,
) -> None:
    fps_out = proc_count / elapsed if elapsed > 0 else 0.0
    print("\n" + "=" * 66)
    print("  PROCESSING COMPLETE")
    print("=" * 66)
    print(f"  Frames processed  : {proc_count} / {total_frames}")
    print(f"  Processing speed  : {fps_out:.1f} fps")
    print(f"  Output video      : {out_path}")
    print()
    print(f"  Breaking-in alerts: {breakin_total}")
    print(f"  Brawl alerts      : {brawl_total}")
    print(f"  Loitering alerts  : {loiter_total}")
    print(f"  Suspicious alerts : {suspicious_total}")
    print(f"  Total incidents   : {loiter_total + brawl_total + breakin_total + suspicious_total}")
    if alert_log:
        print()
        print("  Alert timeline:")
        for entry in alert_log:
            ts    = _fmt_t(entry["video_t"])
            act   = entry.get("activity", "")
            tidst = f" ID:{entry['track_id']}" if entry["track_id"] is not None else ""
            print(f"    [{ts}] {entry['threat'].upper()}{tidst}  ← activity: {act}")
    print()
    print(f"  Opening {out_path} ...")
    subprocess.Popen(["open", str(out_path)])


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def run(video_path: str, display: bool, skip: int) -> None:
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

    print("\n" + "=" * 66)
    print("  POSE-BASED SECURITY COMMAND CENTER")
    print("=" * 66)
    print(f"  Source    : {video_path}")
    print(f"  Resolution: {src_w}x{src_h}  →  display {disp_w}x{TARGET_H}")
    print(f"  Duration  : {_fmt_t(total_t)}  ({total_frames} frames @ {src_fps:.1f}fps)")
    print(f"  Model     : yolov8n-pose  (17 keypoints per person)")
    print(f"  Detection : activity-based, no fixed zones\n")

    print("Loading yolov8n-pose.pt ...")
    model = YOLO("yolov8n-pose.pt")
    print("Model ready.\n")

    Path("demo_output").mkdir(exist_ok=True)
    Path("incidents").mkdir(exist_ok=True)
    out_path = Path("demo_output") / "pose_command_center_output.mp4"
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        min(src_fps / max(skip, 1), 30.0), (canvas_w, canvas_h),
    )

    classifier      = PoseActivityClassifier()
    loitering       = LoiteringTracker()
    dispatcher      = AlertDispatcher()
    rate_limiter    = RateLimiter(ALERT_COOLDOWN_S)
    alert_log: deque = deque(maxlen=50)

    loiter_map: Dict[int, float] = {}
    activity_counts: Dict[str, int] = {}
    loiter_total = brawl_total = breakin_total = suspicious_total = 0
    brawl_s      = 0.0
    frame_idx    = proc_count = 0
    t0           = _time.time()
    fps_actual   = 0.0

    print(f"{'Frame':>7}  {'Time':>8}  {'Tracks':>7}  {'Dwell':>7}  "
          f"{'BrawlS':>7}  {'Activity':>12}  {'Alerts':>7}")
    print("-" * 70)

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

        # Classify activities from pose
        activity_counts = _classify_activities(persons, classifier, active_ids)

        # Loitering (time-based, activity=standing)
        loiter_map, lc = _run_loitering(persons, loitering, sim_t, video_t,
                                         rate_limiter, dispatcher, alert_log, active_ids)
        loiter_total += lc

        # Brawl (activity=fighting/striking, sustained)
        brawl_s, bc = _run_brawl(persons, brawl_s, video_t,
                                   rate_limiter, dispatcher, alert_log, disp_w)
        brawl_total += bc

        # Breaking-in / suspicious (direct from activity)
        act_count = _run_activity_threats(persons, video_t,
                                           rate_limiter, dispatcher, alert_log)
        for p in persons:
            result = p.get("activity_result")
            if result:
                if result.activity in ("climbing", "striking"):
                    breakin_total += act_count and 1
                elif result.activity == "crouching":
                    suspicious_total += act_count and 1
        breakin_total  += sum(1 for e in alert_log
                              if e.get("threat") == "breaking_in" and
                              abs(e["video_t"] - video_t) < 0.1)
        suspicious_total += sum(1 for e in alert_log
                                if e.get("threat") == "suspicious" and
                                abs(e["video_t"] - video_t) < 0.1)

        alert_threats = _active_alert_threats(persons, loiter_map, brawl_s)
        _annotate_frame(small, persons, loiter_map, brawl_s, alert_threats)

        panel  = _render_panel(TARGET_H, video_path, video_t, total_t,
                                len(active_ids), activity_counts,
                                loiter_total, brawl_total, breakin_total,
                                suspicious_total, alert_log)
        status = _render_status(canvas_w, loiter_total, brawl_total,
                                 breakin_total, suspicious_total, fps_actual)
        canvas = np.full((canvas_h, canvas_w, 3), C["bg"], dtype=np.uint8)
        canvas[:TARGET_H, :disp_w]               = small
        canvas[:TARGET_H, disp_w:]               = panel
        canvas[TARGET_H:TARGET_H + STATUS_H, :]  = status
        writer.write(canvas)

        if display:
            cv2.imshow("Pose Command Center", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[User quit]")
                break

        elapsed    = _time.time() - t0
        fps_actual = proc_count / elapsed if elapsed > 0 else 0.0
        if proc_count % 50 == 1 or proc_count == 1:
            max_dwell   = max(loiter_map.values(), default=0.0)
            top_act     = max(activity_counts, key=activity_counts.get,
                              default="none") if activity_counts else "none"
            total_alerts = loiter_total + brawl_total + breakin_total + suspicious_total
            print(f"{frame_idx:>7}  t={video_t:>6.1f}s  tracks={len(active_ids):>3}  "
                  f"dwell={max_dwell:>5.1f}s  brawl={brawl_s:>5.1f}s  "
                  f"{top_act:>12}  alerts={total_alerts:>4}")

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    _print_summary(out_path, proc_count, total_frames, _time.time() - t0,
                   loiter_total, brawl_total, breakin_total, suspicious_total, alert_log)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose-Based Security Command Center")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--display", action="store_true",
                        help="Show live OpenCV window")
    parser.add_argument("--skip", type=int, default=2,
                        help="Process every Nth frame (default 2)")
    args = parser.parse_args()
    run(args.video, args.display, max(args.skip, 1))
