"""
Loitering Detection Demo  — Moving Person + YOLO Detection
===========================================================
Demonstrates the full loitering detection pipeline with:

  • YOLOv8n pretrained (COCO) — actual model inference on every frame
  • A synthetic webcam feed: person walks in, browses, stops → loitering alert
  • Real LoiteringTracker + AlertDispatcher from inference_engine.py

Movement Phases
---------------
  Phase 1  0 –  6 s  Person walks into the scene from the left
  Phase 2  6 – 12 s  Person browses / wanders (small zigzag)
  Phase 3 12 – 45 s  Person stops — LoiteringTracker starts counting
  Alert   42 s        30 s dwell reached → LOITERING ALERT fires

Person Rendering
----------------
  A textured human silhouette is drawn with realistic skin/clothing colours,
  gaussian-noise texture, and edge blur — all to maximise YOLO detection.
  YOLO confidence is lowered to 0.10 so the model has every chance.

  Two detection badges are shown on-screen:
    YOLO   — real model detection (green badge)
    INJECT — fallback synthetic detection (orange badge)

  All loitering logic, alert dispatch, incident JSON and evidence JPEG are
  real regardless of detection mode.

Usage
-----
    python demo_loitering.py              # saves video + opens it
    python demo_loitering.py --display    # also shows live cv2 window
"""

import argparse
import math
import sys
from pathlib import Path
from unittest.mock import patch as _patch

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent))
import inference_engine                                   # noqa: E402
from inference_engine import (                            # noqa: E402
    AlertDispatcher, Incident, LoiteringTracker, THREAT_CONFIG,
    LOITERING_ZONE_RADIUS_PX,
)

# ─── MODULE-LEVEL RNG (numpy.random.Generator — not legacy API) ───────────────
_rng = np.random.default_rng()

# ─── DEMO PARAMETERS ─────────────────────────────────────────────────────────

FPS           = 20
DURATION_S    = 48
W, H          = 1280, 720
CAMERA_ID     = "DEMO-CAM"
CONF_THRESH   = 0.10          # low threshold → max YOLO sensitivity
LOITER_THRESH = THREAT_CONFIG["loitering"]["min_duration_s"]   # 30 s
SIM_START     = 1_000_000.0   # avoid rate-limiter false-suppress

OUTPUT_DIR    = Path("demo_output")

# ─── MOVEMENT PATH ────────────────────────────────────────────────────────────

LOITER_CX     = 680
LOITER_BOTTOM = 550

WAYPOINTS = [
    # (end_frame,  cx,              bottom,               phase_name)
    (0,            50,              550,                  "entering"),
    (FPS * 6,      LOITER_CX,       LOITER_BOTTOM,        "walking_in"),
    (FPS * 8,      LOITER_CX + 60,  LOITER_BOTTOM - 20,   "browsing"),
    (FPS * 10,     LOITER_CX - 40,  LOITER_BOTTOM + 15,   "browsing"),
    (FPS * 12,     LOITER_CX,       LOITER_BOTTOM,        "stopping"),
    (FPS * DURATION_S, LOITER_CX,   LOITER_BOTTOM,        "loitering"),
]

LOITER_START_FRAME = FPS * 12   # LoiteringTracker starts accumulating here


def get_position(frame_num: int) -> tuple:
    """Return (cx, bottom, phase) for the given frame by interpolating waypoints."""
    prev_f, prev_cx, prev_b, _ = WAYPOINTS[0]
    for end_f, end_cx, end_b, phase in WAYPOINTS[1:]:
        if frame_num <= end_f:
            if end_f == prev_f:
                return prev_cx, prev_b, phase
            t = (frame_num - prev_f) / (end_f - prev_f)
            t = t * t * (3 - 2 * t)          # smooth-step easing
            cx = int(prev_cx + (end_cx - prev_cx) * t)
            bt = int(prev_b  + (end_b  - prev_b)  * t)
            return cx, bt, phase
        prev_f, prev_cx, prev_b, _ = end_f, end_cx, end_b, phase
    return WAYPOINTS[-1][1], WAYPOINTS[-1][2], WAYPOINTS[-1][3]


# ─── PERSON SPRITE ────────────────────────────────────────────────────────────

# Colour palette (BGR)
_SKIN   = np.array([105, 145, 195], dtype=np.float32)   # warm peach skin
_HAIR   = np.array([ 25,  20,  35], dtype=np.float32)   # dark brown hair
_SHIRT  = np.array([ 80,  60,  50], dtype=np.float32)   # dark navy shirt
_PANTS  = np.array([ 50,  45,  60], dtype=np.float32)   # dark charcoal jeans
_SHOES  = np.array([ 20,  18,  25], dtype=np.float32)   # black shoes


def _fill_region(canvas: np.ndarray, pts, color: np.ndarray, noise_std: int = 12) -> None:
    """Fill a polygon region with colour + gaussian noise texture."""
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    noise     = _rng.normal(0, noise_std, canvas.shape).astype(np.float32)
    textured  = np.clip(color + noise, 0, 255).astype(np.uint8)
    canvas[mask == 255] = textured[mask == 255]


def build_person_sprite(height: int,
                        step: float = 0.0,
                        is_walking: bool = False) -> tuple:
    """
    Build a person sprite for compositing into the scene.

    step       : animation phase [0, 2π) — drives leg/arm swing
    is_walking : enables leg/arm oscillation
    Returns    : (sprite_bgr, alpha_mask, (cx, shoe_bot, full_w, full_h))
    """
    h  = height
    sw = h // 3
    canvas = np.zeros((h + 20, sw + 40, 3), dtype=np.float32)
    full_h, full_w = canvas.shape[:2]
    cx = full_w // 2

    hr      = max(h // 7, 10)
    hw      = max(h // 9, 7)
    head_cy = hr + 2

    neck_bot  = head_cy + hr
    torso_top = neck_bot
    torso_bot = torso_top + h * 9 // 20
    leg_mid   = torso_bot + h // 20
    foot_bot  = torso_top + h - 8
    shoe_bot  = foot_bot + 10

    swing     = math.sin(step) * 18 if is_walking else 0.0
    arm_swing = -swing * 0.6

    # ── Legs ──────────────────────────────────────────────────────────────────
    ll_bot_x = int(cx - hw // 2 + swing)
    ll_pts = np.array([
        [cx - hw - 2,              leg_mid],
        [cx - 2,                   leg_mid],
        [cx - 2 + int(swing // 2), foot_bot],
        [ll_bot_x - 4,             foot_bot],
    ], dtype=np.int32)
    _fill_region(canvas, ll_pts, _PANTS, noise_std=10)

    rl_bot_x = int(cx + hw // 2 - swing)
    rl_pts = np.array([
        [cx + 2,                   leg_mid],
        [cx + hw + 2,              leg_mid],
        [rl_bot_x + 4,             foot_bot],
        [cx + 2 - int(swing // 2), foot_bot],
    ], dtype=np.int32)
    _fill_region(canvas, rl_pts, _PANTS, noise_std=10)

    # Shoes
    _fill_region(canvas, np.array([
        [ll_bot_x - 6,  foot_bot],
        [ll_bot_x + 4,  foot_bot],
        [ll_bot_x + 8,  shoe_bot],
        [ll_bot_x - 10, shoe_bot],
    ], dtype=np.int32), _SHOES, noise_std=5)

    _fill_region(canvas, np.array([
        [rl_bot_x - 4,  foot_bot],
        [rl_bot_x + 6,  foot_bot],
        [rl_bot_x + 10, shoe_bot],
        [rl_bot_x - 8,  shoe_bot],
    ], dtype=np.int32), _SHOES, noise_std=5)

    # ── Torso ─────────────────────────────────────────────────────────────────
    _fill_region(canvas, np.array([
        [cx - hw - 4, torso_top],
        [cx + hw + 4, torso_top],
        [cx + hw + 1, torso_bot],
        [cx - hw - 1, torso_bot],
    ], dtype=np.int32), _SHIRT, noise_std=12)

    # ── Arms ──────────────────────────────────────────────────────────────────
    arm_top  = torso_top + hr // 2
    arm_len  = h // 4
    la_swing = int(arm_swing)
    ra_swing = -la_swing

    _fill_region(canvas, np.array([
        [cx - hw - 3,             arm_top],
        [cx - hw - 12,            arm_top],
        [cx - hw - 12 + la_swing, arm_top + arm_len],
        [cx - hw - 2  + la_swing, arm_top + arm_len],
    ], dtype=np.int32), _SHIRT, noise_std=12)

    _fill_region(canvas, np.array([
        [cx + hw + 12,            arm_top],
        [cx + hw + 3,             arm_top],
        [cx + hw + 2  + ra_swing, arm_top + arm_len],
        [cx + hw + 12 + ra_swing, arm_top + arm_len],
    ], dtype=np.int32), _SHIRT, noise_std=12)

    # Hands
    _fill_region(canvas, np.array([
        [cx - hw - 14 + la_swing, arm_top + arm_len],
        [cx - hw      + la_swing, arm_top + arm_len],
        [cx - hw      + la_swing, arm_top + arm_len + hr // 2],
        [cx - hw - 14 + la_swing, arm_top + arm_len + hr // 2],
    ], dtype=np.int32), _SKIN, noise_std=6)

    _fill_region(canvas, np.array([
        [cx + hw      + ra_swing, arm_top + arm_len],
        [cx + hw + 14 + ra_swing, arm_top + arm_len],
        [cx + hw + 14 + ra_swing, arm_top + arm_len + hr // 2],
        [cx + hw      + ra_swing, arm_top + arm_len + hr // 2],
    ], dtype=np.int32), _SKIN, noise_std=6)

    # ── Neck ──────────────────────────────────────────────────────────────────
    _fill_region(canvas, np.array([
        [cx - 5, head_cy + hr - 2],
        [cx + 5, head_cy + hr - 2],
        [cx + 5, neck_bot + 2],
        [cx - 5, neck_bot + 2],
    ], dtype=np.int32), _SKIN, noise_std=6)

    # ── Head ──────────────────────────────────────────────────────────────────
    head_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(head_mask, (cx, head_cy), (hr, int(hr * 1.15)), 0, 0, 360, 255, -1)
    noise_h = _rng.normal(0, 8, canvas.shape).astype(np.float32)
    canvas[head_mask == 255] = np.clip(_SKIN + noise_h, 0, 255).astype(np.uint8)[head_mask == 255]

    # Hair
    hair_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(hair_mask, (cx, head_cy - hr // 4),
                (hr + 2, int(hr * 0.75)), 0, 180, 360, 255, -1)
    canvas[hair_mask == 255] = _HAIR.astype(np.uint8)

    # Face detail
    ex, ey = hr // 3, hr // 4
    cv2.circle(canvas, (cx - ex, head_cy - ey), max(hr // 10, 2), (20, 15, 15), -1)
    cv2.circle(canvas, (cx + ex, head_cy - ey), max(hr // 10, 2), (20, 15, 15), -1)
    cv2.ellipse(canvas, (cx, head_cy + ey),
                (ex, max(hr // 12, 2)), 0, 0, 180, (80, 60, 80), -1)

    # ── Lighting gradient ─────────────────────────────────────────────────────
    grad   = np.linspace(1.08, 0.92, full_w, dtype=np.float32)
    canvas = np.clip(canvas * grad[np.newaxis, :, np.newaxis], 0, 255)
    canvas = cv2.GaussianBlur(canvas.astype(np.uint8), (3, 3), 0).astype(np.float32)

    # ── Alpha mask ────────────────────────────────────────────────────────────
    alpha = np.any(canvas > 8, axis=2).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(alpha, (5, 5), 1)

    return canvas.astype(np.uint8), alpha, (cx, shoe_bot, full_w, full_h)


def composite_person(scene: np.ndarray,
                     sprite: np.ndarray,
                     alpha: np.ndarray,
                     sprite_meta: tuple,
                     cx: int, bottom: int) -> tuple:
    """
    Alpha-composite sprite into scene aligning shoe_bot → bottom.
    Returns tight bounding box as (x1, y1, x2, y2).
    """
    sp_cx, sp_sb, sp_w, sp_h = sprite_meta
    paste_x = cx - sp_cx
    paste_y = bottom - sp_sb

    sx1 = max(0, -paste_x);  ex1 = min(sp_w, W - paste_x)
    sy1 = max(0, -paste_y);  ey1 = min(sp_h, H - paste_y)
    dx1 = max(0, paste_x);   dy1 = max(0, paste_y)
    dw  = ex1 - sx1;         dh  = ey1 - sy1

    if dw <= 0 or dh <= 0:
        return (float(cx - 30), float(bottom - 200),
                float(cx + 30), float(bottom))

    roi   = scene[dy1:dy1 + dh, dx1:dx1 + dw].astype(np.float32)
    patch = sprite[sy1:ey1, sx1:ex1].astype(np.float32)
    a     = alpha[sy1:ey1, sx1:ex1].astype(np.float32) / 255.0
    a3    = a[:, :, np.newaxis]

    scene[dy1:dy1 + dh, dx1:dx1 + dw] = np.clip(
        patch * a3 + roi * (1 - a3), 0, 255
    ).astype(np.uint8)

    ys, xs = np.nonzero(a > 0.15)
    if len(xs) == 0:
        return (float(cx - 30), float(bottom - 200),
                float(cx + 30), float(bottom))
    return (float(dx1 + xs.min()), float(dy1 + ys.min()),
            float(dx1 + xs.max()), float(dy1 + ys.max()))


# ─── SCENE BACKGROUND ────────────────────────────────────────────────────────

def _build_background() -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(280):
        t = y / 280
        img[y] = [int(220 - 25 * t), int(215 - 15 * t), int(155 + 35 * t)]
    img[280:390] = [148, 138, 122]
    img[390:]    = [72, 72, 72]
    for x in range(0, W, 90):
        cv2.line(img, (x, 495), (x + 50, 495), (210, 210, 210), 2)
    cv2.rectangle(img, (0, 385), (W, 395), (175, 165, 155), -1)
    cv2.rectangle(img, (0, 40), (200, 282), (150, 140, 130), -1)
    cv2.rectangle(img, (0, 40), (200, 282), (115, 105, 98), 2)
    for wy in range(65, 270, 54):
        for wx in range(18, 182, 50):
            cv2.rectangle(img, (wx, wy), (wx + 34, wy + 38), (185, 225, 255), -1)
    cv2.rectangle(img, (W - 225, 22), (W, 282), (160, 150, 140), -1)
    cv2.rectangle(img, (W - 225, 22), (W, 282), (120, 110, 103), 2)
    for wy in range(48, 270, 54):
        for wx in range(W - 210, W - 18, 50):
            cv2.rectangle(img, (wx, wy), (wx + 34, wy + 38), (185, 225, 255), -1)
    cv2.line(img, (800, 88), (800, 360), (52, 52, 52), 8)
    cv2.line(img, (800, 88), (860, 88),  (52, 52, 52), 7)
    cv2.circle(img, (863, 88), 13, (255, 255, 185), -1)
    cv2.circle(img, (863, 88), 13, (215, 175, 75),  2)
    cv2.rectangle(img, (900, 352), (1065, 372), (102, 62, 28), -1)
    for bx in (912, 1052):
        cv2.line(img, (bx, 372), (bx, 400), (78, 48, 20), 7)
    for px, py in [(240, 310), (310, 305), (1020, 318)]:
        cv2.rectangle(img, (px - 4, py - 30), (px + 4, py), (80, 75, 85), -1)
        cv2.circle(img, (px, py - 34), 5, (110, 130, 160), -1)
    return img


def _draw_ground_shadow(img: np.ndarray, cx: int, bottom: int, scale: float = 1.0) -> None:
    sw, sh = int(55 * scale), int(14 * scale)
    overlay = img.copy()
    cv2.ellipse(overlay, (cx, bottom + 4), (sw, sh), 0, 0, 360, (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)


# ─── ANNOTATION HELPERS ──────────────────────────────────────────────────────

def _box_color(alert_fired: bool, loiter_s: float) -> tuple:
    if alert_fired:
        return (0, 0, 235)
    if loiter_s <= 0:
        return (0, 200, 60)
    progress = min(loiter_s / LOITER_THRESH, 1.0)
    return (0, int(200 - 180 * progress), int(50 + 205 * progress))


def _draw_detection_box(out: np.ndarray, bbox: list,
                        track_id: int, box_col: tuple,
                        yolo_detected: bool) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), box_col, 3)
    badge = "YOLO" if yolo_detected else "INJECT"
    bcol  = (0, 185, 55) if yolo_detected else (0, 145, 245)
    cv2.rectangle(out, (x1, y1 - 24), (x1 + 86, y1), bcol, -1)
    cv2.putText(out, badge, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    lbl = f"person  ID:{track_id}"
    (lw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(out, (x1 + 89, y1 - 24), (x1 + 93 + lw, y1), (35, 35, 35), -1)
    cv2.putText(out, lbl, (x1 + 93, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1)


def _draw_loiter_bar(out: np.ndarray, bbox: list,
                     loiter_s: float, box_col: tuple) -> None:
    x1, _, x2, y2 = [int(v) for v in bbox]
    progress = min(loiter_s / LOITER_THRESH, 1.0)
    bar_x1, bar_y1 = x1, y2 + 10
    bar_x2, bar_y2 = x2, y2 + 26
    bw = bar_x2 - bar_x1
    cv2.rectangle(out, (bar_x1, bar_y1), (bar_x2, bar_y2), (35, 35, 35), -1)
    cv2.rectangle(out, (bar_x1, bar_y1),
                  (bar_x1 + int(bw * progress), bar_y2), box_col, -1)
    cv2.rectangle(out, (bar_x1, bar_y1), (bar_x2, bar_y2), (170, 170, 170), 1)
    cv2.putText(out, f"Loitering: {loiter_s:.1f}s / {LOITER_THRESH}s",
                (bar_x1, bar_y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (215, 215, 215), 1)
    x1i, y1i, x2i, y2i = [int(v) for v in bbox]
    zcx, zcy = (x1i + x2i) // 2, (y1i + y2i) // 2
    ov = out.copy()
    cv2.circle(ov, (zcx, zcy), LOITERING_ZONE_RADIUS_PX, box_col, 1)
    cv2.addWeighted(ov, 0.35, out, 0.65, 0, out)


def _draw_phase_banner(out: np.ndarray, phase: str, alert_fired: bool) -> None:
    phase_map = {
        "entering":   ("ENTERING SCENE",              (180, 180, 60)),
        "walking_in": ("WALKING",                      (180, 180, 60)),
        "browsing":   ("BROWSING / WANDERING",         (60, 180, 200)),
        "stopping":   ("STOPPING...",                  (60, 200, 200)),
        "loitering":  ("STANDING STILL — MONITORING", (60, 200, 60)),
    }
    if alert_fired:
        ph_txt, ph_col = "LOITERING DETECTED", (0, 60, 230)
    else:
        ph_txt, ph_col = phase_map.get(phase, ("", (200, 200, 200)))
    if not ph_txt:
        return
    (pw, _), _ = cv2.getTextSize(ph_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    px = (W - pw) // 2
    cv2.rectangle(out, (px - 10, 10), (px + pw + 10, 42), (20, 20, 20), -1)
    cv2.putText(out, ph_txt, (px, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ph_col, 2)


def _draw_hud(out: np.ndarray, rows: list) -> None:
    ph = len(rows) * 24 + 14
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (440, ph), (12, 12, 12), -1)
    cv2.addWeighted(ov, 0.78, out, 0.22, 0, out)
    cv2.rectangle(out, (0, 0), (440, ph), (60, 60, 60), 1)
    for i, (txt, col, scale) in enumerate(rows):
        cv2.putText(out, txt, (10, 18 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col,
                    2 if i == 0 else 1)


def _draw_legend(out: np.ndarray) -> None:
    items = [
        ("Green  box  = Moving / Just detected",  (0, 200, 60)),
        ("Yellow box  = Monitoring dwell time",   (0, 200, 200)),
        ("Red    box  = Loitering alert active",  (0, 60, 230)),
        ("YOLO   badge = real model hit",         (0, 185, 55)),
        ("INJECT badge = synthetic fallback",     (0, 145, 245)),
    ]
    n = len(items)
    ov = out.copy()
    cv2.rectangle(ov, (0, H - n * 22 - 8), (380, H), (12, 12, 12), -1)
    cv2.addWeighted(ov, 0.70, out, 0.30, 0, out)
    for i, (txt, col) in enumerate(items):
        cv2.putText(out, txt, (8, H - n * 22 + i * 22 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1)


def _draw_alert_banner(out: np.ndarray, loiter_s: float) -> None:
    bw, bh = 740, 90
    bx = (W - bw) // 2
    by = H - bh - 18
    ov = out.copy()
    cv2.rectangle(ov, (bx, by), (bx + bw, by + bh), (0, 0, 175), -1)
    cv2.addWeighted(ov, 0.88, out, 0.12, 0, out)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 0, 245), 3)
    cv2.putText(out, "!  LOITERING ALERT  —  SUSPICIOUS PERSON",
                (bx + 16, by + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2)
    cv2.putText(out,
                f"Dwell: {loiter_s:.0f}s  |  Cam:{CAMERA_ID}"
                "  |  Action: audio_warning  |  Notifying personnel...",
                (bx + 16, by + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)


def _render_annotations(frame: np.ndarray,
                         bbox: list,
                         track_id: int,
                         loiter_s: float,
                         alert_fired: bool,
                         yolo_detected: bool,
                         phase: str,
                         frame_num: int,
                         total_frames: int,
                         yolo_hits: int) -> np.ndarray:
    out      = frame.copy()
    box_col  = _box_color(alert_fired, loiter_s)

    _draw_detection_box(out, bbox, track_id, box_col, yolo_detected)

    if loiter_s > 0 or alert_fired:
        _draw_loiter_bar(out, bbox, loiter_s, box_col)

    _draw_phase_banner(out, phase, alert_fired)

    dwell_s = max(loiter_s, 0.0)
    hud_rows = [
        ("SECURITY SYSTEM — LOITERING DEMO",          (80, 210, 80),  0.60),
        (f"Camera  : {CAMERA_ID}",                    (200, 200, 200), 0.48),
        ("Model   : YOLOv8n  (COCO pretrained)",      (200, 200, 200), 0.48),
        (f"Frame   : {frame_num:4d} / {total_frames}", (200, 200, 200), 0.48),
        (f"YOLO hits: {yolo_hits}/{frame_num + 1}",
         (0, 185, 55) if yolo_detected else (0, 145, 245), 0.48),
        (f"Dwell   : {dwell_s:.1f}s",                 (200, 200, 200), 0.48),
    ]
    if alert_fired:
        hud_rows.append(("STATUS  : *** ALERT TRIGGERED ***", (0, 60, 230), 0.55))
    else:
        hud_rows.append((
            f"STATUS  : {phase.upper().replace('_', ' ')}",
            (80, 210, 80), 0.48,
        ))
    _draw_hud(out, hud_rows)
    _draw_legend(out)

    if alert_fired:
        _draw_alert_banner(out, loiter_s)

    return out


# ─── YOLO INFERENCE ──────────────────────────────────────────────────────────

def _run_yolo(model, raw: np.ndarray) -> dict | None:
    """Run model.track() (fallback: predict) and return best person dict or None."""
    try:
        results = model.track(
            raw,
            persist=True,
            conf=CONF_THRESH,
            verbose=False,
            tracker="bytetrack.yaml",
            classes=[0],
        )
    except Exception:
        results = model.predict(raw, conf=CONF_THRESH, verbose=False, classes=[0])

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue
            tid = (int(box.id[0])
                   if hasattr(box, "id") and box.id is not None
                   else 1)
            return {
                "bbox":     box.xyxy[0].tolist(),
                "track_id": tid,
                "conf":     float(box.conf[0]),
            }
    return None


# ─── ALERT FIRING ────────────────────────────────────────────────────────────

def _fire_alert(dispatcher: AlertDispatcher, frame_num: int,
                loiter_s: float, sim_t: float,
                best_person: dict) -> Incident:
    evid_path = str(OUTPUT_DIR / f"evidence_{CAMERA_ID}_loitering.jpg")
    inc = Incident(
        id          = f"{CAMERA_ID}_loitering_{frame_num}",
        camera_id   = CAMERA_ID,
        threat_type = "loitering",
        severity    = "medium",
        confidence  = best_person["conf"],
        timestamp   = f"demo+{sim_t - SIM_START:.1f}s",
        bbox        = best_person["bbox"],
        track_id    = best_person["track_id"],
        frame_path  = evid_path,
    )
    dispatcher.dispatch(inc)
    return inc


# ─── SPRITE CACHE ────────────────────────────────────────────────────────────

def _build_sprite_cache(height: int) -> dict:
    cache = {}

    def _get(step_key: int, walking: bool) -> tuple:
        key = (step_key, walking)
        if key not in cache:
            cache[key] = build_person_sprite(
                height,
                step=step_key * math.pi / 8,
                is_walking=walking,
            )
        return cache[key]

    print("Pre-building person sprites ...")
    for sk in range(16):
        _get(sk, True)
    _get(0, False)
    print(f"  {len(cache)} sprites cached.\n")
    return cache


# ─── MAIN DEMO ────────────────────────────────────────────────────────────────

def run(display: bool = False) -> None:
    print("\n" + "=" * 62)
    print("  LOITERING DETECTION DEMO  —  Moving Person")
    print("=" * 62)
    print("Loading YOLOv8n pretrained model ...")
    model = YOLO("yolov8n.pt")
    print(f"Model ready.  {len(model.names)} COCO classes.  Person = class 0\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    Path("incidents").mkdir(exist_ok=True)

    video_path = OUTPUT_DIR / "loitering_demo.mp4"
    writer     = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )

    background    = _build_background()
    sprite_cache  = _build_sprite_cache(220)

    def _sprite(step_key: int, walking: bool) -> tuple:
        key = (step_key, walking)
        return sprite_cache.get(key, sprite_cache[(0, False)])

    tracker    = LoiteringTracker()
    dispatcher = AlertDispatcher()

    total_frames = FPS * DURATION_S
    alert_fired  = False
    alert_frame  = None
    yolo_hits    = 0
    step_phase   = 0.0
    inc          = None

    print(f"Rendering {total_frames} frames  ({DURATION_S}s @ {FPS}fps) ...")
    print(f"Alert fires when dwell >= {LOITER_THRESH}s  "
          "(person stops at t=12s → alert at t=42s)\n")
    print(f"{'Frame':>6}  {'SimTime':>8}  {'Phase':>14}  "
          f"{'Dwell':>7}  {'YOLO':>6}  {'Alert':>7}")
    print("-" * 58)

    for frame_num in range(total_frames):
        sim_t = SIM_START + frame_num / FPS

        cx, bottom, phase = get_position(frame_num)
        is_walking = phase in ("entering", "walking_in", "browsing", "stopping")

        if is_walking:
            step_phase += 0.45
        step_key = int((step_phase % (2 * math.pi)) / (math.pi / 8)) % 16

        sp, alpha, sp_meta = _sprite(step_key if is_walking else 0, is_walking)

        # Build raw frame
        raw = background.copy()
        _draw_ground_shadow(raw, cx, bottom,
                            scale=0.6 + 0.4 * (bottom - 300) / 280)
        synth_bbox = composite_person(raw, sp, alpha, sp_meta, cx, bottom)

        # Per-frame noise
        noise = _rng.integers(-3, 4, raw.shape).astype(np.int16)
        raw   = np.clip(raw.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # YOLO inference
        yolo_person   = _run_yolo(model, raw)
        yolo_detected = yolo_person is not None
        if yolo_detected:
            yolo_hits += 1

        best_person = yolo_person or {
            "bbox": list(synth_bbox), "track_id": 1, "conf": 0.0
        }

        # Loitering tracker — only once person is stationary
        if frame_num >= LOITER_START_FRAME:
            det_cx = (best_person["bbox"][0] + best_person["bbox"][2]) / 2
            det_cy = (best_person["bbox"][1] + best_person["bbox"][3]) / 2
            with _patch("inference_engine.time.time", return_value=sim_t):
                loiter_s = tracker.update(best_person["track_id"], det_cx, det_cy)
        else:
            loiter_s = 0.0

        # Fire alert once at threshold
        if loiter_s >= LOITER_THRESH and not alert_fired:
            alert_fired = True
            alert_frame = frame_num
            inc = _fire_alert(dispatcher, frame_num, loiter_s, sim_t, best_person)

        # Annotate frame
        annotated = _render_annotations(
            frame         = raw,
            bbox          = best_person["bbox"],
            track_id      = best_person["track_id"],
            loiter_s      = loiter_s,
            alert_fired   = alert_fired,
            yolo_detected = yolo_detected,
            phase         = phase,
            frame_num     = frame_num,
            total_frames  = total_frames,
            yolo_hits     = yolo_hits,
        )

        # Save evidence JPEG at alert moment
        if alert_fired and alert_frame == frame_num and inc is not None:
            cv2.imwrite(
                str(OUTPUT_DIR / f"evidence_{CAMERA_ID}_loitering.jpg"),
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 93],
            )
            print(f"\n  *** ALERT at frame {frame_num} (t={frame_num/FPS:.1f}s) ***")
            print(f"      Dwell    : {loiter_s:.1f}s")
            print(f"      Track ID : {best_person['track_id']}")
            print(f"      Mode     : {'YOLO' if yolo_detected else 'INJECT'}")
            print(f"      Actions  : {inc.actions}")
            print(f"      JSON     : incidents/{inc.id}.json\n")

        writer.write(annotated)

        if display:
            cv2.imshow("Loitering Demo — Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_num % (FPS * 5) == 0:
            print(f"  {frame_num:5d}/{total_frames}"
                  f"  t={frame_num/FPS:5.1f}s"
                  f"  {phase:>15s}"
                  f"  dwell={loiter_s:5.1f}s"
                  f"  yolo={'HIT ' if yolo_detected else 'miss'}"
                  f"  alert={'YES' if alert_fired else 'no':>3}")

    writer.release()
    if display:
        cv2.destroyAllWindows()

    _print_summary(video_path, total_frames, yolo_hits, alert_fired, alert_frame, inc)


def _print_summary(video_path: Path, total_frames: int, yolo_hits: int,
                   alert_fired: bool, alert_frame: int | None,
                   inc: Incident | None) -> None:
    yolo_pct = yolo_hits / total_frames * 100
    print("\n" + "=" * 62)
    print("  DEMO COMPLETE")
    print("=" * 62)
    print(f"  Video            : {video_path}")
    print(f"  Frames rendered  : {total_frames}")
    print(f"  YOLO detections  : {yolo_hits} / {total_frames} ({yolo_pct:.1f}%)")
    if yolo_pct > 50:
        print(f"  Detection mode   : YOLO (real model — {yolo_pct:.0f}% of frames)")
    elif yolo_pct > 10:
        print(f"  Detection mode   : mixed ({yolo_pct:.0f}% YOLO, rest injected)")
    else:
        print("  Detection mode   : injection fallback "
              "(synthetic silhouette not recognised by COCO model)")
        print("  Note             : connect a real webcam or use a real person photo"
              " for full YOLO detection")
    print(f"  Loitering alert  : {'fired at 30s dwell ✓' if alert_fired else 'NOT fired ✗'}")
    if alert_fired and alert_frame is not None and inc is not None:
        print(f"  Alert at frame   : {alert_frame}  (t={alert_frame/FPS:.1f}s)")
        print(f"  Incident JSON    : incidents/{inc.id}.json")
    print()

    import subprocess
    try:
        subprocess.Popen(["open", str(video_path)])
        print(f"  Opening {video_path} ...")
    except Exception:
        print(f"  To view: open {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loitering Detection Demo — Moving Person"
    )
    parser.add_argument("--display", action="store_true",
                        help="Show live cv2 window while rendering")
    args = parser.parse_args()
    run(display=args.display)
