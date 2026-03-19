"""
Security Inference Engine
==========================
Real-time threat detection on RTSP camera streams.
Runs YOLOv8 detection + multi-camera tracking + alert dispatch.

Usage:
  python inference_engine.py --model best.pt --sources rtsp://cam1,rtsp://cam2
"""

import argparse
import time
import json
import threading
import queue
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── THREAT CONFIG ────────────────────────────────────────────────────────────

THREAT_CONFIG = {
    "loitering": {
        "min_duration_s":   30,        # flag after 30s in same zone
        "severity":         "medium",
        "deter_action":     "audio_warning",
        "notify_personnel": True,
    },
    "brawl": {
        "min_duration_s":   0,         # immediate flag
        "severity":         "high",
        "deter_action":     "audio_alarm",
        "notify_personnel": True,
        "notify_police":    True,
    },
    "fire": {
        "min_duration_s":   0,
        "severity":         "critical",
        "deter_action":     "sprinkler_trigger",
        "notify_personnel": True,
        "notify_police":    True,
        "notify_fire":      True,
    },
    "smoke": {
        "min_duration_s":   5,
        "severity":         "high",
        "deter_action":     "audio_warning",
        "notify_personnel": True,
    },
    "abandoned_bag": {
        "min_duration_s":   60,
        "severity":         "high",
        "deter_action":     "audio_announcement",
        "notify_personnel": True,
        "notify_police":    True,
    },
    "crowd_surge": {
        "min_duration_s":   10,
        "severity":         "medium",
        "deter_action":     "crowd_speaker",
        "notify_personnel": True,
    },
    "weapon": {
        "min_duration_s":   0,
        "severity":         "critical",
        "deter_action":     "lock_gates",
        "notify_personnel": True,
        "notify_police":    True,
    },
    "vehicle_intrusion": {
        "min_duration_s":   0,
        "severity":         "high",
        "deter_action":     "barrier_deploy",
        "notify_personnel": True,
        "notify_police":    True,
    },
}

CONFIDENCE_THRESHOLD = 0.45
LOITERING_ZONE_RADIUS_PX = 80       # pixels; scale to your coverage area


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_name:  str
    confidence:  float
    bbox:        List[float]         # [x1, y1, x2, y2]
    track_id:    Optional[int] = None

@dataclass
class Incident:
    id:          str
    camera_id:   str
    threat_type: str
    severity:    str
    confidence:  float
    timestamp:   str
    bbox:        List[float]
    track_id:    Optional[int]
    frame_path:  Optional[str] = None
    actions:     List[str]     = field(default_factory=list)
    resolved:    bool          = False


# ─── LOITERING TRACKER ────────────────────────────────────────────────────────

class LoiteringTracker:
    """Tracks how long each person-ID stays in approximately the same location."""

    def __init__(self):
        self._tracks: Dict[int, dict] = {}

    def update(self, track_id: int, cx: float, cy: float) -> float:
        """Returns seconds the track has been in its current zone."""
        now = time.time()
        if track_id not in self._tracks:
            self._tracks[track_id] = {"cx": cx, "cy": cy, "since": now}
            return 0.0

        t = self._tracks[track_id]
        dist = ((cx - t["cx"])**2 + (cy - t["cy"])**2) ** 0.5
        if dist > LOITERING_ZONE_RADIUS_PX:
            # Moved to new zone — reset
            self._tracks[track_id] = {"cx": cx, "cy": cy, "since": now}
            return 0.0

        return now - t["since"]

    def evict_stale(self, active_ids: set):
        for tid in list(self._tracks.keys()):
            if tid not in active_ids:
                del self._tracks[tid]


# ─── ALERT DISPATCHER ─────────────────────────────────────────────────────────

class AlertDispatcher:
    """Handles deterrence actions and notifications. Replace stubs with real integrations."""

    def dispatch(self, incident: Incident):
        cfg = THREAT_CONFIG.get(incident.threat_type, {})
        actions = []

        deter = cfg.get("deter_action")
        if deter:
            self._trigger_deterrence(deter, incident)
            actions.append(f"deter:{deter}")

        if cfg.get("notify_personnel"):
            self._notify_personnel(incident)
            actions.append("notify:personnel")

        if cfg.get("notify_police"):
            self._notify_police(incident)
            actions.append("notify:police")

        if cfg.get("notify_fire"):
            self._notify_fire(incident)
            actions.append("notify:fire")

        incident.actions = actions
        log.info(f"[INCIDENT] {incident.severity.upper()} | {incident.threat_type} | cam:{incident.camera_id} | actions:{actions}")
        self._save_incident(incident)

    def _trigger_deterrence(self, action: str, incident: Incident):
        # Integrate with IoT controller (e.g., MQTT, REST API to speaker/gate/barrier)
        log.warning(f"  → DETER: {action} at {incident.camera_id}")

    def _notify_personnel(self, incident: Incident):
        # Push notification via FCM, PagerDuty, or WebSocket to dashboard
        log.warning(f"  → ALERT sent to security personnel")

    def _notify_police(self, incident: Incident):
        # Integrate with emergency dispatch API or SMS gateway
        log.warning(f"  → ALERT sent to police dispatch")

    def _notify_fire(self, incident: Incident):
        log.warning(f"  → ALERT sent to fire department")

    def _save_incident(self, incident: Incident):
        Path("incidents").mkdir(exist_ok=True)
        fpath = f"incidents/{incident.id}.json"
        with open(fpath, "w") as f:
            json.dump(asdict(incident), f, indent=2)


# ─── CAMERA PROCESSOR ─────────────────────────────────────────────────────────

class CameraProcessor:
    """Processes one camera stream in its own thread."""

    def __init__(
        self,
        camera_id:   str,
        source:      str,
        model:       YOLO,
        alert_queue: queue.Queue,
    ):
        self.camera_id   = camera_id
        self.source      = source
        self.model       = model
        self.alert_queue = alert_queue
        self.loitering   = LoiteringTracker()
        self._alerted:   Dict[str, float] = {}   # cooldown per track+threat
        self._running    = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info(f"Camera {self.camera_id} started — source: {self.source}")

    def stop(self):
        self._running = False

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            log.error(f"Cannot open: {self.source}")
            return

        frame_idx = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                log.warning(f"Camera {self.camera_id}: stream ended, retrying...")
                time.sleep(2)
                cap = cv2.VideoCapture(self.source)
                continue

            frame_idx += 1
            if frame_idx % 2 != 0:          # process every other frame
                continue

            self._process_frame(frame, frame_idx)

        cap.release()

    def _process_frame(self, frame: np.ndarray, frame_idx: int):
        results = self.model.track(
            frame,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        active_ids = set()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id     = int(box.cls[0])
                class_name = self.model.names[cls_id]
                conf       = float(box.conf[0])
                bbox       = box.xyxy[0].tolist()
                track_id   = int(box.id[0]) if box.id is not None else None

                det = Detection(class_name, conf, bbox, track_id)

                if track_id is not None:
                    active_ids.add(track_id)
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2

                    # Loitering check on any detected person
                    if class_name == "person":
                        loiter_s = self.loitering.update(track_id, cx, cy)
                        cfg = THREAT_CONFIG["loitering"]
                        if loiter_s >= cfg["min_duration_s"]:
                            self._maybe_raise(det, "loitering", frame, frame_idx, loiter_s)

                    # Direct threat classes
                    if class_name in THREAT_CONFIG and class_name != "person":
                        self._maybe_raise(det, class_name, frame, frame_idx)

        self.loitering.evict_stale(active_ids)

    def _maybe_raise(
        self,
        det:        Detection,
        threat:     str,
        frame:      np.ndarray,
        frame_idx:  int,
        duration_s: float = 0.0,
    ):
        """Rate-limit alerts to avoid spam (one per track+threat per 60s)."""
        key = f"{det.track_id}:{threat}"
        now = time.time()
        if now - self._alerted.get(key, 0) < 60:
            return
        self._alerted[key] = now

        cfg       = THREAT_CONFIG.get(threat, {})
        ts        = datetime.utcnow().isoformat()
        inc_id    = f"{self.camera_id}_{threat}_{int(now)}"
        frame_path = self._save_evidence(frame, inc_id)

        incident = Incident(
            id          = inc_id,
            camera_id   = self.camera_id,
            threat_type = threat,
            severity    = cfg.get("severity", "medium"),
            confidence  = det.confidence,
            timestamp   = ts,
            bbox        = det.bbox,
            track_id    = det.track_id,
            frame_path  = frame_path,
        )
        self.alert_queue.put(incident)

    def _save_evidence(self, frame: np.ndarray, inc_id: str) -> str:
        Path("evidence").mkdir(exist_ok=True)
        fpath = f"evidence/{inc_id}.jpg"
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return fpath


# ─── MAIN ORCHESTRATOR ────────────────────────────────────────────────────────

def run(model_path: str, sources: List[str]):
    log.info(f"Loading model: {model_path}")
    model      = YOLO(model_path)
    dispatcher = AlertDispatcher()
    alert_q    = queue.Queue()
    processors = []

    for i, src in enumerate(sources):
        cam_id = f"CAM-{i+1:02d}"
        proc   = CameraProcessor(cam_id, src, model, alert_q)
        proc.start()
        processors.append(proc)

    log.info(f"Monitoring {len(sources)} camera(s). Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                incident = alert_q.get(timeout=0.5)
                dispatcher.dispatch(incident)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        log.info("Shutting down...")
        for p in processors:
            p.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Security Inference Engine")
    parser.add_argument("--model",   default="best.pt",         help="Path to trained model weights")
    parser.add_argument("--sources", default="0",               help="Comma-separated RTSP URLs or device IDs")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",")]
    run(args.model, sources)
