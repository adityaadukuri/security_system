"""
Shared fixtures for the security system test suite.
"""
import sys
import os
import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_engine import (
    LoiteringTracker,
    AlertDispatcher,
    CameraProcessor,
    Detection,
    Incident,
    THREAT_CONFIG,
    CONFIDENCE_THRESHOLD,
)


# ─── MOCK YOLO HELPERS ────────────────────────────────────────────────────────

def make_box(cls_id: int, conf: float, bbox, track_id=None):
    """Return a mock YOLO Box object matching the API used in _process_frame."""
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [conf]
    box.xyxy = [np.array(bbox, dtype=float)]   # xyxy[0].tolist() required
    box.id = [track_id] if track_id is not None else None
    return box


def make_results(detections: list):
    """
    Build a mock list-of-Results object as returned by model.track().

    detections: list of dicts
        {cls_id, conf, bbox=[x1,y1,x2,y2], track_id (optional)}
    """
    r = MagicMock()
    r.boxes = [
        make_box(d["cls_id"], d["conf"], d["bbox"], d.get("track_id"))
        for d in detections
    ]
    return [r]


# ─── CLASS NAME → ID MAP ──────────────────────────────────────────────────────

CLASS_IDS = {
    "person":              0,
    "loitering":           1,
    "brawl":               2,
    "abandoned_bag":       3,
    "fire":                4,
    "smoke":               5,
    "crowd_surge":         6,
    "weapon":              7,
    "vehicle_intrusion":   8,
    "graffiti_vandalism":  9,
    "suspicious_package": 10,
}


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def loitering_tracker():
    return LoiteringTracker()


@pytest.fixture
def alert_dispatcher(tmp_path, monkeypatch):
    """AlertDispatcher with incidents/ written to tmp_path."""
    monkeypatch.chdir(tmp_path)
    return AlertDispatcher()


@pytest.fixture
def mock_model():
    """Minimal mock YOLO model with security class names."""
    m = MagicMock()
    m.names = {v: k for k, v in CLASS_IDS.items()}
    return m


@pytest.fixture
def alert_queue():
    return queue.Queue()


@pytest.fixture
def camera_processor(mock_model, alert_queue, tmp_path, monkeypatch):
    """CameraProcessor with evidence/ and incidents/ in tmp_path."""
    monkeypatch.chdir(tmp_path)
    return CameraProcessor("CAM-01", "mock_source", mock_model, alert_queue)


@pytest.fixture
def blank_frame():
    """640×480 black frame (3-channel uint8)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_incident():
    return Incident(
        id="CAM-01_brawl_1234567890",
        camera_id="CAM-01",
        threat_type="brawl",
        severity="high",
        confidence=0.87,
        timestamp="2024-01-01T12:00:00",
        bbox=[100.0, 150.0, 300.0, 400.0],
        track_id=5,
    )
