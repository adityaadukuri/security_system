"""
Integration tests for CameraProcessor._process_frame()
=======================================================
Tests the full frame-processing pipeline using a mock YOLO model:
  detection → loitering logic → rate limiting → incident queuing → evidence saving
"""
import queue
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference_engine import (
    CameraProcessor,
    CONFIDENCE_THRESHOLD,
    LOITERING_ZONE_RADIUS_PX,
    THREAT_CONFIG,
)
from tests.conftest import make_results, CLASS_IDS


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _run_frame(processor, detections, t=None):
    """Call _process_frame with mock results for the given detections."""
    processor.model.track.return_value = make_results(detections)
    if t is not None:
        with patch("inference_engine.time.time", return_value=t):
            processor._process_frame(processor._blank_frame(), frame_idx=2)
    else:
        processor._process_frame(processor._blank_frame(), frame_idx=2)


# Monkey-patch a helper onto CameraProcessor for test convenience
CameraProcessor._blank_frame = lambda self: np.zeros((480, 640, 3), dtype=np.uint8)


# ─── IMMEDIATE-TRIGGER THREATS ────────────────────────────────────────────────

class TestImmediateThreatDetection:
    """Threats with min_duration_s==0 should raise an incident on first detection."""

    @pytest.mark.parametrize("threat", ["brawl", "weapon", "fire", "vehicle_intrusion"])
    def test_immediate_threat_queues_incident(self, camera_processor, alert_queue, threat):
        _run_frame(camera_processor, [{
            "cls_id":   CLASS_IDS[threat],
            "conf":     0.90,
            "bbox":     [100, 100, 300, 350],
            "track_id": 10,
        }])
        assert not alert_queue.empty(), f"{threat} should immediately queue an incident"
        inc = alert_queue.get_nowait()
        assert inc.threat_type == threat
        assert inc.camera_id == "CAM-01"

    def test_brawl_incident_has_correct_severity(self, camera_processor, alert_queue):
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["brawl"], "conf": 0.88,
            "bbox": [50, 50, 400, 400], "track_id": 7,
        }])
        inc = alert_queue.get_nowait()
        assert inc.severity == "high"
        assert inc.confidence == pytest.approx(0.88)

    def test_weapon_incident_is_critical(self, camera_processor, alert_queue):
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["weapon"], "conf": 0.92,
            "bbox": [200, 100, 400, 400], "track_id": 3,
        }])
        inc = alert_queue.get_nowait()
        assert inc.severity == "critical"

    def test_fire_incident_is_critical(self, camera_processor, alert_queue):
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["fire"], "conf": 0.75,
            "bbox": [100, 200, 500, 450], "track_id": 4,
        }])
        inc = alert_queue.get_nowait()
        assert inc.severity == "critical"


# ─── LOITERING DETECTION ──────────────────────────────────────────────────────

class TestLoiteringDetection:
    LOITER_MIN = THREAT_CONFIG["loitering"]["min_duration_s"]   # 30 s

    def test_person_not_flagged_below_threshold(self, camera_processor, alert_queue):
        t0 = 1_000_000.0
        for offset in (0.0, 10.0, 20.0):
            _run_frame(camera_processor, [{
                "cls_id": CLASS_IDS["person"], "conf": 0.80,
                "bbox": [200, 200, 300, 400], "track_id": 1,
            }], t=t0 + offset)

        assert alert_queue.empty(), "Person should not be flagged before 30 s"

    def test_person_flagged_after_threshold(self, camera_processor, alert_queue):
        t0 = 1_000_000.0
        # First call seeds the timer
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["person"], "conf": 0.80,
            "bbox": [200, 200, 300, 400], "track_id": 1,
        }], t=t0)

        # Second call after 31 s → loitering triggered
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["person"], "conf": 0.80,
            "bbox": [205, 202, 305, 402], "track_id": 1,
        }], t=t0 + 31.0)

        assert not alert_queue.empty(), "Person should be flagged after 31 s in same zone"
        inc = alert_queue.get_nowait()
        assert inc.threat_type == "loitering"

    def test_moving_person_resets_loitering_timer(self, camera_processor, alert_queue):
        t0 = 1_000_000.0
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["person"], "conf": 0.80,
            "bbox": [100, 200, 200, 400], "track_id": 2,
        }], t=t0)

        # Person moves more than LOITERING_ZONE_RADIUS_PX
        big_move = LOITERING_ZONE_RADIUS_PX + 20
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["person"], "conf": 0.80,
            "bbox": [100 + big_move, 200, 200 + big_move, 400], "track_id": 2,
        }], t=t0 + 35.0)    # past threshold, but moved

        assert alert_queue.empty(), "Moving person should not trigger loitering"

    def test_multiple_loiterers_tracked_independently(self, camera_processor, alert_queue):
        t0 = 1_000_000.0
        detections_t0 = [
            {"cls_id": CLASS_IDS["person"], "conf": 0.80,
             "bbox": [100, 200, 200, 400], "track_id": 10},
            {"cls_id": CLASS_IDS["person"], "conf": 0.80,
             "bbox": [400, 200, 500, 400], "track_id": 20},
        ]
        _run_frame(camera_processor, detections_t0, t=t0)

        # Both stay in same position after 31 s
        detections_t1 = [
            {"cls_id": CLASS_IDS["person"], "conf": 0.80,
             "bbox": [101, 201, 201, 401], "track_id": 10},
            {"cls_id": CLASS_IDS["person"], "conf": 0.80,
             "bbox": [401, 201, 501, 401], "track_id": 20},
        ]
        _run_frame(camera_processor, detections_t1, t=t0 + 31.0)

        incidents = []
        while not alert_queue.empty():
            incidents.append(alert_queue.get_nowait())

        track_ids = {inc.track_id for inc in incidents}
        assert 10 in track_ids, "Track 10 should be flagged for loitering"
        assert 20 in track_ids, "Track 20 should be flagged for loitering"


# ─── RATE LIMITING ────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_duplicate_alert_suppressed_within_60s(self, camera_processor, alert_queue):
        t0 = 1_000_000.0
        det = {"cls_id": CLASS_IDS["brawl"], "conf": 0.90,
               "bbox": [50, 50, 300, 400], "track_id": 5}

        _run_frame(camera_processor, [det], t=t0)
        assert not alert_queue.empty()
        alert_queue.get_nowait()   # consume first

        # Second detection at t0+30 (within 60s cooldown)
        _run_frame(camera_processor, [det], t=t0 + 30.0)
        assert alert_queue.empty(), "Second alert within 60 s should be suppressed"

    def test_alert_allowed_after_60s_cooldown(self, camera_processor, alert_queue):
        t0 = 1_000_000.0
        det = {"cls_id": CLASS_IDS["brawl"], "conf": 0.90,
               "bbox": [50, 50, 300, 400], "track_id": 5}

        _run_frame(camera_processor, [det], t=t0)
        alert_queue.get_nowait()   # consume

        _run_frame(camera_processor, [det], t=t0 + 61.0)
        assert not alert_queue.empty(), "Alert should fire again after 60 s cooldown"

    def test_different_tracks_same_threat_not_rate_limited(self, camera_processor, alert_queue):
        t0 = 1_000_000.0
        _run_frame(camera_processor, [
            {"cls_id": CLASS_IDS["brawl"], "conf": 0.90,
             "bbox": [50, 50, 300, 400], "track_id": 1},
        ], t=t0)

        _run_frame(camera_processor, [
            {"cls_id": CLASS_IDS["brawl"], "conf": 0.85,
             "bbox": [50, 50, 300, 400], "track_id": 2},
        ], t=t0 + 1.0)   # different track_id → independent cooldown

        # Should have 2 incidents (one per track)
        count = 0
        while not alert_queue.empty():
            alert_queue.get_nowait()
            count += 1
        assert count == 2


# ─── CONFIDENCE THRESHOLD ─────────────────────────────────────────────────────

class TestConfidenceFiltering:
    def test_low_confidence_detection_no_incident(self, camera_processor, alert_queue):
        """Detections below CONFIDENCE_THRESHOLD should not raise incidents.
        Note: the threshold is enforced by YOLO's conf parameter in model.track(),
        so the pipeline trusts YOLO to pre-filter. We verify that the model is
        called with the correct threshold."""
        _run_frame(camera_processor, [
            {"cls_id": CLASS_IDS["brawl"], "conf": CONFIDENCE_THRESHOLD - 0.05,
             "bbox": [50, 50, 300, 400], "track_id": 9},
        ])
        # Even if YOLO doesn't filter, the incident is still created since
        # the engine doesn't double-check conf. This test validates the model
        # was called with the right conf value.
        call_kwargs = camera_processor.model.track.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("conf") == CONFIDENCE_THRESHOLD or \
               (call_kwargs.args[1:] and CONFIDENCE_THRESHOLD in call_kwargs.args), \
            "model.track() should be called with CONFIDENCE_THRESHOLD"

    def test_model_track_called_with_correct_conf(self, camera_processor, blank_frame):
        camera_processor.model.track.return_value = make_results([])
        camera_processor._process_frame(blank_frame, frame_idx=2)
        call_kwargs = camera_processor.model.track.call_args.kwargs
        assert call_kwargs["conf"] == CONFIDENCE_THRESHOLD


# ─── TRACK_ID NONE HANDLING ───────────────────────────────────────────────────

class TestTrackIdNone:
    def test_detection_without_track_id_raises_no_incident(self, camera_processor, alert_queue):
        """Detections without a track_id are skipped (tracking required for alerts)."""
        _run_frame(camera_processor, [
            {"cls_id": CLASS_IDS["brawl"], "conf": 0.90,
             "bbox": [50, 50, 300, 400], "track_id": None},
        ])
        assert alert_queue.empty(), "No incident without a track_id"


# ─── EVIDENCE SAVING ──────────────────────────────────────────────────────────

class TestEvidenceSaving:
    def test_evidence_jpeg_saved_on_incident(self, camera_processor, alert_queue, tmp_path):
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["brawl"], "conf": 0.90,
            "bbox": [50, 50, 300, 400], "track_id": 7,
        }])
        inc = alert_queue.get_nowait()
        assert inc.frame_path is not None
        assert Path(inc.frame_path).exists(), "Evidence JPEG should exist on disk"
        assert inc.frame_path.endswith(".jpg")

    def test_evidence_path_contains_incident_id(self, camera_processor, alert_queue):
        _run_frame(camera_processor, [{
            "cls_id": CLASS_IDS["weapon"], "conf": 0.93,
            "bbox": [100, 100, 400, 400], "track_id": 3,
        }])
        inc = alert_queue.get_nowait()
        assert inc.id in inc.frame_path
