"""
Unit tests for LoiteringTracker
================================
Tests the time-based zone-dwell logic that drives loitering detection.
"""
import time
from unittest.mock import patch

import pytest

from inference_engine import LoiteringTracker, LOITERING_ZONE_RADIUS_PX


class TestLoiteringTrackerFirstSeen:
    def test_new_track_returns_zero(self, loitering_tracker):
        """First time a track is seen, dwell time is 0."""
        duration = loitering_tracker.update(track_id=1, cx=320.0, cy=240.0)
        assert duration == 0.0

    def test_second_call_at_same_position_returns_positive_duration(self, loitering_tracker):
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(1, 320.0, 240.0)

        t1 = t0 + 15.0
        with patch("inference_engine.time.time", return_value=t1):
            duration = loitering_tracker.update(1, 320.0, 240.0)

        assert abs(duration - 15.0) < 0.01

    def test_dwell_exceeds_threshold_after_30s(self, loitering_tracker):
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(1, 200.0, 300.0)

        t1 = t0 + 31.0
        with patch("inference_engine.time.time", return_value=t1):
            duration = loitering_tracker.update(1, 200.0, 300.0)

        assert duration >= 30.0

    def test_dwell_not_threshold_within_30s(self, loitering_tracker):
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(1, 200.0, 300.0)

        t1 = t0 + 20.0
        with patch("inference_engine.time.time", return_value=t1):
            duration = loitering_tracker.update(1, 200.0, 300.0)

        assert duration < 30.0


class TestLoiteringTrackerZoneLogic:
    def test_large_movement_resets_timer(self, loitering_tracker):
        """Moving more than LOITERING_ZONE_RADIUS_PX pixels resets dwell to 0."""
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(1, 100.0, 100.0)

        t1 = t0 + 45.0       # long dwell
        big_move = LOITERING_ZONE_RADIUS_PX + 5
        with patch("inference_engine.time.time", return_value=t1):
            duration = loitering_tracker.update(1, 100.0 + big_move, 100.0)

        assert duration == 0.0

    def test_small_movement_within_zone_continues_dwell(self, loitering_tracker):
        """Moving less than LOITERING_ZONE_RADIUS_PX keeps dwell accumulating."""
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(1, 200.0, 200.0)

        t1 = t0 + 35.0
        small_move = LOITERING_ZONE_RADIUS_PX - 5
        with patch("inference_engine.time.time", return_value=t1):
            duration = loitering_tracker.update(1, 200.0 + small_move, 200.0)

        assert duration >= 35.0

    def test_movement_exactly_at_boundary_stays_in_zone(self, loitering_tracker):
        """Exactly at radius boundary — Euclidean distance == radius — is NOT a reset (> check)."""
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(1, 0.0, 0.0)

        t1 = t0 + 31.0
        # Euclidean distance == LOITERING_ZONE_RADIUS_PX (not strictly greater)
        with patch("inference_engine.time.time", return_value=t1):
            duration = loitering_tracker.update(1, float(LOITERING_ZONE_RADIUS_PX), 0.0)

        # dist == radius → condition `dist > radius` is False → stays in zone
        assert duration >= 31.0

    def test_diagonal_movement_uses_euclidean_distance(self, loitering_tracker):
        """Diagonal displacement > radius triggers a reset."""
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(1, 0.0, 0.0)

        t1 = t0 + 40.0
        # 60,60 → distance = sqrt(7200) ≈ 84.85 > 80
        with patch("inference_engine.time.time", return_value=t1):
            duration = loitering_tracker.update(1, 60.0, 60.0)

        assert duration == 0.0


class TestLoiteringTrackerMultipleTracks:
    def test_multiple_tracks_are_independent(self, loitering_tracker):
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(1, 100.0, 100.0)
            loitering_tracker.update(2, 500.0, 400.0)

        t1 = t0 + 35.0
        with patch("inference_engine.time.time", return_value=t1):
            d1 = loitering_tracker.update(1, 100.0, 100.0)  # stays put
            d2 = loitering_tracker.update(2, 500.0 + LOITERING_ZONE_RADIUS_PX + 10, 400.0)  # moves away

        assert d1 >= 35.0     # track 1 loitering
        assert d2 == 0.0      # track 2 reset

    def test_evict_stale_removes_inactive_tracks(self, loitering_tracker):
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            loitering_tracker.update(10, 100.0, 100.0)
            loitering_tracker.update(20, 200.0, 200.0)

        loitering_tracker.evict_stale(active_ids={20})        # track 10 gone
        assert 10 not in loitering_tracker._tracks
        assert 20 in loitering_tracker._tracks

    def test_evict_stale_empty_active_set_clears_all(self, loitering_tracker):
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            for tid in range(5):
                loitering_tracker.update(tid, float(tid * 100), 0.0)

        loitering_tracker.evict_stale(active_ids=set())
        assert len(loitering_tracker._tracks) == 0

    def test_evict_stale_with_all_active_keeps_all(self, loitering_tracker):
        t0 = 1_000_000.0
        with patch("inference_engine.time.time", return_value=t0):
            for tid in (1, 2, 3):
                loitering_tracker.update(tid, float(tid * 100), 0.0)

        loitering_tracker.evict_stale(active_ids={1, 2, 3})
        assert len(loitering_tracker._tracks) == 3
