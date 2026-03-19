"""
Unit tests for AlertDispatcher
================================
Verifies deterrence actions, notification routing, and incident persistence
for each threat type defined in THREAT_CONFIG.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from inference_engine import AlertDispatcher, Incident, THREAT_CONFIG


def _make_incident(threat_type: str, track_id: int = 1, conf: float = 0.80) -> Incident:
    return Incident(
        id=f"CAM-01_{threat_type}_999",
        camera_id="CAM-01",
        threat_type=threat_type,
        severity=THREAT_CONFIG.get(threat_type, {}).get("severity", "medium"),
        confidence=conf,
        timestamp="2024-06-01T10:00:00",
        bbox=[50.0, 50.0, 300.0, 400.0],
        track_id=track_id,
    )


# ─── INCIDENT PERSISTENCE ────────────────────────────────────────────────────

class TestIncidentPersistence:
    def test_dispatch_saves_json_file(self, alert_dispatcher, tmp_path):
        inc = _make_incident("brawl")
        alert_dispatcher.dispatch(inc)

        incident_file = tmp_path / "incidents" / f"{inc.id}.json"
        assert incident_file.exists(), "Incident JSON file should be created"

    def test_saved_json_contains_correct_fields(self, alert_dispatcher, tmp_path):
        inc = _make_incident("fire")
        alert_dispatcher.dispatch(inc)

        incident_file = tmp_path / "incidents" / f"{inc.id}.json"
        data = json.loads(incident_file.read_text())

        assert data["id"] == inc.id
        assert data["camera_id"] == "CAM-01"
        assert data["threat_type"] == "fire"
        assert data["severity"] == "critical"
        assert data["confidence"] == pytest.approx(0.80)
        assert data["track_id"] == 1

    def test_dispatch_populates_actions_list(self, alert_dispatcher):
        inc = _make_incident("brawl")
        alert_dispatcher.dispatch(inc)

        assert len(inc.actions) > 0
        action_types = [a.split(":")[0] for a in inc.actions]
        assert "deter" in action_types
        assert "notify" in action_types

    def test_multiple_incidents_saved_independently(self, alert_dispatcher, tmp_path):
        inc1 = _make_incident("brawl",  track_id=1)
        inc2 = _make_incident("weapon", track_id=2)
        inc1.id = "CAM-01_brawl_111"
        inc2.id = "CAM-01_weapon_222"

        alert_dispatcher.dispatch(inc1)
        alert_dispatcher.dispatch(inc2)

        assert (tmp_path / "incidents" / "CAM-01_brawl_111.json").exists()
        assert (tmp_path / "incidents" / "CAM-01_weapon_222.json").exists()


# ─── NOTIFICATION ROUTING ─────────────────────────────────────────────────────

class TestNotificationRouting:
    def test_brawl_notifies_personnel_and_police(self, alert_dispatcher):
        inc = _make_incident("brawl")
        with patch.object(alert_dispatcher, "_notify_personnel") as p, \
             patch.object(alert_dispatcher, "_notify_police") as po, \
             patch.object(alert_dispatcher, "_notify_fire") as f:
            alert_dispatcher.dispatch(inc)

        p.assert_called_once_with(inc)
        po.assert_called_once_with(inc)
        f.assert_not_called()

    def test_fire_notifies_personnel_police_and_fire_dept(self, alert_dispatcher):
        inc = _make_incident("fire")
        with patch.object(alert_dispatcher, "_notify_personnel") as p, \
             patch.object(alert_dispatcher, "_notify_police") as po, \
             patch.object(alert_dispatcher, "_notify_fire") as f:
            alert_dispatcher.dispatch(inc)

        p.assert_called_once()
        po.assert_called_once()
        f.assert_called_once()

    def test_loitering_notifies_personnel_only(self, alert_dispatcher):
        inc = _make_incident("loitering")
        with patch.object(alert_dispatcher, "_notify_personnel") as p, \
             patch.object(alert_dispatcher, "_notify_police") as po, \
             patch.object(alert_dispatcher, "_notify_fire") as f:
            alert_dispatcher.dispatch(inc)

        p.assert_called_once()
        po.assert_not_called()
        f.assert_not_called()

    def test_smoke_notifies_personnel_only(self, alert_dispatcher):
        inc = _make_incident("smoke")
        with patch.object(alert_dispatcher, "_notify_personnel") as p, \
             patch.object(alert_dispatcher, "_notify_police") as po:
            alert_dispatcher.dispatch(inc)

        p.assert_called_once()
        po.assert_not_called()

    def test_weapon_notifies_personnel_and_police(self, alert_dispatcher):
        inc = _make_incident("weapon")
        with patch.object(alert_dispatcher, "_notify_police") as po:
            alert_dispatcher.dispatch(inc)
        po.assert_called_once()

    def test_abandoned_bag_notifies_personnel_and_police(self, alert_dispatcher):
        inc = _make_incident("abandoned_bag")
        with patch.object(alert_dispatcher, "_notify_police") as po:
            alert_dispatcher.dispatch(inc)
        po.assert_called_once()

    def test_vehicle_intrusion_notifies_personnel_and_police(self, alert_dispatcher):
        inc = _make_incident("vehicle_intrusion")
        with patch.object(alert_dispatcher, "_notify_police") as po:
            alert_dispatcher.dispatch(inc)
        po.assert_called_once()

    def test_crowd_surge_notifies_personnel_only(self, alert_dispatcher):
        inc = _make_incident("crowd_surge")
        with patch.object(alert_dispatcher, "_notify_personnel") as p, \
             patch.object(alert_dispatcher, "_notify_police") as po:
            alert_dispatcher.dispatch(inc)
        p.assert_called_once()
        po.assert_not_called()


# ─── DETERRENCE ACTIONS ──────────────────────────────────────────────────────

class TestDeterrenceActions:
    def _dispatch_and_get_deter_arg(self, dispatcher, threat_type):
        inc = _make_incident(threat_type)
        called_with = []

        def capture_deter(action, incident):
            called_with.append(action)

        with patch.object(dispatcher, "_trigger_deterrence", side_effect=capture_deter):
            dispatcher.dispatch(inc)

        return called_with

    def test_brawl_triggers_audio_alarm(self, alert_dispatcher):
        actions = self._dispatch_and_get_deter_arg(alert_dispatcher, "brawl")
        assert "audio_alarm" in actions

    def test_fire_triggers_sprinkler(self, alert_dispatcher):
        actions = self._dispatch_and_get_deter_arg(alert_dispatcher, "fire")
        assert "sprinkler_trigger" in actions

    def test_weapon_triggers_lock_gates(self, alert_dispatcher):
        actions = self._dispatch_and_get_deter_arg(alert_dispatcher, "weapon")
        assert "lock_gates" in actions

    def test_loitering_triggers_audio_warning(self, alert_dispatcher):
        actions = self._dispatch_and_get_deter_arg(alert_dispatcher, "loitering")
        assert "audio_warning" in actions

    def test_vehicle_intrusion_deploys_barrier(self, alert_dispatcher):
        actions = self._dispatch_and_get_deter_arg(alert_dispatcher, "vehicle_intrusion")
        assert "barrier_deploy" in actions

    def test_unknown_threat_does_not_crash(self, alert_dispatcher):
        """Unknown threat types should be handled gracefully (no THREAT_CONFIG entry)."""
        inc = _make_incident("alien_invasion")
        inc.severity = "unknown"
        # Should not raise
        alert_dispatcher.dispatch(inc)
        assert inc.actions == []    # no config → no actions


# ─── ACTIONS LIST CONTENT ────────────────────────────────────────────────────

class TestActionsListContent:
    def test_brawl_actions_contain_deter_and_notify_entries(self, alert_dispatcher):
        inc = _make_incident("brawl")
        alert_dispatcher.dispatch(inc)

        assert "deter:audio_alarm"    in inc.actions
        assert "notify:personnel"     in inc.actions
        assert "notify:police"        in inc.actions

    def test_fire_actions_contain_all_three_notifies(self, alert_dispatcher):
        inc = _make_incident("fire")
        alert_dispatcher.dispatch(inc)

        assert "deter:sprinkler_trigger" in inc.actions
        assert "notify:personnel"        in inc.actions
        assert "notify:police"           in inc.actions
        assert "notify:fire"             in inc.actions
