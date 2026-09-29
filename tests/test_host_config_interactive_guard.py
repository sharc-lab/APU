"""host_config.py: per-host interactive_guard, query.exe user parsing, and the resulting console-session state that
Lab.row() (t2s_overnight.py) stamps onto every row on evo-x2 instead of refusing to run."""
import io
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import host_config as hc  # noqa: E402

QUERY_USER_ACTIVE = """ USERNAME              SESSIONNAME        ID  STATE   IDLE TIME  LOGON TIME
>ritz                  console             1  Active      none   9/28/2026 2:38 AM
"""

QUERY_USER_IDLE_4MIN = """ USERNAME              SESSIONNAME        ID  STATE   IDLE TIME  LOGON TIME
>ritz                  console             1  Active      4      9/28/2026 2:38 AM
"""

QUERY_USER_IDLE_12MIN = """ USERNAME              SESSIONNAME        ID  STATE   IDLE TIME  LOGON TIME
>ritz                  console             1  Active      12     9/28/2026 2:38 AM
"""

QUERY_USER_IDLE_HOURS = """ USERNAME              SESSIONNAME        ID  STATE   IDLE TIME  LOGON TIME
>ritz                  console             1  Active      2:15   9/28/2026 2:38 AM
"""

QUERY_USER_IDLE_DAYS = """ USERNAME              SESSIONNAME        ID  STATE   IDLE TIME  LOGON TIME
>ritz                  console             1  Active      1+03:15   9/26/2026 2:38 AM
"""

QUERY_USER_NONE = "No User exists for *\n"


# ---------------------------------------------------------------------------------------------------- HOSTS table
def test_interactive_guard_flags():
    assert hc.HOSTS["EVO-T2S"]["interactive_guard"] is True
    assert hc.HOSTS["EVO-X2"]["interactive_guard"] is False


# ---------------------------------------------------------------------------------------------------- parse_idle_time
def test_parse_idle_time_none_is_zero():
    assert hc.parse_idle_time("none") == 0.0


def test_parse_idle_time_minutes():
    assert hc.parse_idle_time("12") == 12 * 60


def test_parse_idle_time_hours_minutes():
    assert hc.parse_idle_time("2:15") == 2 * 3600 + 15 * 60


def test_parse_idle_time_days_hours_minutes():
    assert hc.parse_idle_time("1+03:15") == 86400 + 3 * 3600 + 15 * 60


def test_parse_idle_time_unparseable_or_empty():
    assert hc.parse_idle_time("") is None
    assert hc.parse_idle_time(None) is None
    assert hc.parse_idle_time("garbage") is None


# ---------------------------------------------------------------------------------------------------- parse_query_user
def test_parse_query_user_active_console_session():
    sessions = hc.parse_query_user(QUERY_USER_ACTIVE)
    assert len(sessions) == 1
    assert sessions[0]["username"] == "ritz"
    assert sessions[0]["sessionname"] == "console"
    assert sessions[0]["state"] == "Active"
    assert sessions[0]["idle_s"] == 0.0


def test_parse_query_user_no_user_returns_empty():
    assert hc.parse_query_user(QUERY_USER_NONE) == []
    assert hc.parse_query_user("") == []


# ---------------------------------------------------------------------------------------------------- get_console_session_state
def test_console_session_state_active_now_is_user_active():
    state = hc.get_console_session_state(query_user_fn=lambda: QUERY_USER_ACTIVE)
    assert state["console_session_state"] == "Active"
    assert state["console_idle_s"] == 0.0
    assert state["user_active"] is True


def test_console_session_state_idle_under_5min_is_user_active():
    state = hc.get_console_session_state(query_user_fn=lambda: QUERY_USER_IDLE_4MIN)
    assert state["console_idle_s"] == 4 * 60
    assert state["user_active"] is True  # 240s < 300s threshold


def test_console_session_state_idle_over_5min_is_not_user_active():
    state = hc.get_console_session_state(query_user_fn=lambda: QUERY_USER_IDLE_12MIN)
    assert state["console_idle_s"] == 12 * 60
    assert state["user_active"] is False  # 720s >= 300s threshold

    state2 = hc.get_console_session_state(query_user_fn=lambda: QUERY_USER_IDLE_HOURS)
    assert state2["console_idle_s"] == 2 * 3600 + 15 * 60
    assert state2["user_active"] is False


def test_console_session_state_multi_day_idle_is_not_user_active():
    state = hc.get_console_session_state(query_user_fn=lambda: QUERY_USER_IDLE_DAYS)
    assert state["user_active"] is False


def test_console_session_state_no_session_at_all():
    state = hc.get_console_session_state(query_user_fn=lambda: QUERY_USER_NONE)
    assert state == {"console_session_state": None, "console_idle_s": None, "user_active": False}


# ---------------------------------------------------------------------------------------------------- enforce_or_record_interactive_session
def test_enforce_raises_on_guarded_host_when_someone_is_logged_in():
    host_cfg = {"interactive_guard": True}
    try:
        hc.enforce_or_record_interactive_session(host_cfg, query_user_fn=lambda: QUERY_USER_ACTIVE)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "another interactive session" in str(e)


def test_enforce_does_not_raise_on_guarded_host_when_nobody_logged_in():
    host_cfg = {"interactive_guard": True}
    result = hc.enforce_or_record_interactive_session(host_cfg, query_user_fn=lambda: QUERY_USER_NONE)
    assert result == {}


def test_enforce_never_raises_on_unguarded_host_and_returns_state():
    host_cfg = {"interactive_guard": False}
    result = hc.enforce_or_record_interactive_session(host_cfg, query_user_fn=lambda: QUERY_USER_ACTIVE)
    assert result["user_active"] is True


# ---------------------------------------------------------------------------------------------------- get_ollama_loaded_model
def test_get_ollama_loaded_model_none_when_unreachable():
    assert hc.get_ollama_loaded_model(base_url="http://127.0.0.1:1") is None


def test_get_ollama_loaded_model_returns_name_when_a_model_is_loaded():
    body = b'{"models": [{"name": "qwen3:8b", "size": 5000000000}]}'
    fake_response = io.BytesIO(body)
    fake_response.__enter__ = lambda self=fake_response: fake_response
    fake_response.__exit__ = lambda self, *a: False
    with mock.patch("urllib.request.urlopen", return_value=fake_response):
        assert hc.get_ollama_loaded_model() == "qwen3:8b"


def test_get_ollama_loaded_model_none_when_nothing_loaded():
    body = b'{"models": []}'
    fake_response = io.BytesIO(body)
    fake_response.__enter__ = lambda self=fake_response: fake_response
    fake_response.__exit__ = lambda self, *a: False
    with mock.patch("urllib.request.urlopen", return_value=fake_response):
        assert hc.get_ollama_loaded_model() is None
