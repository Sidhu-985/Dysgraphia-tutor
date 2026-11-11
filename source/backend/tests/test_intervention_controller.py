from app.interventions.intervention_controller import build_intervention
from app.config import InterventionConfig
from app.state.session_context import SessionContext
from app.dysgraphia_rl_bridge import ACTION_NAMES


def _session():
    return SessionContext(session_id="s1", child_id="c1", difficulty_level=2)


def test_decrease_difficulty_clamped_at_min():
    cfg = InterventionConfig(difficulty_min=0, difficulty_max=4, difficulty_step=1)
    session = _session()
    session.difficulty_level = 0
    intervention = build_intervention(0, ACTION_NAMES[0], session, cfg)
    assert intervention.params["difficulty_level"] == 0
    assert session.difficulty_level == 0


def test_increase_difficulty_clamped_at_max():
    cfg = InterventionConfig(difficulty_min=0, difficulty_max=4, difficulty_step=1)
    session = _session()
    session.difficulty_level = 4
    intervention = build_intervention(1, ACTION_NAMES[1], session, cfg)
    assert intervention.params["difficulty_level"] == 4


def test_enlarge_template_increases_scale_within_bounds():
    cfg = InterventionConfig(template_enlarge_factor=1.5, template_scale_max=2.0)
    session = _session()
    session.template_scale = 1.0
    build_intervention(3, ACTION_NAMES[3], session, cfg)
    assert session.template_scale == 1.5

    for _ in range(5):
        build_intervention(3, ACTION_NAMES[3], session, cfg)
    assert session.template_scale <= cfg.template_scale_max


def test_switch_to_tracing_mode_updates_session_mode():
    session = _session()
    build_intervention(5, ACTION_NAMES[5], session)
    assert session.mode == "tracing"


def test_switch_to_free_writing_mode_updates_session_mode():
    session = _session()
    build_intervention(6, ACTION_NAMES[6], session)
    assert session.mode == "free_writing"


def test_suggest_break_returns_break_type_with_duration():
    cfg = InterventionConfig(break_duration_seconds=90)
    session = _session()
    intervention = build_intervention(7, ACTION_NAMES[7], session, cfg)
    assert intervention.type == "break"
    assert intervention.params["duration_seconds"] == 90


def test_continue_no_change_is_a_no_op_on_session_state():
    session = _session()
    before = session.to_dict()
    intervention = build_intervention(9, ACTION_NAMES[9], session)
    after = session.to_dict()
    assert intervention.type == "continue"
    assert before["difficulty_level"] == after["difficulty_level"]
    assert before["template_scale"] == after["template_scale"]


def test_all_ten_actions_produce_a_valid_intervention():
    session = _session()
    for action_id in range(10):
        intervention = build_intervention(action_id, ACTION_NAMES[action_id], session)
        assert intervention.action_id == action_id
        assert isinstance(intervention.message, str) and intervention.message
        assert isinstance(intervention.params, dict)
