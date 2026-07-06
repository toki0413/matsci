"""Tests for the progressive SCF convergence strategy chain."""

from huginn.tools.sim.convergence_strategies import (
    ConvergenceStrategy,
    STRATEGY_CHAIN,
    apply_strategy,
    get_next_strategy,
)


def test_strategy_chain_ordering():
    """Strategies should be ordered by cost_level, 1 to 5."""
    costs = [s.cost_level for s in STRATEGY_CHAIN]
    assert costs == [1, 2, 3, 4, 5]
    # strictly increasing
    for i in range(1, len(costs)):
        assert costs[i] > costs[i - 1]


def test_first_strategy_is_cheapest():
    """The first entry in the chain should have cost_level 1."""
    assert STRATEGY_CHAIN[0].cost_level == 1
    assert STRATEGY_CHAIN[0].name == "reduce_mixing_alpha"
    # it should reduce mixing amplitude
    assert "AMIX" in STRATEGY_CHAIN[0].param_changes
    assert "mixing_beta" in STRATEGY_CHAIN[0].param_changes


def test_get_next_strategy_skips_attempted():
    """get_next_strategy should skip names already in attempted_strategies."""
    params = {"ALGO": "Fast", "NELM": 60, "ENCUT": 520}
    attempted = ["reduce_mixing_alpha"]
    next_s = get_next_strategy(params, attempted)
    assert next_s is not None
    assert next_s.name == "change_algo"
    assert next_s.cost_level == 2

    # skip two
    next_s = get_next_strategy(params, ["reduce_mixing_alpha", "change_algo"])
    assert next_s is not None
    assert next_s.name == "increase_nelm"


def test_exhausted_strategies_returns_none():
    """When all strategies are attempted, get_next_strategy returns None."""
    params = {"ALGO": "Fast"}
    all_names = [s.name for s in STRATEGY_CHAIN]
    result = get_next_strategy(params, all_names)
    assert result is None


def test_get_next_strategy_accepts_objects():
    """get_next_strategy should accept ConvergenceStrategy objects too."""
    params = {"mixing_beta": 0.7}
    attempted = [STRATEGY_CHAIN[0]]  # pass the object, not the name
    next_s = get_next_strategy(params, attempted)
    assert next_s is not None
    assert next_s.name == "change_algo"


def test_apply_strategy_vasp():
    """apply_strategy should write only VASP-style (uppercase) keys."""
    params = {"ALGO": "Fast", "NELM": 60, "ENCUT": 520}
    strategy = STRATEGY_CHAIN[1]  # change_algo
    result = apply_strategy(params, strategy)
    assert result is params  # modified in-place
    assert params["ALGO"] == "Normal"
    assert params["IALGO"] == 38
    # QE keys should NOT be injected into a VASP params dict
    assert "mixing_type" not in params
    assert "mixing_beta" not in params


def test_apply_strategy_qe():
    """apply_strategy should write only QE-style (lowercase) keys."""
    params = {"mixing_beta": 0.7, "ecutwfc": 40.0}
    strategy = STRATEGY_CHAIN[0]  # reduce_mixing_alpha
    apply_strategy(params, strategy)
    assert params["mixing_beta"] == 0.4
    # VASP keys should NOT be injected into a QE params dict
    assert "AMIX" not in params
    assert "BMIX" not in params


def test_last_resort_strategy():
    """The most expensive strategy should switch to exact diagonalization."""
    last = STRATEGY_CHAIN[-1]
    assert last.cost_level == 5
    assert last.param_changes["ALGO"] == "Exact"
    assert last.param_changes["NELM"] == 200
