import numpy as np

from beach import Beach
from active_probe import (
    PROBES, POSITIONS, hidden_write, read_bank,
    choose_probe, update_posterior, strategies, run,
)


def test_hidden_write_and_nonwriting_read():
    b = Beach()
    before = b.sand.copy()
    hidden_write(b, POSITIONS[0])
    assert not np.array_equal(before, b.sand)
    after = b.sand.copy()
    read_bank(b)
    assert np.array_equal(after, b.sand)


def test_query_choice_changes_with_belief():
    # Probe 0 divides class 0 from classes 1/2; probe 1 divides 1 from 2.
    template = np.zeros((3, 2, 1))
    template[:, 0, 0] = [20, 0, 0]
    template[:, 1, 0] = [0, 10, -10]
    sigma = np.ones(2)
    first = choose_probe(np.array([.98, .01, .01]), template, sigma, [0, 1])
    second = choose_probe(np.array([.01, .495, .495]), template, sigma, [0, 1])
    assert first == 0 and second == 1


def test_belief_moves_to_matching_hypothesis():
    template = np.zeros((3, 2, 1))
    template[:, 0, 0] = [-1, 0, 1]
    posterior = np.ones(3) / 3
    result = update_posterior(posterior, template, np.array([1.]),
                              0, np.array([.1, .1]))
    assert int(result.argmax()) == 2


def test_equal_budget_no_duplicate_probes():
    rng = np.random.default_rng(4)
    template = rng.normal(size=(len(POSITIONS), len(PROBES), 48))
    observation = template[2] + rng.normal(0, .5, size=(len(PROBES), 48))
    result = strategies(template, lambda j: observation[j], .5,
                        np.random.default_rng(4), 4)
    for predictions, selected, posterior in result.values():
        assert len(predictions) == len(selected) == 4
        assert len(set(selected)) == 4
        assert np.isclose(posterior.sum(), 1.)


def test_heldout_experiment_is_deterministic():
    a = run(train_beds=3, test_beds=9, seed=7, budgets=(1, 2),
            noise_ratio=1.25)
    b = run(train_beds=3, test_beds=9, seed=7, budgets=(1, 2),
            noise_ratio=1.25)
    assert a == b
    assert a['config']['archive_queries_per_episode'] == len(PROBES)
    for name in ('active', 'fixed', 'random'):
        for budget in ('1', '2'):
            assert 0 <= a['accuracy'][name][budget] <= 1


def test_invalid_budget_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        run(train_beds=1, test_beds=1, budgets=(len(PROBES)+1,))
