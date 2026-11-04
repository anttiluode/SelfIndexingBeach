"""Gate 2: active nonwriting interrogation after an unknown localized write.

The controller is external. Templates are calibrated on training beds, then held-out
sand is hidden. A query callback returns only the selected post-write echo delta.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from beach import Beach

POSITIONS = tuple(range(2, 35, 4))  # nine candidate intervention sites
PROBES = tuple((f, phase) for f in (.035, .055, .075, .105)
               for phase in (0., np.pi / 2))


def hidden_write(b, position, amplitude=.24, frequency=.061):
    """Use the original nonlinear wave-stress scour rule at an unknown port."""
    b.reset_water()
    for t in range(b.p.write_steps):
        env = np.sin(np.pi * (t + 1) / (b.p.write_steps + 1)) ** 2
        b.step(position, amplitude * env * np.sin(2 * np.pi * frequency * t),
               allow_write=True)
    b.reset_water()


def background(rng):
    rough = rng.uniform(0, .11, size=36)
    bed = Beach(sand=(rough + np.roll(rough, 1) + np.roll(rough, -1)) / 3)
    for _ in range(int(rng.integers(1, 5))):
        bed.write('A' if rng.integers(0, 2) else 'B')
    return bed


def read_bank(b):
    """8 nonwriting queries, each with 12 bins x 4 listening ports."""
    return np.stack([b.fingerprint((q,)) for q in PROBES])


def calibration(n, seed):
    """Supervised templates from independent beds; not available at test time."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        b = background(rng)
        before = read_bank(b)
        responses = []
        for pos in POSITIONS:
            candidate = b.copy()
            hidden_write(candidate, pos)
            responses.append(read_bank(candidate) - before)
        deltas.append(responses)
    arr = np.asarray(deltas)  # train bed, hypothesis, probe, feature
    return arr.mean(axis=0)


def choose_probe(posterior, template, sigma, available):
    """Posterior-weighted template variance / noise: a heuristic, NOT exact MI."""
    p = posterior / posterior.sum()
    choices = []
    for j in available:
        mean = np.tensordot(p, template[:, j, :], axes=(0, 0))
        spread = np.sum(p[:, None] * (template[:, j, :] - mean)**2)
        choices.append(spread / sigma[j]**2)
    return available[int(np.argmax(choices))]


def update_posterior(posterior, template, observation, j, sigma):
    logp = np.log(np.maximum(posterior, 1e-300))
    logp -= np.sum((template[:, j, :] - observation)**2, axis=1) / (2 * sigma[j]**2)
    logp -= logp.max()
    prob = np.exp(logp)
    return prob / prob.sum()


def strategies(template, query, noise_sigma, rng, budget):
    """Each policy obtains observations ONLY through the chosen-query callback."""
    sigma = np.full(len(PROBES), noise_sigma)
    prior = np.ones(len(POSITIONS)) / len(POSITIONS)
    fixed_order = []
    remaining = list(range(len(PROBES)))
    for _ in range(len(PROBES)):
        j = choose_probe(prior, template, sigma, remaining)
        fixed_order.append(j)
        remaining.remove(j)

    rankings = {}
    for policy in ('active', 'fixed', 'random'):
        remaining = list(range(len(PROBES)))
        posterior = prior.copy()
        predictions, chosen = [], []
        random_order = list(rng.permutation(len(PROBES)))
        for k in range(budget):
            if policy == 'active':
                j = choose_probe(posterior, template, sigma, remaining)
            elif policy == 'fixed':
                j = fixed_order[k]
            else:
                j = random_order[k]
            assert j in remaining
            remaining.remove(j)
            chosen.append(j)
            posterior = update_posterior(posterior, template, query(j), j, sigma)
            predictions.append(int(np.argmax(posterior)))
        rankings[policy] = (predictions, chosen, posterior)
    return rankings


def run(train_beds=12, test_beds=80, seed=37, budgets=(1, 2, 3, 4),
        noise_ratio=1.25):
    if not (train_beds > 0 and test_beds > 0 and
            0 < noise_ratio and all(1 <= k <= len(PROBES) for k in budgets)):
        raise ValueError('Require positive beds/noise and budget in probe range')
    templates = calibration(train_beds, seed)
    # Noise is set from TRAINING inter-class separation; no held-out peeking.
    pairwise = np.array([
        np.linalg.norm(templates[i] - templates[j], axis=-1)
        for i in range(len(POSITIONS)) for j in range(i+1, len(POSITIONS))
    ])
    noise_sigma = noise_ratio * float(np.median(pairwise)) / np.sqrt(templates.shape[-1])
    rng = np.random.default_rng(seed + 100000)
    correct = {k: {str(n): [] for n in budgets}
               for k in ('active', 'fixed', 'random')}
    picks = {k: np.zeros(len(PROBES), dtype=int) for k in correct}

    for case in range(test_beds):
        beach = background(rng)
        site = case % len(POSITIONS)  # balanced candidate labels
        archive = read_bank(beach)   # 8 PRE-event probes, same for all policies
        damaged = beach.copy()
        amp = .24 * float(rng.uniform(.9, 1.1))
        freq = .061 + float(rng.uniform(-.003, .003))
        hidden_write(damaged, POSITIONS[site], amplitude=amp, frequency=freq)
        truth = read_bank(damaged) - archive
        noisy = truth + rng.normal(0, noise_sigma, size=truth.shape)
        # Precompute a shared noise realization, but only return the queried row.
        for n in budgets:
            policies = strategies(templates, lambda j: noisy[j], noise_sigma,
                                  np.random.default_rng(seed+case+3000), n)
            for name, (pred, chosen, posterior) in policies.items():
                correct[name][str(n)].append(int(pred[-1] == site))
                picks[name][chosen[-1]] += 1

    accuracy = {name: {n: float(np.mean(values))
                       for n, values in table.items()}
                for name, table in correct.items()}
    return {
        'config': {
            'seed': seed, 'train_beds': train_beds, 'test_beds': test_beds,
            'probes': [list(map(float, q)) for q in PROBES],
            'positions': list(POSITIONS), 'budgets': list(budgets),
            'noise_ratio': noise_ratio, 'noise_sigma': noise_sigma,
            'archive_queries_per_episode': len(PROBES)
        },
        'accuracy': accuracy,
        'last_selected_probe_counts': {n: v.tolist() for n, v in picks.items()},
        'chance': 1 / len(POSITIONS)
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=12)
    ap.add_argument('--test', type=int, default=80)
    ap.add_argument('--seed', type=int, default=37)
    ap.add_argument('--noise-ratio', type=float, default=1.25)
    args = ap.parse_args()
    result = run(train_beds=args.train, test_beds=args.test,
                 seed=args.seed, noise_ratio=args.noise_ratio)
    Path('results').mkdir(exist_ok=True)
    Path('results/active_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
