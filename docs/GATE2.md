# Gate 2A — Active interrogation, honest negative result

Date: 2026-11-04. Exploratory follow-up to Gate 1, **not** a preregistered discovery or demonstration of an embodied self-model.

## Question

Does adapting the **next weak ping** to evidence from earlier pings let an external observer detect a change in its own writable medium more efficiently than the best static query schedule?

The event is a previously unseen local A/B-like sine-packet write at one of nine candidate ring positions (2, 6, ..., 34). All mechanisms use the same 36-cell writable wave dynamics from beach.py and the same four listening ports. Weak reads do not alter sand.

## Strict division between learning and evaluation

1. Build response templates from **12 independent training beds**. Each training bed is given a random earlier history of 1–4 standard A/B writes. For each possible hidden position, simulate a packet and record the eight response changes relative to the bed's archived pre-event echoes. Retain only the per-class mean template; the inference controller sees neither training nor held-out sand.
2. A held-out bed is constructed from a new random background and random earlier A/B writes. The unknown writing site rotates evenly through the nine candidates, while its write amplitude varies ±10% and frequency varies ±0.003. These variations were **not** included in template calibration.
3. The archive contains **eight pre-event pings**, available to every policy. This baseline is essential to obtain the post-minus-pre change; its measurement cost is *not free*, even though common to all compared policies.
4. After the event, each policy receives only the selected probe delta. The same synthetic Gaussian noise matrix, generated once per test case, supplies observations for every policy. No policy may reuse a query.
5. Policy variants: **active** chooses the probe maximizing current posterior-weighted predicted template variance (not exact information gain); **fixed** ranks probes by that same score under the original uniform prior, without adapting; **random** uses a seeded random permutation. All three use the same calibrated Gaussian likelihood for classification and the same post-event query budget.
6. Noise standard deviation is a declared multiple of the median *training* inter-class template separation per feature; it is set without looking at test responses.

The policy is a **coded active heuristic** with a learned observation model, not a learned policy trained end-to-end and not a physical component inside the beach. Treating it as a self-aware or autonomous material memory would be an error.

## Reproduce

    python -m pip install -r requirements.txt
    python -m pytest -q
    OPENBLAS_NUM_THREADS=1 python active_probe.py --train 12 --test 81 --seed 37 --noise-ratio 0.5
    OPENBLAS_NUM_THREADS=1 python active_probe.py --train 12 --test 81 --seed 37 --noise-ratio 1.25
    OPENBLAS_NUM_THREADS=1 python active_probe.py --train 12 --test 81 --seed 37 --noise-ratio 2.5

Results are written to results/active_probe.json; successive commands replace that file. Noise magnitudes are relative, synthetic and dependent on this readout. Compute with deterministic NumPy seeds on CPU.

## Exploratory local results (81 held-out beds, seed 37)

Accuracy (higher is better). Chance is 1/9 = 11.1%.

| Noise ratio | Post-write queries | Adaptive | Fixed | Random |
|---|---:|---:|---:|---:|
| 0.5 | 2 | 95.1% | 95.1% | 90.1% |
| 0.5 | 4 | 97.5% | 96.3% | 96.3% |
| 1.25 | 2 | 86.4% | 87.7% | 80.2% |
| 1.25 | 4 | 92.6% | 91.4% | 86.4% |
| 2.5 | 2 | 80.2% | 79.0% | 64.2% |
| 2.5 | 4 | 87.7% | 87.7% | 71.6% |

**Verdict: active does not consistently outperform fixed.** At the hardest noise level the four-query accuracy is exactly tied; at medium noise, static is better with two queries. This is *positive evidence for informative probes* against a random policy in this toy, not evidence that adaptivity is valuable by itself.

No multi-seed uncertainty interval or preregistered threshold is claimed. Individual differences of 1/81 (1.2 percentage points) should not be interpreted as reliable advantages. The 8-query pre-event archive, matched access to calibration labels, deterministic forward physics, and externally programmed controller make this a limited test.

## Important next controls

- **Zero archive:** remove pre-event readings to measure how much gain relies on an independent historical reference. Also evaluate noisy archive measurements rather than a noiseless archive.
- **Cost accounting:** count the eight baseline reads plus training simulations and post-event reads. A learned physics emulator and oracle forward model are important comparisons.
- **Stronger static policy:** train a fixed multi-query order to optimize conditional information across the population; current fixed order is merely greedy marginal separability.
- **Nonstationarity:** independently shift write rules, material roughness, port placement and noise floor after calibration.
- **Self-model vs classifier:** forecast a future echo under an unobserved intervention, not just identify a labeled intervention location.
- **Embodiment:** put a writable confidence/error trace and probe routing into the beach itself. A Python-coded external Bayesian classifier is not an embodied self-index.

## Interpretation

Gate 1 showed that the current echoes help predict the effects of a *known* future A/B write. Gate 2A shows that a compact external observer can identify an *unknown* localized write from a few learned-diagnostic pings. However, the self-chosen pings are not yet clearly better than a calibrated static schedule.

This negative boundary is worth preserving: **a medium that is readable, a system that models it, and a system that benefits from choosing its own questions are three different claims.**
