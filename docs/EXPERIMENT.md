# Gate 1 — Observable self-prediction / a first self-indexing instrument

Date: 2026-11-04. **Exploratory/calibration** experiment, not a preregistered discovery.

## What is being tested

A passive wave ring contains 36 mutable link coefficients. A and B write events change these coefficients according to a known local wave-stress rule. A query is one weak, nonwriting pulse at port 0 with a selectable phase and frequency; four output sensors record echoes.

The external estimator is shown pairs `(echo_before, next_write_event) -> Δecho_after`. It never sees the 36 sand coefficients. We fit a ridge-regressed map from the current echo, interactions with event A/B, and fixed probe set to next-step echo change. Train and test use different episode seeds and random histories; every episode has its own initial bed.

The model is **not controlling its own probes**, **not a physical process inside the water**, **not trained by reinforcement**, and **not identifying true sediment maps**. It is a state observer that approximates one effective transition operator.

## Numerical protocol

- Config: 36 cells, periodic boundary, `friction=0.14`, `leak=0.018`, `c0=0.14`, `beta=2.3`.
- Link sand begins as spatially averaged positive random values; a fresh bed is sampled for each episode. No random noise is added to the resulting echoes.
- Each event has 100 discrete writing steps. Fast water and flux reset between events, leaving sand as the only inter-event persistent state.
- Each fingerprint consists of four fixed probes `(0.04,0)`, `(0.07,0)`, `(0.11,0)`, `(0.07,π/2)`; each yields four output channels binned into 12 temporal windows. Total 192 features.
- Default split: 60 train episodes and 20 completely separate test episodes, 6 successive writes per episode, train seed 13 / test seed 10013. There are 360 train and 120 test transitions.
- Ridge regularization `alpha=0.1` for all learned models. Input feature scaling fit on training data only. A/B event-only, no-event, and shuffled-fingerprint controls; zero-change naive baseline.
- Metric: `mean[(Δecho_predict - Δecho_true)^2] / mean[Δecho_true^2]`. This is not an R² based on centered target variance.
- Finite-difference sensitivity uses a +1e-4 perturbation of each of 36 sand links. The quoted counts are singular values >1e-4 in **unstandardized physical echo units**, NOT dynamical Hankel rank. Adding queries can improve singular directions without revealing their semantic meaning.

## Results and vulnerabilities

The default seed shows that observations of the *present* medium can make future read-response changes predictable, because the write dynamics are deterministic, smooth and partially observable. The event-only baseline is notably worse. But sample size matters: a much smaller training set can make the read-conditioned regressor overfit, so calibration does not imply arbitrary data efficiency. Phase and multiscale probes need not always add independent information.

AB and BA order differ in sand at roughly `1.11e-4` L2, but their fingerprint difference is just `3.10e-7` L2. The present instrument does **not** establish noise-robust recovery of write order. Investigating nonlinear, phase-gated, matched-energy interactive probes is the next hypothesis, not a positive result here.

## Proposed gates (to freeze before future experiments)

1. **Generalization and noise:** Sweep random seeds, episode lengths, noise floors and shifted event amplitudes. Split by full episode. Compare read-conditioned ridge with event-only, direct full-sand oracle, and matched-parameter autoregressive controls.
2. **Closed-loop probing:** A self-probing controller chooses among probes based on predictive disagreement, with a fixed probe/energy budget. Compare random probes, fixed probes, and an independent archival observer. The controller must forecast held-out responses to interventions.
3. **Order retrieval:** AB vs BA classification on new stochastic writer realizations. Compare single probe, multi-frequency, phase-conditioned, and two-ping nonlinear interaction readers. Lock analysis before evaluating held-out sets.
4. **Model embodiment:** Implement predictive error and probe selection using the same evolving medium rather than an external Python matrix. Only then claim a physical system that *learns to predict its own responses*.
5. **Gauge:** Distinguish sensor-sensitive sand directions from weak/null directions; reparameterizations of the hidden material state should not change output-based criteria. Do not confuse finite local Jacobian rank with dynamical-state Hankel rank.
