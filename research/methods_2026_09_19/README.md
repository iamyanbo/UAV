# Proposed method components

These are implementable starting points written by the operator at the user's request. They are not trained UAV systems, validated improvements, complete reproductions, or novelty certificates. Read the three method cards in `presearch/injections/2026-09-19-invention-reboot/` before extending them.

`method_kernels.py` implements:

1. **SemanticCorrectionTransport**: an online history summary and trainable arrival-time correction operator, an exact chronological replay target, and combined representation/decision distillation. It implements the proposed late-correction learning interface. An actual fast policy, slow visual-semantic adapter, training data and rendered closed-loop integration remain to be built. Only one outstanding request with a fixed checkpoint is covered initially; concurrent/revised messages require explicit lineage handling. Summarizer work must count toward total latency.
2. **ObservationBranchModel**: an action-conditioned likelihood over future observation branches, coupled by Bayes' rule to persistent hypothesis weights, predicted latent observations, real-observation assimilation, and observation-contingent continuation costs. Branch masses and posteriors conserve the expected prior by construction. This algebra is established Bayesian inference. The research hypothesis concerns learning this interface in a compact predictive representation for UAV sensing/control. Hypothesis creation, temporal dynamics/association, visual encoding, calibration, training and deployment remain substantive work.
3. **Reconstruction curriculum primitives**: observed-parameter projection, selection of capture-consistent/revealable challenges, and a one-step information-matched regret calculation. These are ingredients for a proposed joint reconstruction/policy training method. They do not yet implement a 3DGS renderer, a completion generator, a recurrent teacher, or the alternating training loop. Masking reconstruction parameters alone cannot guarantee identical rendered captures; all observed views must also be checked.

The kernels take feature tensors and run on CPU. Encoded inputs are not automatically visual/Qwen/JEPA features. The executor should attach actual visual features and document which encoders are frozen or trained. Training the optional shared/nested model is a later architectural extension; the present code does not claim to be a nested Qwen model.

Verification command from repository root:

```powershell
py -3.10 -m unittest discover -s test -p test_uav_method_kernels.py -v
```

Checks cover zero-correction/zero-age behavior, chronological replay, teacher gradient isolation, trainable gradients, Bayesian conservation, uninformative sensing, actual observation assimilation, non-privileged branch decisions, observed-scene constraints and irreducible-regret handling. They establish implementation consistency only. No success-rate, latency advantage, learned-model quality or publication claim follows from them.

For actual training respect the existing 60% local VRAM preference; CPU-first kernel tests do not allocate CUDA memory. The research provider remains Spark. The method cards require trained checkpoints and integrated behavior before claiming a UAV improvement, while allowing the lead to revise or replace these proposals when evidence warrants it.
