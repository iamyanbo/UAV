# UAV literature and prior-art protocol

This protocol is an audit trail, not a completeness certificate. It is used whenever CURI-UAV makes an architecture, limitation, or novelty claim.

## Search passes

1. Exact identity: title, method name, acronym, authors, project URL, code repository, dataset.
2. Vocabulary expansion: UAV, drone, aerial, autonomous flight, aerial VLN, visual-language navigation, VLM, VLA, embodied AI, waypoint, trajectory, action chunking, memory, active perception, world model, JEPA, latent dynamics, 3DGS, Gaussian splatting, MPC, MPPI, safety filter, sim-to-real, asynchronous planning, latency, stale action.
3. Functional equivalent: same input/output interface or same failure mode under different terminology.
4. Component interaction: memory plus navigation, VLM plus MPC/MPPI, VLA plus low-level controller, world model plus action selection, fast confidence model plus slow planner, real-to-sim reconstruction plus training.
5. Source expansion: conference and workshop proceedings, arXiv, supplementary material, official project pages, released code, datasets, benchmark pages, theses, technical reports, patents where relevant.
6. Citation graph: backward citations from the closest papers and forward citations from scholarly indexes and project pages.
7. Negative search: query the proposed mechanism as a problem statement and as an ablation/failure mode, not only as the candidate's name.

## Architecture card

```text
Identity / primary source:
Task and language interface:
Sensors and privileged information:
Representation / map / memory:
Model role: perception | grounding | memory | waypoint/action | world prediction | end-to-end control | selector
Model inputs and outputs:
Planner and controller:
Training data and objective:
Inference schedule and asynchronous path:
Latency, hardware, parameters, and action age:
Simulator / real platform / sim-to-real:
Evaluation and baselines:
Failure cases and missing ablations:
Code/supplement verified:
What this rules out:
What remains open:
```

Do not replace unknown fields with assumptions. “Not reported” is itself a limitation.

## Evidence grading

- Primary: paper, supplement, official code, official project/dataset documentation.
- Secondary: survey or review used to discover sources, not to settle architecture details.
- Tertiary: search result, blog, or discussion used only as a lead.

Every durable claim names the source and the inspected location. Every missing detail is marked unavailable, not inferred.

## Coverage report

Before a novelty synthesis, report the date, databases/search engines, exact query families, venues/source types, citation expansion performed, inaccessible sources, concurrent-work uncertainty, and the final search boundary. The safe conclusion is “apparently open under this boundary,” never “not done anywhere.”
