# Sol novelty-adversary audit: UAV research ideas

This is a bounded literature and architecture audit, not an implementation task and not an idea-generation task.

## Objective

Determine whether any of the current UAV research families contains a defensible algorithmic or systems contribution after implementation-level comparison with prior work. The answer may be **zero survivors**. Do not create replacement ideas merely to fill the output.

Read all of the following before forming a verdict:

- `reports/VLM UAV navigation research ideas.md`
- `reports/UAV Method Proposals - 2026-09-19.md`
- `reports/UAV VLM Literature Review - Full.md`
- `reports/UAV Research Agenda - Navigation Tracking Exploration.md`
- `presearch/injections/2026-09-19-invention-reboot/prior-art.md`
- `presearch/injections/2026-09-19-invention-reboot/execution-contract.md`
- the preserved Method 1 and Method 2 artifacts and CURI outcomes, if they are relevant

The current families to audit are:

1. Transported or replayed semantic corrections under inference/communication delay.
2. Action-conditioned predictive observation branches, belief-space prediction or JEPA-style planning.
3. Capture-constrained environment completion coupled to sensing/navigation policy learning.

Also audit the earlier discarded or component-level ideas in the reports if they are plausible substitutes: small local JEV-like decision models, fast/slow VLM-VLA stacks, latency-conditioned motion, World Atlas or real-to-sim reconstruction, 3D Gaussian splats, memory, active perception, exploration, tracking and speed-aware control.

## Required search boundary

Search beyond UAV-VLN. Inspect primary or official sources in aerial navigation, robotics, autonomous driving, VLM/VLA, cloud-edge robotics, delayed/out-of-order sensing, memory and belief filtering, active perception, world models, JEPA, sim-to-real, adversarial environment generation, neural rendering, 3DGS and real-time control. Search conference proceedings, official project pages, code and supplementary material where available. A title or abstract match is not enough: inspect the actual input/output interface, update loop, training objective, scheduling rule, controller coupling and evaluation protocol.

## Required output

Write `reports/UAV Novelty Adversary Audit - 2026-09-19.md` and preserve source notes or URLs in inspectable workspace-relative files. Do not modify the candidate method files to make them appear novel.

For every candidate, provide a table with:

- candidate claim;
- closest exact prior art;
- closest functional equivalent from another field;
- what that prior method actually implements;
- exact overlap at the module, training, runtime and evaluation levels;
- the smallest remaining difference;
- whether the difference is a new mechanism, an application transfer, an engineering improvement, or only a new benchmark/environment;
- whether it is implementable on an RTX 3060 Ti under the project’s 60% VRAM limit;
- the strongest falsifier;
- verdict: `DROP`, `REPRODUCTION`, `INCREMENTAL`, or `SURVIVES_AUDIT`.

For every `SURVIVES_AUDIT` candidate, require a one-sentence residual contribution of this form:

> Existing method X fails on condition Y because of limitation Z; our mechanism A changes interface/objective/update rule B, which should change measurable outcome C while holding D fixed.

Reject the candidate if the sentence can be implemented by configuring, combining or renaming an existing method. Reject “VLM/VLA/JEPA/world model for UAVs”, “new simulator”, “new dataset”, “smaller model”, “latency-aware”, “memory”, “3DGS”, “sim-to-real” or “active perception” as sufficient novelty on their own.

## Strict anti-hallucination rules

- Cite every decisive claim to a primary or official source, with title, URL, publication status and the inspected implementation location.
- Distinguish verified implementation, abstract/project-page discovery, inference and unresolved search lead.
- Do not call something novel because no exact phrase was found.
- Do not treat a small pilot result as novelty evidence.
- Do not run code, train models, create a benchmark or propose a replacement method in this task.
- Do not return a status memo without the comparison table and explicit drop/survive verdicts.
- If evidence is incomplete, state `UNRESOLVED`, identify the missing source or implementation, and do not promote the idea.

## Final decision

End with:

1. `SURVIVORS: 0` or the exact surviving candidates;
2. a ranked list of the three most dangerous prior-art overlaps;
3. which existing CURI experiments should be retained only as baselines or negative evidence;
4. what one follow-up source check would most likely change the verdict;
5. a recommendation on whether Astra should be used again, and under what restricted role.

This audit is the gate before any new CURI experiment. The lead must not queue implementation work from it automatically. A surviving candidate still requires a separate design review before code or training is authorized.
