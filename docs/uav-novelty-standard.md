# UAV novelty standard

Novelty is a relation to a stated prior-art boundary, not a feeling and not a score improvement.

## Five statuses

1. **Exact prior art** — the proposed mechanism or architecture is already implemented or explicitly described.
2. **Functional equivalent** — different words or components implement the same relevant mechanism or interface.
3. **Component overlap** — individual parts are known, but the proposed interaction may still be distinct; the burden is to show why the interaction is not obvious and is testable.
4. **Adjacent prior art** — related technology or task, but not the same mechanism or deployment claim.
5. **Apparently open under the search boundary** — no collision found after the documented search; this is the strongest claim CURI-UAV may make before supervisor review.

## Collision tests

Ask:

- Is this simply a VLM/VLA placed on top of an existing planner?
- Is it only a new prompt, backbone, quantization, dataset, simulator, benchmark, or environment?
- Is the module interaction already present in a close paper under another name?
- Does the change alter the training target, information flow, action interface, planner/controller coupling, safety behavior, or deployment constraint?
- Does it address a documented failure and make a falsifiable prediction?
- Can an ablation remove the proposed mechanism while holding model, data, simulator, and tuning budget fixed?

If the answer is “no” to the last two questions, the idea is not ready for a novelty claim.

## Accepted claim form

Use:

> Under the search boundary [date, sources, query families, citation expansion], the closest functional prior art is [A/B]. Those systems implement [mechanism]. This proposal changes [specific mechanism/interface/training signal/deployment constraint] by [difference]. The difference is expected to affect [failure mode], tested by [matched baseline and ablation]. Unverified concurrent work remains [boundary].

Do not use “first,” “novel,” or “never done” without supervisor approval and a complete collision report.
