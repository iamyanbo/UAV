# Invention mode versus replication mode

The finance CURI inherited a useful but conservative loop: find evidence, reproduce or implement a bounded test, preserve the result, and verify it. That is appropriate when the knowledge base is sparse or when the claim is whether a known mechanism works. It is insufficient as the only mode for novel UAV research because it can converge on replication without creating a defensible new interaction.

CURI-UAV therefore has two explicit modes:

## Evidence mode

Use this for literature cards, architecture reconstruction, source verification, reproduction, and failure analysis. The output is knowledge and a bounded result—not a contribution claim.

## Invention mode

Enter only after a documented limitation and a technology-frontier card exist. The lead must form:

```text
UAV failure
  -> enabling mechanism from presearch
  -> proposed UAV-specific interaction or adaptation
  -> falsifiable prediction
  -> smallest executable experiment
  -> prior-art adversary
```

The invention is not the enabling method by itself. It must change a mechanism that matters to UAV navigation: information flow, training target, action interface, planner/controller coupling, safety behavior, or a measurable deployment constraint.

The executor may still reproduce a close baseline first. That reproduction is an instrument for attribution and feasibility; it is not the final research output. If the adversary finds that the proposed interaction is already implemented or is an obvious composition, the pipeline records it as an implementation option or drops it.

## Why this is not unconstrained brainstorming

Invention is constrained by evidence, compute, latency, safety, and falsifiability. It is allowed to combine mechanisms from outside UAV literature, but every transfer claim must state what is preserved, what breaks, and how it will be tested. The system may conclude that no novel implementation is justified.
