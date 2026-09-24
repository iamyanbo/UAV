# Handoff model development contract

The authoritative execution policy is docs/uav-scientific-contract.md, revision uav-model-development-2026-09-20-v2. Use configured Spark/Qwen. Read the staged .research-guidance/HANDOFF.md, original ideas, and source architectures.

Develop the internal models: visual JEV-like scoring, action-conditioned JEPA/world prediction, VLM/VLA and shared fast/slow computation, geometric representations, and their learning inside a flight system. Memory participates in the model. The former restriction to interface mechanisms and requirement for SURVIVES_AUDIT before implementation are withdrawn.

Use one persistent artifact program. Inspect primary papers as implementations, build a faithful reusable baseline, propose a precise changed neural computation or training procedure, train/integrate it, and evaluate representative flight behavior. Useful development can continue with unresolved novelty; claims must remain accurate. Preserve checkpoints and PROJECT.md across tasks. Follow the current contract's partial/representative evidence rules and 60% local VRAM cap.
