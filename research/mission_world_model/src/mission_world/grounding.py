"""Frozen official Qwen2.5-VL-3B AWQ, prefix-only visual mission grounding."""
import json
from pathlib import Path
import re
import torch
from PIL import Image
from .contracts import GroundedMission
from .upstream import sha256


class QwenGrounder:
    def __init__(self, root):
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        root = Path(root)
        checkpoint = root / "weights/qwen-vl"
        digest = sha256(checkpoint / "model.safetensors")
        if digest != "72014d68fe47d4974e21fcab0cabef2e90de711f9fb1a179c5fa1400e71672d7":
            raise RuntimeError("Qwen checkpoint checksum mismatch")
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False,
                                                       min_pixels=256*28*28, max_pixels=256*28*28)
        # The AWQ checkpoint explicitly excludes its dense visual tower from
        # quantization. Keep that tower on CPU; all AWQ layers must remain CUDA.
        # Some Transformers versions reject *any* CPU device globally. Relax that
        # validator for this single nonquantized tower only, not quantized blocks.
        from transformers.quantizers.quantizer_awq import AwqQuantizer
        original = AwqQuantizer.validate_environment
        mapping = {"visual": "cpu", "model": "cuda:0", "lm_head": "cuda:0"}
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        # The released model ties lm_head to the dense token embedding; it has
        # no quantized lm_head weights. Exclude it explicitly for this HF loader.
        config.quantization_config["modules_to_not_convert"] = ["visual", "lm_head"]
        def validate_tower_offload(instance, *args, **kwargs):
            actual = kwargs.get("device_map")
            if actual == mapping:
                checked = dict(kwargs); checked["device_map"] = {"": "cuda:0"}
                return original(instance, *args, **checked)
            return original(instance, *args, **kwargs)
        AwqQuantizer.validate_environment = validate_tower_offload
        try:
            self.model, loading_info = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True,
                config=config, torch_dtype=torch.float16, device_map=mapping, low_cpu_mem_usage=True,
                attn_implementation="sdpa", output_loading_info=True)
        finally:
            AwqQuantizer.validate_environment = original
        self.model.eval().requires_grad_(False)
        if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Qwen did not load strictly: {loading_info}")
        # device_map='cpu' normally means CPU *storage* with execution on the
        # main GPU. Restore the dense tower and run it genuinely on CPU instead.
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(self.model.visual, recurse=True)
        self.model.visual.to(device="cpu", dtype=torch.float32)
        def cpu_visual_inputs(module, args, kwargs):
            def convert(value):
                return value.to(device="cpu", dtype=torch.float32 if value.is_floating_point() else value.dtype) if torch.is_tensor(value) else value
            return tuple(convert(value) for value in args), {key: convert(value) for key, value in kwargs.items()}
        self.model.visual.register_forward_pre_hook(cpu_visual_inputs, with_kwargs=True)
        self.model.visual.register_forward_hook(lambda module, args, output: output.to(device="cuda", dtype=torch.float16))
        if any(parameter.device.type != "cpu" for parameter in self.model.visual.parameters()):
            raise RuntimeError("Visual tower did not move to genuine CPU execution")
        for name, module in self.model.named_modules():
            if hasattr(module, "qweight") and module.qweight.device.type != "cuda":
                raise RuntimeError(f"Quantized layer was incorrectly offloaded: {name}")
        self.receipt = dict(model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", sha256=digest,
            revision="e7b623934290c5a4da0ee3c6e1e57bfb6b5abbf2", quantization="official AWQ checkpoint",
            adaptation="dense visual tower CPU float32; quantized language layers CUDA float16; no backbone substitution",
            processor_min_pixels=256*28*28, processor_max_pixels=256*28*28, loading_info=loading_info)

    @torch.inference_mode()
    def ground(self, instruction, frame_id, image):
        image = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image
        schema = dict(kind="inspect", bbox_xyxy=[0.1, 0.1, 0.5, 0.5], unresolved=False)
        text = (f"You are grounding an instruction for a UAV from observed evidence only. The supplied image is frame {frame_id}. "
                f"Instruction: {instruction}\nReturn one JSON object with exactly these keys: {list(schema)}. "
                "kind must be navigate, inspect or reobserve. "
                "bbox_xyxy is the visible referenced object's [left, top, right, bottom], each normalized to [0,1]. "
                "Do not invent hidden geometry or coordinates. If the reference cannot be grounded, set unresolved=true and bbox_xyxy=null. "
                "Return JSON only; no explanation, markdown or extra keys. "
                f"Formatting example (not an answer): {json.dumps(schema)}")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        result = self.model.generate(**inputs, max_new_tokens=192, do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
                                     return_dict_in_generate=True, output_hidden_states=True)
        response = self.processor.batch_decode(result.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response).strip()
        parsed = json.loads(cleaned)
        print(json.dumps({"grounding_frame": frame_id, "instruction": instruction, "raw_response": response}), flush=True)
        if set(parsed) != set(schema):
            raise ValueError("Grounder returned keys outside its interpretation schema")
        # Instruction identity and evidence provenance are transport-owned facts,
        # not language-model predictions. Exactly this image was supplied.
        parsed.update(instruction=instruction, evidence_frames=[frame_id])
        mission = GroundedMission.parse(parsed, available_frames=[frame_id])
        embedding = result.hidden_states[0][-1][0, -1].float().cpu()
        if embedding.shape != (2048,) or not torch.isfinite(embedding).all():
            raise RuntimeError("Unexpected Qwen contextual instruction-embedding shape")
        return dict(mission=parsed, embedding=embedding, prompt=prompt, response=response,
                    input_token_count=int(inputs.input_ids.shape[1]), provenance=self.receipt)


def grounding_gate(root):
    grounder = QwenGrounder(root)
    output = Path(root) / "integration/grounding"
    output.mkdir(parents=True, exist_ok=True)
    instruction = "Inspect the prominent building closest to the center of the image."
    result = grounder.ground(instruction, 7, Path(root) / "integration/flight/0007.png")
    torch.save(result["embedding"], output / "mission_embedding.pt")
    (output / "grounding.json").write_text(json.dumps({k: v for k, v in result.items() if k != "embedding"}, indent=2))
    from .pipeline import receipt
    receipt(Path(root), "grounding", {"passed": True, **grounder.receipt, "mission": result["mission"],
                                      "embedding_shape": list(result["embedding"].shape), "observed_frames": [7]})
