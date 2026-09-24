"""Strict, frozen DINOv2-B/14; no random-weight or smaller-backbone fallback."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .upstream import verify_revision, sha256


class DinoVision:
    def __init__(self, root, device="cuda"):
        root = Path(root)
        repo = root / "upstream/dinov2"
        revision = verify_revision(repo, "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
        checkpoint = root / "weights/dinov2_vitb14.pth"
        digest = sha256(checkpoint)
        if digest != "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73":
            raise ValueError("DINOv2 checkpoint hash mismatch")
        self.model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.device = device
        self.receipt = dict(backbone="dinov2_vitb14", source_revision=revision, sha256=digest,
                            loaded_parameter_fraction=1.0, parameters=sum(p.numel() for p in self.model.parameters()),
                            preprocessing="RGB -> direct 448x448 bicubic -> ImageNet normalization; UAV adaptation",
                            native_reference_preprocessing="separate 224x224 direct resize for interface check")

    @torch.inference_mode()
    def encode(self, image, side=448):
        if side not in (224, 448):
            raise ValueError("Only explicit reference/research resolutions are accepted")
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB")
        array = np.array(image.resize((side, side), Image.Resampling.BICUBIC), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(self.device)
        mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
        std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device.startswith("cuda"), dtype=torch.float16):
            features = self.model.forward_features((tensor - mean) / std)["x_norm_patchtokens"]
        expected = (1, (side // 14) ** 2, 768)
        if features.shape != expected or not torch.isfinite(features).all():
            raise RuntimeError(f"Invalid real backbone output: {features.shape}")
        return features[0].float().cpu()
