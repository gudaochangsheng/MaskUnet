import torch
from hpsv2.src.open_clip import create_model, get_tokenizer
import torch.nn as nn
import huggingface_hub

class HPSV2Loss(nn.Module):
    """HPS reward loss function for optimization."""

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        super(HPSV2Loss, self).__init__()  # 先调用父类的初始化方法
        self.hps_model = create_model(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision=dtype,
            device=device,
            cache_dir=cache_dir,
        )
        checkpoint_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", "HPS_v2.1_compressed.pt", cache_dir=cache_dir
        )
        self.hps_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)["state_dict"]
        )
        self.hps_tokenizer = get_tokenizer("ViT-H-14")
        if memsave:
            import memsave_torch.nn

            self.hps_model = memsave_torch.nn.convert_to_memory_saving(self.hps_model)
        self.hps_model = self.hps_model.to(device, dtype=dtype)
        self.hps_model.eval()
        self.freeze_parameters(self.hps_model.parameters())
        # super().__init__("HPS")
        self.hps_model.set_grad_checkpointing(True)

    @staticmethod
    def freeze_parameters(params: torch.nn.ParameterList):
        for param in params:
            param.requires_grad = False

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        hps_image_features = self.hps_model.encode_image(image)
        return hps_image_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        hps_text = self.hps_tokenizer(prompt).to("cuda")
        hps_text_features = self.hps_model.encode_text(hps_text)
        return hps_text_features

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        logits_per_image = image_features @ text_features.T
        hps_loss = 1 - torch.diagonal(logits_per_image)[0]
        return hps_loss

    def process_features(self, features: torch.Tensor) -> torch.Tensor:
        features_normed = features / features.norm(dim=-1, keepdim=True)
        return features_normed

    def score_grad(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        image_features = self.get_image_features(image)
        text_features = self.get_text_features(prompt)

        image_features_normed = self.process_features(image_features)
        text_features_normed = self.process_features(text_features)

        loss = self.compute_loss(image_features_normed, text_features_normed)
        return loss