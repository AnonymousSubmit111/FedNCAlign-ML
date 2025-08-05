import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
from typing import List


class ViTEncoderWrapper(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 encoder_dim: int,
                 device: torch.device,
                 imagenet_pretrain: bool):
        """
        Wrapper for  ViT-based image encoders.
        Compatible with models from timm (e.g., vit_base_patch16_224).

        Args:
            backbone: Vision Transformer model (without classifier head).
            encoder_dim: Output feature dimension (e.g., 768 for ViT-Base).
            device: torch.device.
            imagenet_pretrain: Whether to use ImageNet normalization.
        """
        super().__init__()
        self.backbone = backbone
        self.encoder_dim = encoder_dim
        self.device = device
        self.to(device)

        if imagenet_pretrain:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Standard ViT input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet-pretrained models (e.g., ResNet, ViT, EfficientNet, etc., trained on ImageNet-1k), the official ImageNet means and stds 
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Training from scratch or small-scale datasets
            ])

    def process_inputs(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess a list of PIL images into a batch tensor.

        Args:
            images: List of PIL images.

        Returns:
            torch.Tensor of shape (B, 3, H, W).
        """

        tensor_batch = [self.image_transform(img) for img in images]
        return torch.stack(tensor_batch).to(self.device)

    def forward(self, images: torch.Tensor) -> torch.FloatTensor:
        """
        Args:
            images: torch.Tensor of shape (B, 3, H, W)

        Returns:
            torch.FloatTensor: shape (B, encoder_dim)
        """
        with torch.set_grad_enabled(self.training):
            features = self.backbone(images)  # (B, encoder_dim)
            return features

    def freeze_all_weights(self):
        """
        Freeze all parameters in the encoder.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_bottom_k_layers(self, k: int):
        """
        Freeze bottom K transformer blocks (ViT specific).

        Args:
            k: number of lowest blocks to freeze.
        """
        if hasattr(self.backbone, 'blocks'):
            for i in range(min(k, len(self.backbone.blocks))):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = False
        else:
            raise AttributeError("Backbone does not have `blocks` attribute (expected for ViT models)")
