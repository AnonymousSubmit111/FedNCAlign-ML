import torch
from typing import List, Dict
from timm import create_model
import torch.nn as nn
from PIL import Image

from modeling.vit.vit_encoder import ViTEncoderWrapper
from modeling.vit.vit_multi_head_learner import ViTMultiHeadLearner
from modeling.vit.vit_learner import ViTLearner


def load_vit_encoder(logger, checkpoint_name: str, device: torch.device, imagenet_pretrain: bool, backbone: str = 'vit_tiny_patch16_224'):
    """
    Loads a ViT encoder with optional checkpoint override.

    Args:
        logger: logging instance
        checkpoint_name: path to a checkpoint or "imagenet" for default pretrained weights
        device: torch.device
        backbone: ViT variant: vit_tiny_patch16_224 (resnet18), vit_small_patch16_224 (resnet34), vit_base_patch16_224 (resnet50, most commonly used ViT model), etc.

    Returns:
        encoder: ViTEncoderWrapper instance
    """
    logger.info("-" * 100)
    logger.info(f"vit | load_vit_encoder | Loading ViT encoder: {backbone}")

    # Load ViT backbone choose from: 'vit_tiny_patch16_224', 'vit_base_patch16_224'
    model = create_model(backbone, pretrained=(checkpoint_name == "imagenet"))

    # Remove classification head
    model.reset_classifier(0) 

    encoder = ViTEncoderWrapper(model, model.num_features, device, imagenet_pretrain)

    if checkpoint_name not in ["imagenet", ""]:
        logger.info(f"vit | load_vit_encoder | Loading weights from checkpoint: {checkpoint_name}")
        ckpt = torch.load(checkpoint_name, map_location=device)
        encoder.load_state_dict(ckpt)

    logger.info("vit | load_vit_encoder | Successfully loaded ViT encoder")
    return encoder


def create_vit_model(logger, model_name_or_path: str, client_specific_head: bool, client_list: List[str], model_config: Dict, task_configs: Dict, 
                        device: torch.device, dataset_path: str, remove_text: True, remove_image: False, class_wise_MLP: False, 
                        imagenet_pretrain: False, backbone_name="vit_tiny_patch16_224"):
    """
    Args:
        logger: logging instance
        model_name_or_path: checkpoint or 'imagenet'
        client_specific_head: whether to use a multi-head learner
        client_list: clients for continual learning
        model_config: configuration including 'backbone' and 'encoder_dim'
        task_configs: task-specific configs
        device: torch.device

    Returns:
        model: ResNetLearner instance
    """
    encoder = load_vit_encoder(
        logger=logger,
        checkpoint_name=model_name_or_path,
        device=device,
        backbone=model_config.get("backbone", backbone_name),
        imagenet_pretrain=imagenet_pretrain
    )

    if client_specific_head:
        model = ViTMultiHeadLearner(
            client_list=client_list,
            encoder=encoder,
            encoder_dim=model_config["encoder_dim"],
            task_configs=task_configs,
            device=device,
            adapter_config=model_config['adapter_config'] if 'adapter_config' in model_config else None,
            dataset_path=dataset_path,
            remove_text=remove_text,
            remove_image=remove_image,
            class_wise_MLP=class_wise_MLP
        )
    else:
        model = ViTLearner(
            client_list=client_list,
            encoder=encoder,
            encoder_dim=model_config["encoder_dim"],
            task_configs=task_configs,
            device=device,
            adapter_config=model_config['adapter_config'] if 'adapter_config' in model_config else None,
            dataset_path=dataset_path,
            remove_text=remove_text,
            remove_image=remove_image,
            class_wise_MLP=class_wise_MLP
        )

    logger.info("vit | Successfully created and initialized ViT Learner model")
    return model

def convert_batch_to_vit_input_dict(batch: Dict):
    """
    Convert inputs from batch_collate into format consumable by ResNet (image-only).
    """
    return {"images": batch["images"]}


def convert_seq_batch_to_vit_input_dict(batch: List, mean_image: Image):
    """
    Sequence input for tasks like sentence similarity.
    Only mean_image is used for all samples.
    """
    return {"images": [mean_image]}


def convert_mc_batch_to_vit_input_dict(batch: List, mean_image: Image):
    """
    Convert multi-choice batch to ResNet-compatible input (image-only).
    All text options are ignored — only image is used (repeated if needed).
    """
    bs = len(batch[0])  # batch[0] = texts_a, batch[1] = list of texts_b
    return {"images": [mean_image] * bs}


def create_vit_t_16_model(logger, model_name_or_path: str, client_specific_head: bool, client_list: List[str], model_config: Dict, task_configs: Dict, 
                        device: torch.device, dataset_path: str, remove_text: True, remove_image: False, imagenet_pretrain: False, class_wise_MLP: False):
    model = create_vit_model(logger, model_name_or_path, client_specific_head, client_list, model_config, task_configs, 
                                device, dataset_path, remove_text, remove_image, class_wise_MLP, imagenet_pretrain, backbone_name="vit_tiny_patch16_224")
    return model
    

def create_vit_b_16_model(logger, model_name_or_path: str, client_specific_head: bool, client_list: List[str], model_config: Dict, task_configs: Dict, 
                        device: torch.device, dataset_path: str, remove_text: True, remove_image: False, imagenet_pretrain: False, class_wise_MLP: False):
    model = create_vit_model(logger, model_name_or_path, client_specific_head, client_list, model_config, task_configs, 
                                device, dataset_path, remove_text, remove_image, class_wise_MLP, imagenet_pretrain, backbone_name="vit_base_patch16_224")
    return model