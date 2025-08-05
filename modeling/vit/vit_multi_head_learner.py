import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict
from typing_extensions import OrderedDict
from medmnist import INFO

from modeling.continual_learner import ContinualLearner
from modeling.vit.vit_encoder import ViTEncoderWrapper
from modeling.mlp.mlp import PerClassMLPs

from neural_collapse.cosine_classifier import FCClassifierLayer, CosineClassifierLayer
from neural_collapse.etf_classifier import ETFClassifier


class ViTMultiHeadLearner(ContinualLearner):
    def __init__(self,
                 client_list: List[str],
                 encoder: ViTEncoderWrapper,
                 encoder_dim: int,
                 task_configs: Dict,
                 device: torch.device,
                 adapter_config: str,
                 dataset_path: str,
                 remove_text=True,
                 remove_image=False, 
                 imagenet_pretrain=False,
                 class_wise_MLP=False
                 ):
        """
        Multi-head ViT-based learner for multiple clients/tasks.
        Each client gets its own head.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.client_list = client_list
        self.task_configs = task_configs
        self.device = device
        self.adapter_config = adapter_config
        self.dataset_path = dataset_path
        self.remove_text = remove_text
        self.remove_image = remove_image
        self.class_wise_MLP = class_wise_MLP

        buff = client_list[0].split("_")
        task_config_name = "{0}_train".format(buff[0])
        task_config = task_configs[task_config_name]
        self.task_config = task_config
        self.classifier_type = task_config['classifier_type']

        if "medmnist" not in task_config["task_name"]:
            num_labels = task_config["num_labels"]
        else:
            dataset_name = "{0}mnist".format(task_config["images_source"].split("_")[-1])
            num_labels = len(INFO[dataset_name]['label'])
        num_images = task_config["num_images"]
        print(f"resnet_learner | num_labels: {num_labels}, num_images: {num_images}")

        if task_config["model_type"] != "classification":
            raise ValueError("resnet | Only classification is supported for image-only model")

        self.task_layer_dict = {}
        for client_key in client_list:
            buff = client_key.split("_")
            task_config_name = f"{buff[0]}_train"
            self.classifier_type = task_configs[task_config_name]['classifier_type']
            self.add_task_layer(client_key, task_configs[task_config_name])
        self.task_layer = nn.ModuleDict(self.task_layer_dict)

        if self.classifier_type == "Dual_Classifier":
            self.etf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device)
            self.etf_layer.eval()

        if self.class_wise_MLP:
            self.class_wise_MLP_layer = PerClassMLPs(num_classes=num_labels, input_dim=self.encoder_dim * num_images, hidden_dim=self.encoder_dim * 2, output_dim=self.encoder_dim * num_images)

    def add_task_layer(self, client_key: str, task_config: Dict):
        """
        Create a task-specific head (classifier layer) for each client.
        """
        num_labels = task_config["num_labels"]
        num_images = task_config["num_images"]
        
        if (self.classifier_type == "FC_Classifier") or (self.classifier_type == "Dual_Classifier"):
            clf_layer = FCClassifierLayer(self.encoder_dim * num_images, num_images, num_labels)
        elif self.classifier_type == "Cosine_Classifier":
            clf_layer = CosineClassifierLayer(self.encoder_dim, num_images, num_labels)
        elif self.classifier_type == "MultiLabel_ETF_Classifier":
            clf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device)
        elif self.classifier_type == "MultiLabel_ETF_Classifier_w_feature_normalized":
            clf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device, feature_normalized=True)
        else:
            raise ValueError("vit | Unsupported classifier type.")

        self.task_layer_dict[client_key] = clf_layer

    def forward(self, client_key: str, images: List):
        """
        Forward pass depending on number of images per sample (single or multiple).
        """
        task_config_key = "openi_train" if "openi" in client_key else "mimic_train"
        task_config = self.task_configs[task_config_key]

        num_images = task_config["num_images"]

        if num_images == 1:
            return self.forward_single_image(client_key, images)
        else:
            return self.forward_multi_images(client_key, images, num_images)

    def forward_single_image(self, client_key: str, images: List):
        """
        Forward pass for single-image inputs.
        """
        inputs = self.encoder.process_inputs(images).to(self.device)
        features = self.encoder(inputs)  # shape [B, D]
        logits = self.task_layer[client_key](features)
        if self.classifier_type == "Dual_Classifier":
            with torch.no_grad():
                etf_logits = self.etf_layer(features)
            return features, logits, etf_logits
        else:
            return features, logits

    def forward_multi_images(self, client_key: str, images: List[List], num_images: int = 2):
        """
        Forward pass for multiple images per sample.
        """
        flat_images_list = list(itertools.chain(*images))
        inputs = self.encoder.process_inputs(flat_images_list).to(self.device)

        bs = len(images)
        features = self.encoder(inputs)  # shape [B * num_images, D]
        pooled_output = features.view(bs, -1)  # shape [B, num_images * D]

        logits = self.task_layer[client_key](pooled_output)
        return pooled_output, logits
