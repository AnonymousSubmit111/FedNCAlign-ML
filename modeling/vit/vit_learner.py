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


class ViTLearner(ContinualLearner):
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
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.device = device
        self.client_list = client_list
        self.task_configs = task_configs
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
        self.num_class = num_labels
        num_images = task_config["num_images"]
        print(f"resnet_learner | num_labels: {num_labels}, num_images: {num_images}")

        if task_config["model_type"] != "classification":
            raise ValueError("vit | Only classification is supported for image-only model")

        # -------------------------------- Classifier --------------------------------
        # ----------------------------------------------------------------------------
        if self.classifier_type == "FC_Classifier":
            self.clf_layer = FCClassifierLayer(self.encoder_dim * num_images, num_images, num_labels)
        elif self.classifier_type == "Cosine_Classifier":
            self.clf_layer = CosineClassifierLayer(self.encoder_dim * num_images, num_images, num_labels)
        elif self.classifier_type == "MultiLabel_ETF_Classifier":
            self.clf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device)
        elif self.classifier_type == "MultiLabel_ETF_Classifier_w_feature_normalized":
            self.clf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device, feature_normalized=True)
        elif self.classifier_type == "Dual_Classifier":
            self.clf_layer = FCClassifierLayer(self.encoder_dim * num_images, num_images, num_labels)
            self.etf_layer = ETFClassifier(num_classes=num_labels, feature_dim=self.encoder_dim * num_images, device=self.device)
            self.etf_layer.eval()
        else:
            raise ValueError("vit | Invalid classifier type!")
        
        # ---------------------------- Class-specific MLP ----------------------------
        # ----------------------------------------------------------------------------
        if self.class_wise_MLP:
            self.class_wise_MLP_layer = PerClassMLPs(num_classes=num_labels, input_dim=self.encoder_dim * num_images, hidden_dim=self.encoder_dim * 2, output_dim=self.encoder_dim * num_images)


    def forward(self, client_key: str, images: List):
        task_config_key = "openi_train" if "openi" in client_key else "mimic_train"
        task_config = self.task_configs[task_config_key]

        assert task_config["model_type"] == "classification"

        if task_config["num_images"] == 1:
            return self.forward_single_image(images)
        else:
            return self.forward_multi_images(images, task_config["num_images"])

    def forward_single_image(self, images: List):
        x = self.encoder.process_inputs(images).to(self.device)
        features = self.encoder(x)  # [B, D]

        if self.class_wise_MLP:
            total_cls_logits = []
            total_cls_feature = []
            logits = torch.zeros(features.size(0), self.num_class, device=features.device)
            for cls_index in range(self.num_class):
                cls_features = self.class_wise_MLP_layer(features, cls_index)
                cls_logits = self.clf_layer(cls_features)
                # Zero out all logits except the one for cls_index
                mask = torch.zeros_like(cls_logits)
                mask[:, cls_index] = cls_logits[:, cls_index]
                # Accumulate into final_logits
                logits += mask
                total_cls_logits.append(cls_logits)
                total_cls_feature.append(cls_features)
            total_cls_logits_tensor = torch.stack(total_cls_logits)  
            total_cls_feature_tensor = torch.stack(total_cls_feature)
            return total_cls_feature_tensor, logits, total_cls_logits_tensor
        else:
            logits = self.clf_layer(features)
        if self.classifier_type == "Dual_Classifier":
            etf_logits = self.etf_layer(features)
            return features, logits, etf_logits
        else:
            return features, logits

    def forward_multi_images(self, images: List[List], num_images: int):
        batch_size = len(images)
        flat_images = [img for sublist in images for img in sublist]
        x = self.encoder.process_inputs(flat_images).to(self.device)
        x = x.view(batch_size * num_images, 3, x.size(-2), x.size(-1))

        features = self.encoder(x)  # [B*num_images, D]
        features = features.view(batch_size, num_images * self.encoder_dim)
        logits = self.clf_layer(features)
        if self.classifier_type == "Dual_Classifier":
            etf_logits = self.etf_layer(features)
            return features, logits, etf_logits
        else:
            return features, logits
