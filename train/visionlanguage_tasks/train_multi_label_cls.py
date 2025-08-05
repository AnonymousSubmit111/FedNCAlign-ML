import numpy as np
import torch
from torch import nn
import argparse
import sys
from tqdm import tqdm
from typing import Dict
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0, ".")

from data.image_datasets.openi_dataset import OpenI_ImagesDataset
from data.image_datasets.mimic_dataset import Mimic_ImagesDataset
from data.image_datasets.mnist_dataset import build_mnist_dataloader
from data.image_datasets.cifar10_dataset import build_cifar10_dataloader
from data.image_datasets.med_mnist_dataset import build_medmnist_dataloader
from data.image_datasets.voc_dataset import build_voc2012_dataloader
from data.image_datasets.isic_dataset import build_isic2018_dataloader
from data.image_datasets.xray_dataset import build_xray14_dataloader
from data.visionlanguage_datasets.vqa_dataset import build_vqa_vilt_dataloader

from train.visionlanguage_tasks.task_trainer import TaskTrainer
from utils.seed_utils import set_seed

from loss.angular_penalty_w_reject_loss import AngularPenaltywRejectionSMLoss
from loss.binary_ce_w_reject_loss import BinaryCE_wRejectionSMLoss
from loss.binary_ce_w_feature_contrastive_loss import BinaryCE_wContrastiveLoss
from loss.binary_ce_w_reject_n_feature_contrastive_loss import BinaryCE_wRejectContrastiveLoss


class MultiLabelTrainer(TaskTrainer):
    def __init__(self, logger, args: argparse.Namespace, task_configs: Dict, model_config: Dict, device: torch.device, task_key: str, task_output_dir=None, client_id=-1, accelerator=None):
        """
        Initializes a Trainer that handles training of a model on the VQA task

        args: Arguments provided by user
        task_configs: dictionary containing task-specific configuration parameters for all tasks
        model_config: dictionary containing model-specific configuration parameters
        device: cuda/cpu
        """
        super().__init__()
        self.accelerator = accelerator
        self.device = self.accelerator.device
        set_seed(args.seed + self.accelerator.process_index)   # make sure different process gets different seed
        self.logger = logger

        if args.do_wandb_logging:  # Create W&B experiment
            if self.accelerator.is_main_process:
                self.accelerator.init_trackers(project_name="missing_modality")
                self.accelerator.trackers[0].run.name = (task_output_dir.split("/")[-4] + "/" + task_output_dir.split("/")[-3] + "/" + task_output_dir.split("/")[-1])

        self.args = args
        self.local_epochs = args.local_epochs
        self.task_key = task_key
        self.task_output_dir = task_output_dir
        self.task_config = task_configs[args.task_config_key]
        self.batch2inputs_converter = model_config["batch2inputs_converter"]
        self.classifier_type = args.classifier_type
        self.obtain_class_wise_feature = args.obtain_class_wise_feature
        self.projection_type = args.projection_type
        
        self.scale_target = args.scale_target
        self.peak_uniformity_regularizer = args.peak_uniformity_regularizer
        self.CenterLoss_regularization = args.CenterLoss_regularization
        self.HNM_regularization = args.HNM_regularization

        buff = task_key.split('_')
        self.client_key = '{0}_client_{1}'.format(buff[0], buff[-1])

        # ------------ Create dataloaders for training, validation and test ------------
        # # ------------------------------------------------------------------------------
        self.model_name = args.encoder_name
        self.visual_input_type = model_config["visual_input_type"]  # pil_image
        
        if 'openi' in task_key or 'mimic' in task_key:
            if 'openi' in task_key:
                try:
                    self.images_dataset = OpenI_ImagesDataset(coco_dir=args.data_dir, data_dir=None, visual_input_type=args.visual_input_type, 
                                                                task_key=self.task_key, img_augmentation=args.image_augmentation)
                except Exception as e:
                    print(f"Error instantiating OpenI_ImagesDataset: {e}")
            elif 'mimic' in task_key:
                try:
                    self.images_dataset = Mimic_ImagesDataset(coco_dir=args.data_dir, data_dir=None, visual_input_type=args.visual_input_type, 
                                                                task_key=self.task_key, img_augmentation=args.image_augmentation)
                except Exception as e:
                    print(f"Error instantiating Mimic_ImagesDataset: {e}")

            self.vqa_train_dataloader, train_dataset = build_vqa_vilt_dataloader(
                logger=self.logger, args=args, images_dataset=self.images_dataset, split=self.args.splits[0], task_key=self.task_key, 
                visual_input_type=self.visual_input_type, client_id=client_id)
            self.vqa_val_dataloader, val_dataset = build_vqa_vilt_dataloader(
                logger=self.logger, args=args, images_dataset=self.images_dataset, split=self.args.splits[1], task_key=self.task_key,
                visual_input_type=self.visual_input_type, client_id=client_id)
            self.vqa_test_dataloader, test_dataset = build_vqa_vilt_dataloader(
                logger=self.logger, args=args, images_dataset=self.images_dataset, split=self.args.splits[2], task_key=self.task_key,
                visual_input_type=self.visual_input_type, client_id=client_id)
        elif 'medmnist' in task_key:
            self.vqa_train_dataloader, train_dataset = build_medmnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], task_key=self.task_key, client_id=client_id)
            self.vqa_val_dataloader, val_dataset = build_medmnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], task_key=self.task_key, client_id=client_id)
            self.vqa_test_dataloader, test_dataset = build_medmnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], task_key=self.task_key, client_id=client_id)
        elif 'mnist' in task_key:
            self.vqa_train_dataloader, train_dataset = build_mnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], task_key=self.task_key, client_id=client_id)
            self.vqa_val_dataloader, val_dataset = build_mnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], task_key=self.task_key, client_id=client_id)
            self.vqa_test_dataloader, test_dataset = build_mnist_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], task_key=self.task_key, client_id=client_id)
        elif 'cifar10' in task_key:
            self.vqa_train_dataloader, train_dataset = build_cifar10_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.vqa_val_dataloader, val_dataset = build_cifar10_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.vqa_test_dataloader, test_dataset = build_cifar10_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], label_dir=args.json_text_folder,
                task_key=self.task_key, client_id=client_id)
        elif 'voc2012' in task_key:
            self.vqa_train_dataloader, train_dataset = build_voc2012_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.vqa_val_dataloader, val_dataset = build_voc2012_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.vqa_test_dataloader, test_dataset = build_voc2012_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], label_dir=args.json_text_folder,
                task_key=self.task_key, client_id=client_id)
        elif 'isic2018' in task_key:
            self.vqa_train_dataloader, train_dataset = build_isic2018_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.vqa_val_dataloader, val_dataset = build_isic2018_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.vqa_test_dataloader, test_dataset = build_isic2018_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], label_dir=args.json_text_folder,
                task_key=self.task_key, client_id=client_id)
        elif 'xray14' in task_key:
            self.vqa_train_dataloader, train_dataset = build_xray14_dataloader(
                logger=self.logger, args=args, split=self.args.splits[0], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.vqa_val_dataloader, val_dataset = build_xray14_dataloader(
                logger=self.logger, args=args, split=self.args.splits[1], label_dir=args.json_text_folder, 
                task_key=self.task_key, client_id=client_id)
            self.vqa_test_dataloader, test_dataset = build_xray14_dataloader(
                logger=self.logger, args=args, split=self.args.splits[2], label_dir=args.json_text_folder,
                task_key=self.task_key, client_id=client_id)
        else:
            raise ValueError("train_multi_label_cls | Invalid dataset!")
        
        self.num_of_training_data = len(train_dataset)
        self.num_of_validation_data = len(val_dataset)
        self.num_of_testing_data = len(test_dataset)
        self.label_distribution_training = train_dataset.get_label_distribution()
        self.label_distribution_validation = val_dataset.get_label_distribution()
        self.label_distribution_testing = test_dataset.get_label_distribution()

        logger.info("Dataset | {}: len={}, {}: len={}, {}: len={}".format(
            self.args.splits[0], self.num_of_training_data, self.args.splits[1], self.num_of_validation_data, self.args.splits[2], self.num_of_testing_data))
        # ------------------------------------------------------------------------------

        # -------------------------- Training hyperparameters --------------------------
        # ------------------------------------------------------------------------------
        self.lr = self.args.lr
        self.adam_epsilon = self.task_config["adam_epsilon"]
        self.weight_decay = self.task_config["weight_decay"]
        self.loss_type = args.loss_type

        if args.loss_type == "binary_ce":
            self.loss_criterion = nn.BCEWithLogitsLoss(reduction='none')  # binary cross-entropy with sigmoid built in 
        elif args.loss_type == "binary_ce_w_prob":
            self.loss_criterion = torch.nn.BCELoss(reduction='none')
        elif args.loss_type == "dual_binary_ce":
            self.loss_criterion = Dual_BinaryCE(reduction='none', input_prob=False)
        elif args.loss_type == "binary_ce_w_dual_reg_rejection_n_contrastive_topK_PSC":
            self.loss_criterion = BinaryCE_wRejectContrastiveLoss(reduction='none', rejection_type="topK", contrastive_type="PSC", 
                                                                  rejection_margin=self.args.rejection_loss_threshold, 
                                                                  hyparam_rejection=self.args.hyparam_rejection_loss, 
                                                                  hyparam_contractive=self.args.hyparam_contractive_loss,
                                                                  use_focal_bce=args.focal_bce_flag)
        elif (args.loss_type == "cross_entropy") and (not self.obtain_class_wise_feature):
            self.loss_criterion = nn.CrossEntropyLoss(reduction='none')  # cross-entropy with softmax built in
        elif args.loss_type == "focal_ce":
            class_wise_alpha = calculate_dynamic_alpha_from_dataloader_consider_zero_sample_condition(self.vqa_train_dataloader).to(self.device)
            self.loss_criterion = FocalLoss_CE(alpha=class_wise_alpha, reduction='none')
        elif args.loss_type == "focal_ce_no_alpha":
            class_wise_alpha = 1
            self.loss_criterion = FocalLoss_CE(alpha=class_wise_alpha, reduction='none')
        elif args.loss_type == "focal_ce_no_alpha_v2":
            class_wise_alpha = 1
            self.loss_criterion = FocalLoss_CE_v2(alpha=class_wise_alpha, reduction='none')
        elif args.loss_type == "asymmetric_focal_binary_ce":
            self.loss_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        elif args.loss_type == "asymmetric_optimized_focal_binary_ce":
            self.loss_criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        elif args.loss_type == "cosface":
            self.loss_criterion = AngularPenaltySMLoss(loss_type='cosface', s=30.0, m=0.4)
        elif (args.loss_type == "cross_entropy") and self.obtain_class_wise_feature:
            self.loss_criterion = AngularPenaltySMLoss(loss_type='cross_entropy')
        elif (args.loss_type == "cross_entropy_w_rejection") and self.obtain_class_wise_feature:
            self.loss_criterion = AngularPenaltywRejectionSMLoss(loss_type='cross_entropy')
        else:
            raise ValueError("train_multi_label_cls | Invalid loss type!")
        print("train_multi_label_cls | MultiLabelTrainer | loss_type: {0}".format(args.loss_type))

        if self.classifier_type == "Dual_Classifier":
            if self.obtain_class_wise_feature:
                self.etf_reg_loss = nn.BCEWithLogitsLoss(reduction='none')  # binary cross-entropy with sigmoid built in 
            else:
                self.etf_reg_loss = nn.CrossEntropyLoss(reduction='none')  # cross-entropy with softmax built in

        if self.CenterLoss_regularization:
            feat_dim = model_config["encoder_dim"]
            num_classes = self.task_config["num_labels"]
            self.center_loss_reg = MultiLabelCenterLoss(num_classes, feat_dim)

        self.num_epochs = self.args.num_epochs
        self.max_steps = len(self.vqa_train_dataloader) * self.num_epochs
        self.warmup_ratio = 0.1  # TODO remove hard code
        # ------------------------------------------------------------------------------

    def train_step(self, model, step, batch, optimizer=None, scheduler=None, hooks=None, epoch=None):
        loss_dict = {}

        if isinstance(batch, dict) and "target_scores" in batch.keys():
            target = batch["target_scores"].to(self.device)
        
            if self.obtain_class_wise_feature:
                if self.classifier_type == "Dual_Classifier":
                    total_cls_feature, logits, total_cls_logits, etf_logits, total_cls_etf_logits = self.forward_pass(model, batch, do_eval=False)
                else:
                    total_cls_feature, logits, total_cls_logits = self.forward_pass(model, batch, do_eval=False)
            else:
                if self.classifier_type == "Dual_Classifier":
                    features, logits, etf_logits = self.forward_pass(model, batch, do_eval=False)
                else:
                    # features: encoder feature; logits: sigmoid predtion probability
                    features, logits = self.forward_pass(model, batch, do_eval=False)

            label_counts = target.sum(dim=1)
            if self.scale_target:
                label_counts_1 = label_counts.unsqueeze(1)  # shape: [N, 1]
                label_counts_1 = label_counts_1.expand_as(target)  # shape: (N, C)
                target = target / label_counts_1
            
            if self.obtain_class_wise_feature:
                if "binary_ce_w_dual_reg_rejection_n_contrastive" in self.loss_type:
                    prototypes = model.module.clf_layer.get_etf_matrix()
                    loss_ce = self.loss_criterion(logits, total_cls_logits, total_cls_feature, target, prototypes)
                elif "binary_ce_w_feature_contrastive" in self.loss_type:
                    prototypes = model.module.clf_layer.get_etf_matrix()
                    loss_ce = self.loss_criterion(logits, total_cls_logits, total_cls_feature, target, prototypes)
                elif ("binary_ce_w_rejection" in self.loss_type) or self.loss_type == "dual_binary_ce":
                    loss_ce = self.loss_criterion(logits, total_cls_logits, target)
                elif "binary_ce" in self.loss_type or "mse" in self.loss_type:
                    loss_ce = self.loss_criterion(logits, target)
                else: # cross_entropy or angular penalty
                    loss_ce = self.loss_criterion(total_cls_logits, target)
            else:
                loss_ce = self.loss_criterion(logits, target)

            loss_ce = loss_ce.mean(dim=0).sum()  # mean over batch, sum over classes
            
            # -------------------------- regularization --------------------------
            # --------------------------------------------------------------------
            if self.peak_uniformity_regularizer:
                prob_output = F.softmax(logits, dim=1)
                # peak_regularization = peak_uniformity_regularizer(prob_output, label_counts)
                peak_reg = peak_uniformity_regularizer_v2(prob_output, target)
            else:
                peak_reg = 0
            peak_reg_hparam = 500

            if self.classifier_type == "Dual_Classifier":
                etf_reg = self.etf_reg_loss(etf_logits, target)
                if self.obtain_class_wise_feature:
                    etf_reg = etf_reg.mean(dim=0).sum()  # mean over batch, sum over classes
                else:
                    etf_reg = etf_reg.mean(dim=0)  # mean over batch
            else:
                etf_reg = 0
            etf_reg_hparam = 1

            if self.CenterLoss_regularization:
                center_loss = self.center_loss_reg(features, target)
            else:
                center_loss = 0
            center_loss_hparam = 0.01

            if self.HNM_regularization:
                hnm_loss = hard_negative_loss(features, target, margin=0.5)
            else:
                hnm_loss = 0
            hnm_loss_hparam = 0.1

            loss = loss_ce + (peak_reg_hparam * peak_reg)  + (etf_reg_hparam * etf_reg) + (center_loss_hparam * center_loss) + (hnm_loss_hparam * hnm_loss)
            self.accelerator.backward(loss)

            if optimizer is not None:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            loss_dict["loss_ce"] = loss_ce
            loss_dict["etf_reg"] = etf_reg
            loss_dict["center_loss_reg"] = center_loss
            loss_dict["hnm_loss_reg"] = hnm_loss
            loss_dict["loss_total"] = loss
            return loss_dict

    def compute_f1_score_with_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        probabilities = torch.sigmoid(logits)  # Apply sigmoid to logits to get probabilities
        preds = (probabilities > 0.5).float()  # Binarize the probabilities with a threshold of 0.5
        preds_np = preds.cpu().numpy()  # Convert tensors to numpy arrays for scikit-learn's f1_score function
        labels_np = labels.cpu().numpy()
        
        # zero_division: Sets the value to return when there is a zero division
        f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)  # Calculate F1 score
        try:
            auc = roc_auc_score(labels_np, preds_np, average='macro', multi_class='ovr')
        except ValueError:
            auc = 0.0  # If only one class is present, set AUC to a default value (e.g., 0.0)
        return f1, auc

    def get_prediction_probability(self, logits: torch.Tensor, labels: torch.Tensor, return_raw_pred=False) -> float:
        if self.args.get_pred_probability_function == 'sigmoid':
            probabilities = torch.sigmoid(logits)  # Apply sigmoid to logits to get probabilities
        elif self.args.get_pred_probability_function == 'softmax':
            probabilities = F.softmax(logits, dim=1)  # Apply softmax to logits to get probabilities
        elif self.args.get_pred_probability_function == 'non':
            probabilities = logits
            max_val_out = torch.max(probabilities)  # Find max and min
            min_val_out = torch.min(probabilities)
            # print("probabilities | max_val_out: {0}, min_val_out: {1}".format(max_val_out, min_val_out))
        else:
            raise ValueError("train_multi_label_cls MultiLabelTrainer | Invalid get_pred_probability_function: {0}!".format(self.args.get_pred_probability_function))

        labels_np = labels.cpu().numpy()
        soft_preds_np = probabilities.cpu().numpy()
        num_classes = labels_np.shape[1]  # Number of classes
        label_list = []
        soft_pred_list = []
        for i in range(num_classes):  # Prediction and label list for each class 
            soft_pred_list.append(soft_preds_np[:, i])
            label_list.append(labels_np[:, i])
        if not return_raw_pred:
            return label_list, soft_pred_list
        else:
            return label_list, soft_pred_list, probabilities

    def get_train_dataloader(self):
        return self.vqa_train_dataloader

    def get_collate_fn(self):
        return self.vqa_train_dataloader.collate_fn

    def add_alpha(self, epoch, batch, step):
        if epoch > 0:  # alpha is for distill
            alpha = 0.4
        else:
            alpha = 0.4 * min(1, step / len(self.vqa_train_dataloader))
        batch.append(alpha)
        return batch
    
    def get_num_of_training_data(self):
        return self.num_of_training_data

    def get_num_of_validation_data(self):
        return self.num_of_validation_data
    
    def get_num_of_testing_data(self):
        return self.num_of_testing_data
    
    def get_training_data_label_distribution(self):
        return self.label_distribution_training
    
    def get_validation_data_label_distribution(self):
        return self.label_distribution_validation
    
    def get_testing_data_label_distribution(self):
        return self.label_distribution_testing
    

    def compute_fisher_information_matrix(self, model) -> Dict[str, torch.Tensor]:
        """
        Computes the diagonal Fisher Information Matrix (FIM) for a given model.
        Returns a dictionary mapping parameter names to their estimated FIM values.
        """
        model = self.accelerator.prepare(model)
        model.eval()

        loader = self.vqa_train_dataloader
        fisher_info = {}
        total_fim = 0.0
        total_count = 0
        all_fim_values = []
        for name, param in model.named_parameters():
            if torch.is_floating_point(param):
                fisher_info[name] = torch.zeros_like(param, dtype=torch.float32)
        num_samples = 0

        for step, batch in enumerate(tqdm(loader, desc="Computing Fisher Information Matrix")):
            model.zero_grad()

            with torch.set_grad_enabled(True):
                if self.obtain_class_wise_feature:
                    if self.classifier_type == "Dual_Classifier":
                        _, logits, total_cls_logits, _, _ = self.forward_pass(model, batch, do_eval=False)
                    else:
                        _, logits, total_cls_logits = self.forward_pass(model, batch, do_eval=False)
                else:
                    if self.classifier_type == "Dual_Classifier":
                        features, logits, _ = self.forward_pass(model, batch, do_eval=False)
                    else:
                        features, logits = self.forward_pass(model, batch, do_eval=False)

                target = batch["target_scores"].to(self.device)

                if self.scale_target:
                    label_counts = target.sum(dim=1).unsqueeze(1)
                    target = target / label_counts.expand_as(target)

                if self.obtain_class_wise_feature:
                    if self.loss_type == "binary_ce_w_rejection":
                        loss_ce = self.loss_criterion(logits, total_cls_logits, target)
                    elif "binary_ce" in self.loss_type or "mse" in self.loss_type:
                        loss_ce = self.loss_criterion(logits, target)
                    else:  # cross_entropy or angular penalty
                        loss_ce = self.loss_criterion(total_cls_logits, target)
                else:
                    loss_ce = self.loss_criterion(logits, target)

                loss = loss_ce.mean(dim=0).sum()  # Mean over batch, sum over classes
                loss = loss / len(loader)         # Normalize
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None and name in fisher_info:
                        fisher_info[name] += (param.grad.detach() ** 2)

                num_samples += 1

        for name in fisher_info:  # Average over all batches
            fisher_info[name] /= num_samples
            total_fim += fisher_info[name].sum().item()
            total_count += fisher_info[name].numel()
            all_fim_values.append(fisher_info[name].flatten())

        model_avg_fim = total_fim / total_count if total_count > 0 else 0.0

        # Normalize using min-max across all values
        all_fim_flat = torch.cat(all_fim_values)
        min_val = all_fim_flat.min()
        max_val = all_fim_flat.max()
        range_val = max_val - min_val if max_val > min_val else 1.0  # Avoid divide-by-zero

        for name in fisher_info:
            fisher_info[name] = (fisher_info[name] - min_val) / range_val

        unwrapped_model = self.accelerator.unwrap_model(model)  # Clean up
        torch.cuda.empty_cache()
        self.accelerator.free_memory()

        return fisher_info, model_avg_fim


    def compute_wanda_information_matrix(self, model) -> (Dict[str, torch.Tensor], float):
        """
        Computes Wanda-based importance scores using w * |x| where x is the input activation.
        Returns:
            - wanda_scores: dict mapping parameter names to importance tensors
            - model_avg_wanda: scalar float representing average importance
        """
        model = self.accelerator.prepare(model)
        model.eval()

        loader = self.vqa_train_dataloader
        wanda_scores = {}
        total_wanda = 0.0
        total_count = 0
        all_scores_flattened = []

        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1:  # Only weight matrices
                wanda_scores[name] = torch.zeros_like(param, dtype=torch.float32)

        num_batches = 0

        for step, batch in enumerate(tqdm(loader, desc="Computing Wanda Importance")):
            model.zero_grad()

            if hasattr(model.module, "forward_with_activations"):
                inputs = self.batch2inputs_converter(batch)
                output, activations = model.module.forward_with_activations(client_key=self.client_key, **inputs)
            else:
                raise RuntimeError("Model must implement forward_with_activations(batch) that returns activations dict")
    
            for name, param in model.named_parameters():
                if name in activations and name in wanda_scores:
                    try:
                        x = activations[name]  # [B, in_features]
                        w = param.data         # [out_features, in_features]

                        x_mean = x.mean(dim=0, keepdim=True)
                        x_norm = nn.functional.normalize(x_mean, p=2, dim=1)
                        score = torch.abs(param.data) * x_norm.to(param.device)

                        wanda_scores[name] += score

                        del x, x_mean, x_norm, w, score
                    except Exception as e:
                        print(f"[Warning] Skipping {name}: {e}")

            del activations
            torch.cuda.empty_cache()
            num_batches += 1

        # Average Wanda scores across batches and compute stats
        for name in wanda_scores:
            wanda_scores[name] /= num_batches
            total_wanda += wanda_scores[name].sum().item()
            total_count += wanda_scores[name].numel()
            all_scores_flattened.append(wanda_scores[name].flatten())

        model_avg_wanda = total_wanda / total_count if total_count > 0 else 0.0
        print("model_avg_wanda: {0}".format(model_avg_wanda))

        # Normalize each tensor using global min-max
        all_scores_concat = torch.cat(all_scores_flattened)
        min_val, max_val = all_scores_concat.min(), all_scores_concat.max()
        range_val = max_val - min_val if max_val > min_val else 1.0

        for name in wanda_scores:
            wanda_scores[name] = (wanda_scores[name] - min_val) / range_val

        unwrapped_model = self.accelerator.unwrap_model(model)
        torch.cuda.empty_cache()
        self.accelerator.free_memory()

        return wanda_scores, model_avg_wanda
