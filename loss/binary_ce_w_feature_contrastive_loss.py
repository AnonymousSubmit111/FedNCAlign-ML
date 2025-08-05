import torch
from torch import nn
import torch.nn.functional as F

from loss.focal_binary_ce_loss import FocalLoss_BinaryCE


def compute_psc_loss(selected_features, class_indices, prototypes, tau=0.07):
    """
    Compute PSC loss with correct denominator (includes all classes).
    Args:
        selected_features: [N, D]
        class_indices: [N]
        prototypes: [C, D]
    Returns:
        psc_loss: [N] per-sample PSC loss
    """
    # Normalize
    selected_features = F.normalize(selected_features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    # Similarity matrix [N, C]
    sim_matrix = torch.matmul(selected_features, prototypes.T)

    # Compute logits (softmax inputs)
    logits = sim_matrix / tau  # [N, C]

    # Use cross-entropy style loss: softmax over all classes
    psc_loss = F.cross_entropy(logits, class_indices, reduction='none')  # [N]
    return psc_loss


class BinaryCE_wContrastiveLoss(nn.Module):
    def __init__(self, reduction='none', hyparam_contractive=1, contrastive_type="PSC", use_focal_bce=False):
        super(BinaryCE_wContrastiveLoss, self).__init__()
        self.hyparam_contractive = hyparam_contractive
        self.contrastive_type = contrastive_type

        if use_focal_bce:
            class_wise_alpha = 1
            self.bce_loss_criterion = FocalLoss_BinaryCE(alpha=class_wise_alpha, reduction='none')  # focal binary cross-entropy
        else:
            self.bce_loss_criterion = nn.BCEWithLogitsLoss(reduction=reduction)

        print("BinaryCE_wContrastiveLoss | use_focal_bce: {0}".format(use_focal_bce))
        print("BinaryCE_wContrastiveLoss | contrastive_type: {0}, hyparam_contractive: {1}".format(contrastive_type, hyparam_contractive))

    def forward(self, logits, total_cls_logits, total_cls_feature, labels, prototypes):
        # --------------------- BCE loss ---------------------
        # ----------------------------------------------------
        bce_loss = self.bce_loss_criterion(logits, labels)
        bce_loss_per_sample = bce_loss.sum(dim=1)

        # ------------- Feature Contractive Regularization -------------
        # --------------------------------------------------------------
        # --- Step 1: Extract relevant logits and Features ---
        nonzero_indices = labels.nonzero(as_tuple=False)  # [N_active, 2]
        # If a sample has multiple 1s in its label, labels.nonzero() will return one row per active class, even if they come from the same sample.
        batch_indices = nonzero_indices[:, 0]  # [N_active]
        class_indices = nonzero_indices[:, 1]  # [N_active]
       
        # total_cls_logits: [C, B, logit_len] → permute to [B, C, logit_len] for indexing
        total_cls_logits_transposed = total_cls_logits.permute(1, 0, 2)  # [B, C, logit_len]
        # total_cls_feature: [C, B, feature_len] → permute to [B, C, feature_len] for indexing
        total_cls_features_transposed = total_cls_feature.permute(1, 0, 2)  # [B, C, feature_len]
        
        # Selected logits for each (sample, class) pair
        selected_logits = total_cls_logits_transposed[batch_indices, class_indices]  # shape: [N_active, logit_len]
        selected_features = total_cls_features_transposed[batch_indices, class_indices]  # shape: [N_active, feature_len]

        # --- Step 2: Extract leftover logits and Features ---
        B, C, _ = total_cls_logits_transposed.shape  # [B, C, logit_len]
        total_pairs = torch.cartesian_prod(torch.arange(B, device=total_cls_logits.device), torch.arange(C, device=total_cls_logits.device))  # [B*C, 2]
        selected_pairs = torch.stack([batch_indices, class_indices], dim=1)  # [N_active, 2]

        # Use set difference to get leftover indices
        selected_set = set(map(tuple, selected_pairs.tolist()))
        leftover_pairs = [pair for pair in total_pairs.tolist() if tuple(pair) not in selected_set]
        leftover_pairs = torch.tensor(leftover_pairs, dtype=torch.long, device=total_cls_logits.device)  # [N_left, 2]
        left_batch_indices = leftover_pairs[:, 0]
        left_class_indices = leftover_pairs[:, 1]
        
        leftover_logits = total_cls_logits_transposed[left_batch_indices, left_class_indices]  # [N_left, logit_len]
        leftover_features = total_cls_features_transposed[left_batch_indices, left_class_indices]  # [N_left, logit_len]

        # --- Step 3: Compute contractive loss (pull features closer to its prototype and away from other prototypes) ---
        if self.contrastive_type == "L2":
            # prototypes: [C, feature_dim] — one vector per class

            # Gather the relevant prototypes for each selected feature's class
            selected_prototypes = prototypes[class_indices]  # [N_active, feature_dim]

            # Compute L2 distance between features and their corresponding class prototype
            contractive_distances = F.mse_loss(selected_features, selected_prototypes, reduction='none')  # [N_active, feature_dim]
            contractive_loss_per_pair = contractive_distances.sum(dim=1)  # [N_active]

            # Aggregate loss per sample (if multiple active labels per sample)
            contractive_loss_per_sample = torch.zeros(logits.size(0), device=logits.device)  # [B]
            contractive_loss_per_sample = contractive_loss_per_sample.index_add(0, batch_indices, contractive_loss_per_pair)

        elif self.contrastive_type == "Cosine":
            # prototypes: [C, feature_dim] — one vector per class

            # Gather the relevant prototypes for each selected feature's class
            selected_prototypes = prototypes[class_indices]  # [N_active, feature_dim]

            # Compute cosine similarity between features and their corresponding class prototype
            cosine_sim = F.cosine_similarity(selected_features, selected_prototypes, dim=1)  # [N_active]
            contractive_loss_per_pair = 1 - cosine_sim  # higher when features and prototype are dissimilar
            
            # Aggregate loss per sample (if multiple active labels per sample)
            contractive_loss_per_sample = torch.zeros(logits.size(0), device=logits.device)  # [B]
            contractive_loss_per_sample = contractive_loss_per_sample.index_add(0, batch_indices, contractive_loss_per_pair)

        elif self.contrastive_type == "PSC":
            # Prototype Similarity Contrastive (PSC) Loss
            psc_loss = compute_psc_loss(selected_features=selected_features, class_indices=class_indices, prototypes=prototypes)

            # Aggregate loss per sample (if multiple active labels per sample)
            contractive_loss_per_sample = torch.zeros(logits.size(0), device=logits.device)  # [B]
            contractive_loss_per_sample = contractive_loss_per_sample.index_add(0, batch_indices, psc_loss)

        # ------------------ Combine losses ------------------ 
        # ----------------------------------------------------
        total_loss = bce_loss_per_sample + self.hyparam_contractive * contractive_loss_per_sample
        """
        print("BinaryCE_wRejectionSMLoss | total_loss: {0} | bce_loss_per_sample: {1}, contractive_loss_per_sample: {2}".format(
            total_loss.mean(), bce_loss_per_sample.mean(), contractive_loss_per_sample.mean()))
        """
        return total_loss

        