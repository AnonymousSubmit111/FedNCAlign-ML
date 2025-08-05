import torch
from torch import nn
import torch.nn.functional as F

from loss.focal_binary_ce_loss import FocalLoss_BinaryCE


class BinaryCE_wRejectionSMLoss(nn.Module):
    def __init__(self, reduction='none', rejection_margin=0.3, hyparam_rejection=1, balanced_random=False, balanced_topK=False, use_focal_bce=False):
        super(BinaryCE_wRejectionSMLoss, self).__init__()
        self.rejection_margin = rejection_margin
        self.hyparam_rejection = hyparam_rejection
        self.balanced_random = balanced_random
        self.balanced_topK = balanced_topK

        if use_focal_bce:
            class_wise_alpha = 1
            self.bce_loss_criterion = FocalLoss_BinaryCE(alpha=class_wise_alpha, reduction='none')  # focal binary cross-entropy
        else:
            self.bce_loss_criterion = nn.BCEWithLogitsLoss(reduction=reduction)

        print("BinaryCE_wRejectionSMLoss | use_focal_bce: {0}".format(use_focal_bce))
        print("BinaryCE_wRejectionSMLoss | balanced_random: {0}, balanced_topK: {1}".format(balanced_random, balanced_topK))
        print("BinaryCE_wRejectionSMLoss | rejection_margin: {0}, hyparam_rejection: {1}".format(rejection_margin, hyparam_rejection))

    def forward(self, logits, wf, labels):
        # --------------------- BCE loss ---------------------
        # ----------------------------------------------------
        bce_loss = self.bce_loss_criterion(logits, labels)
        bce_loss_per_sample = bce_loss.sum(dim=1)

        # ------------- Rejection Regularization -------------
        # ----------------------------------------------------
        # --- Step 1: Extract relevant logits ---
        nonzero_indices = labels.nonzero(as_tuple=False)  # [N_active, 2]
        # If a sample has multiple 1s in its label, labels.nonzero() will return one row per active class, even if they come from the same sample.
        batch_indices = nonzero_indices[:, 0]  # [N_active]
        class_indices = nonzero_indices[:, 1]  # [N_active]
       
         # wf: [10, B, 10] → permute to [B, 10, 10] for indexing
        wf_transposed = wf.permute(1, 0, 2)  # [B, 10, 10]
        
        # Selected logits for each (sample, class) pair
        selected_logits = wf_transposed[batch_indices, class_indices]  # shape: [N_active, 10]

        # --- Step 2: Extract leftover logits for rejection regularizaion loss ---
        B, C, _ = wf_transposed.shape  # [B, 10, 10]
        total_pairs = torch.cartesian_prod(torch.arange(B, device=wf.device), torch.arange(C, device=wf.device))  # [B*C, 2]
        selected_pairs = torch.stack([batch_indices, class_indices], dim=1)  # [N_active, 2]

        # Use set difference to get leftover indices
        selected_set = set(map(tuple, selected_pairs.tolist()))
        leftover_pairs = [pair for pair in total_pairs.tolist() if tuple(pair) not in selected_set]
        leftover_pairs = torch.tensor(leftover_pairs, dtype=torch.long, device=wf.device)  # [N_left, 2]

        left_batch_indices = leftover_pairs[:, 0]
        left_class_indices = leftover_pairs[:, 1]
        leftover_logits = wf_transposed[left_batch_indices, left_class_indices]  # [N_left, 10]

        number_of_selected_feature = selected_logits.size()[0]
        number_of_leftover_feature = leftover_logits.size()[0]

        assert not (self.balanced_random and self.balanced_topK), "Choose only one balancing method at a time."
        if self.balanced_random:
            # Randomly choose the same number of leftover logits as selected_logits
            if number_of_leftover_feature >= number_of_selected_feature:
                rand_indices = torch.randperm(number_of_leftover_feature, device=wf.device)[:number_of_selected_feature]
                leftover_logits = leftover_logits[rand_indices]
                left_batch_indices = left_batch_indices[rand_indices]
            else:
                # Not enough leftover samples — keep all
                print("Warning: Not enough leftover logits for balanced_random. Using all leftovers.")
        if self.balanced_topK:
            # Use top-K highest confidence logits (by max-similarity score) from leftover_logits
            if number_of_leftover_feature >= number_of_selected_feature:
                max_sim = leftover_logits.max(dim=1)[0]  # [N_left]
                topk_values, topk_indices = torch.topk(max_sim, number_of_selected_feature)
                leftover_logits = leftover_logits[topk_indices]
                left_batch_indices = left_batch_indices[topk_indices]
            else:
                print("Warning: Not enough leftover logits for balanced_topK. Using all leftovers.")

        # print("selected_logits size: {0}".format(selected_logits.size()))
        # print("leftover_logits size: {0}".format(leftover_logits.size()))

        # --- step 3: Calculate rejection regularizaion loss ---
        if leftover_logits.numel() > 0:
            max_sim = leftover_logits.max(dim=1)[0]  # [N_left]
            max_sim_prob = torch.sigmoid(max_sim)  # convert to probability [0, 1]
            rejection_loss = torch.clamp(max_sim_prob - self.rejection_margin, min=0)
            rejection_loss_per_sample = torch.zeros(labels.size(0), device=wf.device)  # [batch_size]
            rejection_loss_per_sample = rejection_loss_per_sample.index_add(0, left_batch_indices, rejection_loss)  # sum per sample
        else:
            rejection_loss_per_sample = torch.tensor(0.0, device=wf.device)

        # ------------------ Combine losses ------------------ 
        # ----------------------------------------------------
        total_loss = bce_loss_per_sample + self.hyparam_rejection * rejection_loss_per_sample
        """
        print("BinaryCE_wRejectionSMLoss | total_loss: {0} | bce_loss_per_sample: {1}, rejection_loss_per_sample: {2}".format(
            total_loss, bce_loss_per_sample, rejection_loss_per_sample))
        """
        return total_loss

        