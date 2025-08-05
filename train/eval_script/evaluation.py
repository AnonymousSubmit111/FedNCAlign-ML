import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve

from configs.model_configs import model_configs
from configs.task_configs_fed import task_configs
from data.dataset_info import OPENI_LABEL_LIST, INC_TASK_OPENI_INDEX_TEST_LIST
from train.eval_script.evaluation_metric import accelerate_and_free_cache
from train.eval_script.evaluation_metric import compute_mcc, compute_precision, compute_recall, compute_specificity, compute_f1, compute_auc


def perform_eval(inc_step, args, logger, model, task_trainer, accelerator, device):
    model_config = model_configs[args.encoder_name]
    result_content = []

    with torch.no_grad():
        eval_result = task_trainer.eval(model)
        label_list = eval_result['label_list']
        soft_pred_list = eval_result['pred_list']

        hard_pred_list = []
        pred_threshold_list = []
        for cls_index, cls_cm in enumerate(label_list):          
            cls_soft_pred = soft_pred_list[cls_index]
            cls_label_list = label_list[cls_index]

            if args.pred_threshold_type == 'fix_05':
                cls_hard_pred = (cls_soft_pred > 0.5).astype(float)
                pred_threshold_list.append(0.5)
            elif args.pred_threshold_type == 'PR_curve':    
                precision, recall, thresholds = precision_recall_curve(cls_label_list, cls_soft_pred)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
                best_threshold = thresholds[np.argmax(f1_scores[:-1])]  # Find threshold maximizing F1-score
                cls_hard_pred = (cls_soft_pred > best_threshold).astype(float)
                pred_threshold_list.append(best_threshold)
            elif args.pred_threshold_type == 'class_mean':
                mean_value = np.mean(cls_soft_pred, axis=0)
                cls_hard_pred = (cls_soft_pred > mean_value).astype(float)
                pred_threshold_list.append(mean_value)
            else:
                raise ValueError("The prediction threshold type is set wrong!")
            hard_pred_list.append(cls_hard_pred)

        cm_matrix = []
        for cls_index, cls_hard_pred in enumerate(hard_pred_list):
            cm_class = confusion_matrix(label_list[cls_index], hard_pred_list[cls_index]).ravel()  # TN, FP, FN, TP
            if cm_class.shape == (1,):
                cm_class = np.array([cm_class[0], 0, 0, 0])
            cm_matrix.append(cm_class)
            
        cls_wise_precision = []
        cls_wise_recall = []
        cls_wise_f1 = []
        cls_wise_auc = []
        cls_wise_specificity = []
        total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
        for cls_index, cls_cm in enumerate(cm_matrix):  # cls_cm = [TN, FP, FN, TP]
            tn, fp, fn, tp = cls_cm
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) * 100.0 if (precision + recall) > 0 else 0
            auc = roc_auc_score(label_list[cls_index], soft_pred_list[cls_index]) * 100.0
            
            precision = precision * 100.0
            recall = recall * 100.0
            specificity = specificity * 100.0
            
            cls_wise_precision.append(precision)
            cls_wise_recall.append(recall)
            cls_wise_specificity.append(specificity)
            cls_wise_f1.append(f1)
            cls_wise_auc.append(auc)
            
            total_tn = total_tn + tn
            total_fp = total_fp + fp
            total_fn = total_fn + fn
            total_tp = total_tp + tp
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_avg_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) * 100.0 if (overall_precision + overall_recall) > 0 else 0
        flatten_label_list = np.concatenate(label_list)
        flatten_pred_list = np.concatenate(soft_pred_list)
        micro_avg_auc = roc_auc_score(flatten_label_list, flatten_pred_list) * 100.0

        current_step_cls_wise_precision = [cls_wise_precision[i] for i in range(len(cls_wise_precision)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
        current_step_cls_wise_recall = [cls_wise_recall[i] for i in range(len(cls_wise_recall)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
        current_step_cls_wise_f1 = [cls_wise_f1[i] for i in range(len(cls_wise_f1)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
        current_step_cls_wise_specificity = [cls_wise_specificity[i] for i in range(len(cls_wise_specificity)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
        current_step_cls_wise_auc = [cls_wise_auc[i] for i in range(len(cls_wise_auc)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
        current_step_cls_wise_mcc = [cls_wise_mcc[i] for i in range(len(cls_wise_mcc)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]

        # print("inc_step: {0} | INC_TASK_OPENI_INDEX_TEST_LIST: {1}".format(inc_step, INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]))
        current_step_cls_wise_f1_dist = {}
        current_step_cls_wise_auc_dist = {}
        current_step_cls_wise_mcc_dist = {}
        current_step_cls_wise_precision_dist = {}
        current_step_cls_wise_recall_dist = {}
        current_step_cls_wise_specificity_dist = {}
        current_step_pred_threshold_dist = {}
        for i in range(len(OPENI_LABEL_LIST)):
            if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]:
                current_step_cls_wise_f1_dist[OPENI_LABEL_LIST[i]] = cls_wise_f1[i]
                current_step_cls_wise_auc_dist[OPENI_LABEL_LIST[i]] = cls_wise_auc[i]
                current_step_cls_wise_mcc_dist[OPENI_LABEL_LIST[i]] = cls_wise_mcc[i]
                current_step_cls_wise_precision_dist[OPENI_LABEL_LIST[i]] = cls_wise_precision[i]
                current_step_cls_wise_recall_dist[OPENI_LABEL_LIST[i]] = cls_wise_recall[i]
                current_step_cls_wise_specificity_dist[OPENI_LABEL_LIST[i]] = cls_wise_specificity[i]
                current_step_pred_threshold_dist[OPENI_LABEL_LIST[i]] = pred_threshold_list[i]
        
        cls_avg_f1 = sum(current_step_cls_wise_f1) / len(current_step_cls_wise_f1)
        cls_avg_auc = sum(current_step_cls_wise_auc) / len(current_step_cls_wise_auc)
        cls_avg_precision = sum(current_step_cls_wise_precision) / len(current_step_cls_wise_precision)
        cls_avg_recall = sum(current_step_cls_wise_recall) / len(current_step_cls_wise_recall)
        cls_avg_specificity = sum(current_step_cls_wise_specificity) / len(current_step_cls_wise_specificity)

        org_f1_score = eval_result['f1_score']
        org_auc_score = eval_result['auc_score']
        
        logger.info("* Evaluation | inc_step: {0} *".format(inc_step))
        logger.info("* class-wise prediction threshold = {}".format(current_step_pred_threshold_dist))
        logger.info("* cls-avg (macro-avg) F1 score = {:.2f}% | micro-avg F1 score = {:.2f}% | cls-wise F1 score = {}".format(cls_avg_f1, micro_avg_f1, current_step_cls_wise_f1_dist))
        logger.info("* cls-avg (macro-avg) AUC score = {:.2f}% | micro_avg_auc score = {:.2f}% | cls-wise AUC score = {}".format(cls_avg_auc, micro_avg_auc, current_step_cls_wise_auc_dist))
        logger.info("* cls-avg (macro-avg) AUC score = {:.2f}% | micro_avg_auc score = {:.2f}% | cls-wise AUC score = {}".format(cls_avg_auc, micro_avg_auc, current_step_cls_wise_auc_dist))
        logger.info("* cls-avg precision = {:.2f}% | cls-wise precision = {}".format(cls_avg_precision, current_step_cls_wise_precision_dist))
        logger.info("* cls-avg recall = {:.2f}% | cls-wise recall = {}".format(cls_avg_recall, current_step_cls_wise_recall_dist))
        logger.info("* cls-avg specificity = {:.2f}% | cls-wise specificity = {}".format(cls_avg_specificity, current_step_cls_wise_specificity_dist))
        logger.info("* org_f1_score = {:.2f}% | org_f1_score = {:.2f}%".format(org_f1_score, org_auc_score))
        
        result_content.append("* Evaluation | inc_step: {0} *".format(inc_step))
        result_content.append("* cls-avg (macro-avg) F1 score = {:.2f}% | micro-avg F1 score = {:.2f}% | cls-wise F1 score = {}".format(cls_avg_f1, micro_avg_f1, current_step_cls_wise_f1_dist))
        result_content.append("* cls-avg (macro-avg) AUC score = {:.2f}% | micro_avg_auc score = {:.2f}% | cls-wise AUC score = {}".format(cls_avg_auc, micro_avg_auc, current_step_cls_wise_auc_dist))
        result_content.append("* cls-avg precision = {:.2f}% | cls-wise precision = {}".format(cls_avg_precision, current_step_cls_wise_precision_dist))
        result_content.append("* cls-avg recall = {:.2f}% | cls-wise recall = {}".format(cls_avg_recall, current_step_cls_wise_recall_dist))
        result_content.append("* cls-avg specificity = {:.2f}% | cls-wise specificity = {}".format(cls_avg_specificity, current_step_cls_wise_specificity_dist))
        result_content.append("* org_f1_score = {:.2f}% | org_f1_score = {:.2f}%".format(org_f1_score, org_auc_score))


        del task_trainer
        accelerate_and_free_cache(accelerator)
          
    return result_content