import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve

from configs.model_configs import model_configs
from configs.task_configs_fed import task_configs
from data.dataset_info import OPENI_LABEL_LIST, INC_TASK_OPENI_INDEX_TEST_LIST, MNIST_LABEL_LIST, CIFAR10_LABEL_LIST
from data.dataset_info import VOC2012_LABEL_CLASSES, ISIC2018_LABEL_CLASSES, XRAY14_LABEL_CLASSES
from data.dataset_info import CHESTMNIST_LABEL_LIST, CHESTMNIST_LABEL_LIST_W_BACKGROUND, SKIN_LABEL_LIST, CANCER_LABEL_LIST
from train.eval_script.evaluation_metric import accelerate_and_free_cache
from train.eval_script.evaluation_metric import compute_mcc, compute_acc, compute_precision, compute_recall, compute_specificity, compute_f1, compute_auc


def perform_eval_for_local_train(inc_step, comm_round, client_index, args, logger, tensorboard_writer, model, task_trainer, accelerator, device):
    if "openi" in args.dataset_name or "mimic" in args.dataset_name:
        LABEL_LIST = OPENI_LABEL_LIST
    elif "medmnist_chest" in args.dataset_name:
        if args.seperate_background_class:
            LABEL_LIST = CHESTMNIST_LABEL_LIST_W_BACKGROUND
        else:
            LABEL_LIST = CHESTMNIST_LABEL_LIST
    elif "medmnist_path" in args.dataset_name:
        LABEL_LIST = CANCER_LABEL_LIST
    elif "medmnist_derma" in args.dataset_name:
        LABEL_LIST = SKIN_LABEL_LIST
    elif "mnist" in args.dataset_name:
        LABEL_LIST = MNIST_LABEL_LIST
    elif "cifar10" in args.dataset_name:
        LABEL_LIST = CIFAR10_LABEL_LIST
    elif "voc2012" in args.dataset_name:
        LABEL_LIST = VOC2012_LABEL_CLASSES
    elif "isic2018" in args.dataset_name:
        LABEL_LIST = ISIC2018_LABEL_CLASSES
    elif "xray14" in args.dataset_name:
        LABEL_LIST = XRAY14_LABEL_CLASSES
    else:
        raise ValueError("Something wrong with the LABEL_LIST!")
    
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

            if args.pred_threshold_type in ['fix_05', 'PR_curve', 'class_mean']:
                hard_pred_list = []
                pred_threshold_list = []
                for cls_index, _ in enumerate(label_list):          
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
            elif args.pred_threshold_type in ['max_w_tolerance']:
                soft_pred_tensor = torch.tensor(soft_pred_list)
                # Find the maximum value for each batch along the class dimension (dim=0)
                max_values, _ = torch.max(soft_pred_tensor, dim=0, keepdim=True)  # Shape: [1, batch_size]
                # Find all indices where the value is within the tolerance of the maximum value
                peak_mask = (soft_pred_tensor >= max_values - args.max_w_tolerance_value) & (soft_pred_tensor <= max_values + args.max_w_tolerance_value)
                # Initialize a multi-hot label tensor with zeros
                multi_hot_labels = torch.zeros_like(soft_pred_tensor, dtype=torch.int)
                # Set the positions corresponding to peak indices to 1
                multi_hot_labels[peak_mask] = 1
                # Convert the multi-hot label tensor to a 2-dimensional list
                multi_hot_labels_list = multi_hot_labels.tolist()
                hard_pred_list = multi_hot_labels_list
                pred_threshold_list = [-1 for cls_index, _ in enumerate(label_list)]

        # ------ get class-wise confusion matrix for all classes ------
        cm_matrix = []
        for cls_index, cls_hard_pred in enumerate(hard_pred_list):
            cm_class = confusion_matrix(label_list[cls_index], hard_pred_list[cls_index]).ravel()  # TN, FP, FN, TP
            if cm_class.shape == (1,):
                cm_class = np.array([cm_class[0], 0, 0, 0])
            cm_matrix.append(cm_class)

        # ------ get class-wise performance evaluation for all classes ------
        cls_wise_precision, cls_wise_recall, cls_wise_specificity = [], [], []
        cls_wise_f1, cls_wise_auc, cls_wise_acc, cls_wise_mcc = [], [], [], []
        cls_wise_tn, cls_wise_fp, cls_wise_fn, cls_wise_tp = [], [], [], []
        for cls_index, cls_cm in enumerate(cm_matrix):  # cls_cm = [TN, FP, FN, TP]
            tn, fp, fn, tp = cls_cm
            precision = compute_precision(tp, tn, fp, fn)
            recall = compute_recall(tp, tn, fp, fn)
            specificity = compute_specificity(tp, tn, fp, fn)
            f1 = compute_f1(precision, recall)
            auc = compute_auc(label_list[cls_index], soft_pred_list[cls_index])
            mcc = compute_mcc(tp, tn, fp, fn)
            acc = compute_acc(tp, tn, fp, fn)

            cls_wise_precision.append(precision)
            cls_wise_recall.append(recall)
            cls_wise_specificity.append(specificity)
            cls_wise_f1.append(f1)
            cls_wise_auc.append(auc)
            cls_wise_mcc.append(mcc)
            cls_wise_acc.append(acc)
            cls_wise_tn.append(tn)
            cls_wise_fp.append(fp)
            cls_wise_fn.append(fn)
            cls_wise_tp.append(tp)
        
        # ------ get class-wise performance evaluation for current step classes ------
        if not args.customer_label_list_flag:
            current_step_cls_wise_precision = [cls_wise_precision[i] for i in range(len(cls_wise_precision)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_recall = [cls_wise_recall[i] for i in range(len(cls_wise_recall)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_f1 = [cls_wise_f1[i] for i in range(len(cls_wise_f1)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_specificity = [cls_wise_specificity[i] for i in range(len(cls_wise_specificity)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_auc = [cls_wise_auc[i] for i in range(len(cls_wise_auc)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_mcc = [cls_wise_mcc[i] for i in range(len(cls_wise_mcc)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_acc = [cls_wise_acc[i] for i in range(len(cls_wise_acc)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_tn = [cls_wise_tn[i] for i in range(len(cls_wise_tn)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_fp = [cls_wise_fp[i] for i in range(len(cls_wise_fp)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_fn = [cls_wise_fn[i] for i in range(len(cls_wise_fn)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_cls_wise_tp = [cls_wise_tp[i] for i in range(len(cls_wise_tp)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_label_list = [label_list[i] for i in range(len(label_list)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
            current_step_soft_label_list = [soft_pred_list[i] for i in range(len(soft_pred_list)) if i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]]
        else:
            customer_label_index_list = []
            for index, label in enumerate(LABEL_LIST):
                if label in args.customer_label_list:
                    customer_label_index_list.append(index)
            # print("----- evaluation | LABEL_LIST: {0}".format(LABEL_LIST))
            # print("----- evaluation | args.customer_label_list: {0}".format(args.customer_label_list))
            # print("----- evaluation | customer_label_index_list: {0}".format(customer_label_index_list))
            current_step_cls_wise_precision = [cls_wise_precision[i] for i in range(len(cls_wise_precision)) if i in customer_label_index_list]
            current_step_cls_wise_recall = [cls_wise_recall[i] for i in range(len(cls_wise_recall)) if i in customer_label_index_list]
            current_step_cls_wise_f1 = [cls_wise_f1[i] for i in range(len(cls_wise_f1)) if i in customer_label_index_list]
            current_step_cls_wise_specificity = [cls_wise_specificity[i] for i in range(len(cls_wise_specificity)) if i in customer_label_index_list]
            current_step_cls_wise_auc = [cls_wise_auc[i] for i in range(len(cls_wise_auc)) if i in customer_label_index_list]
            current_step_cls_wise_mcc = [cls_wise_mcc[i] for i in range(len(cls_wise_mcc)) if i in customer_label_index_list]
            current_step_cls_wise_acc = [cls_wise_acc[i] for i in range(len(cls_wise_acc)) if i in customer_label_index_list]
            current_step_cls_wise_tn = [cls_wise_tn[i] for i in range(len(cls_wise_tn)) if i in customer_label_index_list]
            current_step_cls_wise_fp = [cls_wise_fp[i] for i in range(len(cls_wise_fp)) if i in customer_label_index_list]
            current_step_cls_wise_fn = [cls_wise_fn[i] for i in range(len(cls_wise_fn)) if i in customer_label_index_list]
            current_step_cls_wise_tp = [cls_wise_tp[i] for i in range(len(cls_wise_tp)) if i in customer_label_index_list]
            current_step_label_list = [label_list[i] for i in range(len(label_list)) if i in customer_label_index_list]
            current_step_soft_label_list = [soft_pred_list[i] for i in range(len(soft_pred_list)) if i in customer_label_index_list]

        # print("inc_step: {0} | INC_TASK_OPENI_INDEX_TEST_LIST: {1}".format(inc_step, INC_TASK_OPENI_INDEX_TEST_LIST[inc_step]))
        current_step_cls_wise_f1_dist = {}
        current_step_cls_wise_auc_dist = {}
        current_step_cls_wise_mcc_dist = {}
        current_step_cls_wise_acc_dist = {}
        current_step_cls_wise_precision_dist = {}
        current_step_cls_wise_recall_dist = {}
        current_step_cls_wise_specificity_dist = {}
        current_step_pred_threshold_dist = {}
        current_step_label_name_list = []
        for i in range(len(LABEL_LIST)):
            if ((not args.customer_label_list_flag) and (i in INC_TASK_OPENI_INDEX_TEST_LIST[inc_step])) or ((args.customer_label_list_flag) and (i in customer_label_index_list)):
                current_step_cls_wise_f1_dist[LABEL_LIST[i]] = cls_wise_f1[i]
                current_step_cls_wise_auc_dist[LABEL_LIST[i]] = cls_wise_auc[i]
                current_step_cls_wise_mcc_dist[LABEL_LIST[i]] = cls_wise_mcc[i]
                current_step_cls_wise_acc_dist[LABEL_LIST[i]] = cls_wise_acc[i]
                current_step_cls_wise_precision_dist[LABEL_LIST[i]] = cls_wise_precision[i]
                current_step_cls_wise_recall_dist[LABEL_LIST[i]] = cls_wise_recall[i]
                current_step_cls_wise_specificity_dist[LABEL_LIST[i]] = cls_wise_specificity[i]
                current_step_pred_threshold_dist[LABEL_LIST[i]] = pred_threshold_list[i]
                current_step_label_name_list.append(LABEL_LIST[i])
        
        # ------ get class-average performance for current step classes ------
        cls_avg_f1 = sum(current_step_cls_wise_f1) / len(current_step_cls_wise_f1)
        cls_avg_auc = sum(current_step_cls_wise_auc) / len(current_step_cls_wise_auc)
        cls_avg_mcc = sum(current_step_cls_wise_mcc) / len(current_step_cls_wise_mcc)
        cls_avg_acc = sum(current_step_cls_wise_acc) / len(current_step_cls_wise_acc)
        cls_avg_precision = sum(current_step_cls_wise_precision) / len(current_step_cls_wise_precision)
        cls_avg_recall = sum(current_step_cls_wise_recall) / len(current_step_cls_wise_recall)
        cls_avg_specificity = sum(current_step_cls_wise_specificity) / len(current_step_cls_wise_specificity)

        total_tn = sum(current_step_cls_wise_tn)
        total_fp = sum(current_step_cls_wise_fp)
        total_fn = sum(current_step_cls_wise_fn)
        total_tp = sum(current_step_cls_wise_tp)
        overall_precision = compute_precision(total_tp, total_tn, total_fp, total_fn)
        overall_recall = compute_recall(total_tp, total_tn, total_fp, total_fn)
        overall_specificity = compute_specificity(total_tp, total_tn, total_fp, total_fn)
        micro_avg_f1 = compute_f1(overall_precision, overall_recall)
        micro_avg_mcc = compute_mcc(total_tp, total_tn, total_fp, total_fn)
        micro_avg_acc = compute_acc(total_tp, total_tn, total_fp, total_fn)
        flatten_label_list = np.concatenate(current_step_label_list)
        flatten_pred_list = np.concatenate(current_step_soft_label_list)
        micro_avg_auc = compute_auc(flatten_label_list, flatten_pred_list)
    
        org_f1_score = eval_result['f1_score']
        org_auc_score = eval_result['auc_score']
        
        # ------ record logger ------
        logger.info("* Evaluation | inc_step: {0}, comm_round: {1}, client: {2}, centralized_train *".format(inc_step, comm_round + 1, client_index + 1))
        logger.info("* class-wise prediction threshold = {}".format(current_step_pred_threshold_dist))
        logger.info("* cls-avg (macro-avg) F1 score = {:.2f}% | micro-avg F1 score = {:.2f}% | cls-wise F1 score = {}".format(cls_avg_f1 * 100, micro_avg_f1 * 100, current_step_cls_wise_f1_dist))
        logger.info("* cls-avg (macro-avg) AUC = {:.2f}% | micro_avg_auc score = {:.2f}% | cls-wise AUC score = {}".format(cls_avg_auc * 100, micro_avg_auc * 100, current_step_cls_wise_auc_dist))
        logger.info("* cls-avg (macro-avg) MCC = {:.2f}% | micro_avg_mcc score = {:.2f}% | cls-wise MCC score = {}".format(cls_avg_mcc * 100, micro_avg_mcc * 100, current_step_cls_wise_mcc_dist))
        logger.info("* cls-avg (macro-avg) ACC = {:.2f}% | micro_avg_acc score = {:.2f}% | cls-wise ACC score = {}".format(cls_avg_acc * 100, micro_avg_acc * 100, current_step_cls_wise_acc_dist))
        logger.info("* cls-avg precision = {:.2f}% | cls-wise precision = {}".format(cls_avg_precision * 100, current_step_cls_wise_precision_dist))
        logger.info("* cls-avg recall = {:.2f}% | cls-wise recall = {}".format(cls_avg_recall * 100, current_step_cls_wise_recall_dist))
        logger.info("* cls-avg specificity = {:.2f}% | cls-wise specificity = {}".format(cls_avg_specificity * 100, current_step_cls_wise_specificity_dist))
        logger.info("* org_f1_score = {:.2f}% | org_f1_score = {:.2f}%".format(org_f1_score, org_auc_score))
        
        result_content.append("* Evaluation | inc_step: {0}, comm_round: {1}, client: {2}, centralized_train *".format(inc_step, comm_round + 1, client_index + 1))
        result_content.append("* cls-avg (macro-avg) F1 score = {:.2f}% | micro-avg F1 score = {:.2f}% | cls-wise F1 score = {}".format(cls_avg_f1 * 100, micro_avg_f1 * 100, current_step_cls_wise_f1_dist))
        result_content.append("* cls-avg (macro-avg) AUC = {:.2f}% | micro_avg_auc score = {:.2f}% | cls-wise AUC score = {}".format(cls_avg_auc * 100, micro_avg_auc * 100, current_step_cls_wise_auc_dist))
        result_content.append("* cls-avg (macro-avg) MCC = {:.2f}% | micro_avg_mcc score = {:.2f}% | cls-wise MCC score = {}".format(cls_avg_mcc * 100, micro_avg_mcc * 100, current_step_cls_wise_mcc_dist))
        result_content.append("* cls-avg (macro-avg) ACC = {:.2f}% | micro_avg_acc score = {:.2f}% | cls-wise ACC score = {}".format(cls_avg_acc * 100, micro_avg_acc * 100, current_step_cls_wise_acc_dist))
        result_content.append("* cls-avg precision = {:.2f}% | cls-wise precision = {}".format(cls_avg_precision * 100, current_step_cls_wise_precision_dist))
        result_content.append("* cls-avg recall = {:.2f}% | cls-wise recall = {}".format(cls_avg_recall * 100, current_step_cls_wise_recall_dist))
        result_content.append("* cls-avg specificity = {:.2f}% | cls-wise specificity = {}".format(cls_avg_specificity * 100, current_step_cls_wise_specificity_dist))
        result_content.append("* org_f1_score = {:.2f}% | org_f1_score = {:.2f}%".format(org_f1_score, org_auc_score))

        # ------ tensorboard_writer ------
        tensorboard_writer.add_scalar('class_average_f1/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_f1, comm_round)
        tensorboard_writer.add_scalar('class_average_auc/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_auc, comm_round)
        tensorboard_writer.add_scalar('class_average_mcc/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_mcc, comm_round)
        tensorboard_writer.add_scalar('class_average_acc/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_acc, comm_round)
        tensorboard_writer.add_scalar('class_average_precision/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_precision, comm_round)
        tensorboard_writer.add_scalar('class_average_recall/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_recall, comm_round)
        tensorboard_writer.add_scalar('class_average_specificity/step_{0}_client_{1}'.format(inc_step, client_index+1), cls_avg_specificity, comm_round)
        tensorboard_writer.add_scalar('org_F1_Centralized/step_{0}_client_{1}'.format(inc_step, client_index+1), org_f1_score, comm_round)
        tensorboard_writer.add_scalar('org_AUC_Centralized/step_{0}_client_{1}'.format(inc_step, client_index+1), org_auc_score, comm_round)

        # ------ record all results ------
        result_dict = {}
        result_dict["comm_round"] = comm_round
        result_dict["class_average_f1"] = cls_avg_f1
        result_dict["class_average_auc"] = cls_avg_auc
        result_dict["class_average_mcc"] = cls_avg_mcc
        result_dict["class_average_acc"] = cls_avg_acc
        result_dict["class_average_precision"] = cls_avg_precision
        result_dict["class_average_recall"] = cls_avg_recall
        result_dict["class_average_specificity"] = cls_avg_specificity
        result_dict["current_step_cls_wise_f1"] = current_step_cls_wise_f1
        result_dict["current_step_cls_wise_auc"] = current_step_cls_wise_auc
        result_dict["current_step_cls_wise_mcc"] = current_step_cls_wise_mcc
        result_dict["current_step_cls_wise_acc"] = current_step_cls_wise_acc
        result_dict["current_step_cls_wise_precision"] = current_step_cls_wise_precision
        result_dict["current_step_cls_wise_recall"] = current_step_cls_wise_recall
        result_dict["current_step_cls_wise_specificity"] = current_step_cls_wise_specificity
        result_dict["current_step_label_name_list"] = current_step_label_name_list

        del task_trainer
        accelerate_and_free_cache(accelerator)
          
    return result_content, cls_avg_f1, result_dict


