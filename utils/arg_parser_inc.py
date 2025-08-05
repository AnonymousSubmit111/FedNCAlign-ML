import argparse

from configs.model_configs import ALLOWED_CL_ENCODERS
from configs.adapter_configs import ADAPTER_MAP


def get_parser():
    # --------------- Create a parser ---------------
    parser = argparse.ArgumentParser()

    # --------------- Add arguments ---------------
    ## Model parameters
    parser.add_argument("--encoder_name", default=None, type=str, required=True, choices=ALLOWED_CL_ENCODERS, help="The name of the base pretrained encoder.")
    parser.add_argument("--pretrained_model_name", default=None, type=str, required=True, help="Name of pretrained model weights to load.")
    parser.add_argument("--client_specific_head", action="store_true", help="Flag for whether using client specific head for each client or using shared head for all clients.")
    parser.add_argument("--obtain_class_wise_feature", action="store_true", help="Flag for whether using class-specific MLP for each class or attention block to obtain single-semantic feature.")
    parser.add_argument("--projection_type", default="MLP", type=str, 
                        help="The name of the projection_type: class_wise_MLP, " \
                        "Attention_learnable_random_init, Attention_learnable_etf_init, Attention_fixed_etf_init.")
    parser.add_argument("--norm_type", type=str, default='batch_norm', help="Choose from batch_norm, layer_norm, instance_norm.")
   

    parser.add_argument("--portion", default=1.0, type=float, help="The name of optimization mode.")
    parser.add_argument("--optimizer_mode", default='none', type=str, help="The name of optimization mode. Choose from: dat, adapter, freeze_bottom_k_layers, full")
    parser.add_argument("--debug", type=int, default=0, help="If True, debug the code with minimum setting")

    parser.add_argument("--do_train", action='store_true', help="If True, train the model.")
    parser.add_argument("--do_test", action='store_true', help="If True, evaluate pre-trained model.")
    parser.add_argument("--test_train_set", action='store_true', help="If True, evaluate pre-trained model on the trainset.")
    parser.add_argument("--multiplicity_1", action='store_true', help="If True, evaluate pre-trained model only on data with multiplicity = 1 (single label).")

    # Arguments specific to Adapters algorithm
    parser.add_argument("--adapter_config", choices=list(ADAPTER_MAP.keys()), help="Type of Adapter architecture")
    parser.add_argument("--adapter_reduction_factor", type=int, default=0, help="Downsampling ratio for adapter layers")
    
    # Arguments specific to frozen bottom-k layers algorithm
    parser.add_argument("--layers_to_freeze", type=int, default=0, help="Number of layers to freeze (if freezing bottom-k layers)")
    parser.add_argument("--output_dir", type=str, required=True, help="Name of output directory, where all experiment results and checkpoints are saved.")
    
    parser.add_argument("--do_wandb_logging", action="store_true", help="Log experiments in W&B.")
    parser.add_argument("--wandb_freq", type=int, default=100, help="Log frequency in W&B.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=15, help="Maximum number of epochs to train.")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Test Batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloader")
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--comm_rounds", type=int, default=20, help="Number of communication rounds.")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of communication rounds.")
    
    parser.add_argument("--checkpoint", type=str, default=None, help="path to the checkpoint of singletask_ft, which is loaded for testing")
    parser.add_argument("--model_path", type=str, default=None, help="path to the model for evaluation")
    
    parser.add_argument("--data_dir", type=str, required=True, default='/data/datasets/MCL/', help="Directory where all the MCL data is stored")
    parser.add_argument("--json_text_folder", type=str, required=True, default='json_text_folder', help="Name of the dataset json text file.")
    parser.add_argument("--json_img_folder", type=str, required=True, default='json_img_folder', help="ame of the dataset json image file.")

    parser.add_argument("--dataset_name", type=str, help="Choose from openi, mimic.")
    parser.add_argument("--seperate_background_class",action='store_true', help="For samples with no pulse (healthy sample), regard them as a new class.")
    
    parser.add_argument("--num_cl_tasks", type=int, default=1, help="Number of incremental tasks.")
    parser.add_argument("--num_fl_tasks", type=int, default=1, help="Number of federated tasks.")
    parser.add_argument("--partial_client_join", action="store_true", help="Flag for only part of clients join training per round.")
    parser.add_argument("--num_of_client_per_round", type=int, default=-1, help="Number of clients participate in each round if partial_client_join.")


    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--customer_label_list_flag", action="store_true", help="Flag for manually entering interested label names.")
    parser.add_argument("--customer_label_list", nargs="+", help="Manually entering interested label names.")

    parser.add_argument("--loss_type", type=str, required=True, default='binary_ce',
                        help="Choose from: binary_ce, focal_binary_ce, class_balanced_binary_ce, focal_binary_ce_no_alpha"
                        "cross_entropy, focal_ce, focal_ce_no_alpha, cosface"
                        "asymmetric_focal_binary_ce, asymmetric_optimized_focal_binary_ce," \
                        "cross_entropy_w_rejection, cosface_w_rejection, binary_ce_w_rejection, dual_binary_ce," \
                        "binary_ce_w_rejection_balanced_by_random, binary_ce_w_rejection_balanced_by_topK," \
                        "binary_ce_w_feature_contrastive_cosine, binary_ce_w_feature_contrastive_PSC," \
                        "binary_ce_w_dual_reg_rejection_n_contrastive_all_PSC, binary_ce_w_dual_reg_rejection_n_contrastive_all_cosine," \
                        "binary_ce_w_dual_reg_rejection_n_contrastive_topK_PSC, binary_ce_w_dual_reg_rejection_n_contrastive_topK_cosine,")
    parser.add_argument("--focal_bce_flag", action="store_true", help="Flag for using focal binary loss instead of binary loss.")

    parser.add_argument("--rejection_loss_threshold", default=0.3, type=float, 
                        help="The negative feature rejection loss to remove possible noisy effect from negative features.")
    parser.add_argument("--hyparam_rejection_loss", default=1, type=float, help="The hyperparameter for balancing regularization loss.")
    parser.add_argument("--hyparam_contractive_loss", default=1, type=float, help="The hyperparameter for balancing regularization loss.")
    
    
    parser.add_argument("--classifier_type", type=str, required=True, default='FC_Classifier',
                        help="Choose from: FC_Classifier, Cosine_Classifier," \
                        "MultiLabel_ETF_Classifier, MultiLabel_ETF_Classifier_w_feature_normalized, Dual_Classifier.")

    parser.add_argument("--cl_method", type=str, required=True, default='direct_ft', help="Choose from: direct_ft, lwf.")
    parser.add_argument("--fl_method", type=str, required=True, default='centralized', help="Choose from: centralized, fedavg.")
    parser.add_argument("--fedavg_w_data_amount_weighted", action="store_true", help="Weighted averaging of client models with number of data each client contain.")
    parser.add_argument("--fedavg_importance_ab_param_wise", action="store_true", help="Model aggregation with importance (absolute value).")
    parser.add_argument("--fedavg_importance_ab_model_wise", action="store_true", help="Model aggregation with importance (absolute value).")
    parser.add_argument("--fedavg_importance_fim_param_wise", action="store_true", help="Model aggregation with importance (fisher information matrix).")
    parser.add_argument("--fedavg_importance_fim_model_wise", action="store_true", help="Model aggregation with importance (fisher information matrix).")
    parser.add_argument("--fedavg_label_distribution", action="store_true", help="Model aggregation with label distribution infromation.")
    parser.add_argument("--fedavg_importance_ab_model_wise_w_label_distribution", action="store_true", help="Model aggregation with label distribution infromation.")
    parser.add_argument("--fedavg_importance_wanda_param_wise", action="store_true", help="Model aggregation with importance (fisher information matrix).")
    parser.add_argument("--fedavg_importance_wanda_model_wise", action="store_true", help="Model aggregation with importance (fisher information matrix).")
    
    parser.add_argument("--evaluate_only_first_client_flag", action="store_true", 
                        help="Flag for only the first client model for general FL since the received global model is the same for all clients.")
    
    parser.add_argument("--remove_old_cls_label", action="store_true", 
                        help="Flag for remove all old class labels from the given gound truth labels.")

    parser.add_argument("--get_pred_probability_function", default='sigmoid',
                        help="Choose from: sigmoid, softmax, non.")
    parser.add_argument("--pred_threshold_type", type=str, required=True, default='fix_05',
                        help="Choose from fix_05, PR_curve, class_mean, max_w_tolerance.")
    parser.add_argument("--max_w_tolerance_value", default=0.02, type=float, help="The tolerance to find the hard prediction using max_w_tolerance.")

    parser.add_argument("--img_augmentation", action="store_true", 
                        help="Apply traditional data augmentation to image data.")
    
    parser.add_argument("--remove_text", action="store_true", help="Remove text data from the data when send it to the model.")
    parser.add_argument("--remove_image", action="store_true", help="Remove text data from the data when send it to the model.")
    
    parser.add_argument("--scale_target", action="store_true", 
                        help="Scale the one-hot target to probability distribution according to number of labels it has.")
    parser.add_argument("--peak_uniformity_regularizer", action="store_true", 
                        help="Apply peak_uniformity_regularizer to obtain uniform distributed peaks.")
    parser.add_argument("--CenterLoss_regularization", action="store_true", 
                        help="Apply Center Loss regularization to the model training.")
    parser.add_argument("--HNM_regularization", action="store_true", 
                        help="Apply Hard Negative Mining (HNM) regularization to the model training.")

    parser.add_argument("--fedprox_mu", default=0.01, type=float, 
                        help="Proximal term coefficient for FedProx; controls the strength of regularization to keep local models close to the global model.")
    parser.add_argument("--fedcurv_lambda", default=0.1, type=float, 
                        help="EWC regularization loss term coefficient for fedcurv_lambda; controls the strength of regularization to keep local models close to the global model.")
    return parser
