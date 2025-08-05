import torch
import os
import copy
import torch.nn as nn
import torch.distributed as dist
from typing import Dict
import sys
import random
Newlimit = 5000  # New limit 
sys.setrecursionlimit(Newlimit)  # Using sys.setrecursionlimit() method  
sys.path.insert(0, ".")

from configs.task_configs_fed import task_configs
from aggregation.fedavg import get_average_net
from aggregation.param_wise_importance_aware_fedavg import get_parameter_importance_by_absolution, get_importance_weighted_net_parameter_wise
from aggregation.param_wise_label_distribution_aware_fedavg import get_label_distribution_aware_average_net

from train.eval_script.evaluation_fl import perform_eval_for_federated_train

from visualization.radar_figure import plot_radar_chart_figure


def accelerate_and_free_cache(accelerator):
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.free_memory()   

def generate_random_integers(n, m):
    """
    Generate n random integers in the range [0, m].

    Parameters:
        n (int): Number of integers to generate.
        m (int): Upper bound (inclusive) of the range.

    Returns:
        List[int]: A list of n random integers between 0 and m.
    """
    return [random.randint(0, m) for _ in range(n)]

def record_training_information(args, logger, tensorboard_writer, loss_dict, task_key, inc_step, client_index, comm_round):
    loss_ce = loss_dict["loss_ce"]
    etf_reg = loss_dict["etf_reg"]
    center_loss_reg = loss_dict["center_loss_reg"]
    hnm_loss = loss_dict["hnm_loss_reg"]
    loss_total = loss_dict["loss_total"]
    if args.CenterLoss_regularization:
        logger.info("Task: {} | inc_tep: {}, client: {}, comm_round: {}/{}, loss: {:.2f} (loss_ce: {:.2f}, etf_reg: {:.2f}, center_loss_reg: {:.2f})".
                format(task_key, inc_step, client_index + 1, comm_round, args.comm_rounds, loss_total, loss_ce, etf_reg, center_loss_reg))
    elif args.HNM_regularization:
        logger.info("Task: {} | inc_tep: {}, client: {}, comm_round: {}/{}, loss: {:.2f} (loss_ce: {:.2f}, etf_reg: {:.2f}, hnm_loss_reg: {:.2f})".
                format(task_key, inc_step, client_index + 1, comm_round, args.comm_rounds, loss_total, loss_ce, etf_reg, hnm_loss))
    else:
        logger.info("Task: {} | inc_tep: {}, client: {}, comm_round: {}/{}, loss: {:.2f} (loss_ce: {:.2f}, etf_reg: {:.2f})".
                format(task_key, inc_step, client_index + 1, comm_round, args.comm_rounds, loss_total, loss_ce, etf_reg))
    tensorboard_writer.add_scalar('loss_total/step_{0}_client_{1}'.format(inc_step, client_index+1), loss_total, comm_round)
    tensorboard_writer.add_scalar('loss_ce/step_{0}_client_{1}'.format(inc_step, client_index+1), loss_ce, comm_round)
    tensorboard_writer.add_scalar('loss_etf/step_{0}_client_{1}'.format(inc_step, client_index+1), etf_reg, comm_round)

def strip_module_prefix(fim_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Removes the 'module.' prefix from keys in a state_dict or FIM dictionary.

    Args:
        fim_dict: Dictionary of parameter names to tensors (e.g., FIM values)

    Returns:
        A new dictionary with 'module.' prefix stripped from keys.
    """
    new_fim_dict = {}

    for key, value in fim_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_fim_dict[new_key] = value

    return new_fim_dict


def fedavg_train(inc_step, args, logger, tensorboard_writer, model, model_config, device, accelerator):
    #  --------- start federated training and communication  ---------
    best_f1 = 0
    task_output_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.isdir(task_output_dir):
        os.makedirs(task_output_dir, exist_ok=True)

    print("fedavg_train | number of clients: {0}".format(len(args.ordered_fcl_tasks[inc_step])))
    print("fedavg_train | partial_client_join: {0}".format(args.partial_client_join))
    print("fedavg_train | num_of_client_per_round: {0}".format(args.num_of_client_per_round))

    for comm_round in range(args.comm_rounds):  # ------- For each round -------
        c_models_list = []

        if args.partial_client_join:
            chosen_client_index = generate_random_integers(args.num_of_client_per_round, len(args.client_list))
            client_weight = [1 for _ in range(args.num_of_client_per_round)]
        else:
            client_weight = [1 for _ in range(len(args.client_list))]

        client_output_dir_list = []

        if args.fedavg_importance_fim_param_wise or args.fedavg_importance_fim_model_wise:
            c_models_fim_list = []
            model_avg_fim = []
        if args.fedavg_label_distribution or args.fedavg_importance_ab_model_wise_w_label_distribution:
            c_models_label_dis = []
        if args.fedavg_importance_wanda_param_wise or args.fedavg_importance_wanda_model_wise:
            c_models_wanda_list = []
            model_avg_wanda = []

        for client_index, task_key in enumerate(args.ordered_fcl_tasks[inc_step]):  # ------- For each client -------
            client_output_dir = os.path.join(args.output_dir, "checkpoints", "client{}_{}".format(client_index + 1, task_key))
            if not os.path.isdir(client_output_dir):
                os.makedirs(client_output_dir, exist_ok=True)
            client_output_dir_list.append(client_output_dir)

            if args.partial_client_join and not (client_index in chosen_client_index):
                continue

            # Local train the model
            temp_model = copy.deepcopy(model)
            task_trainer_class = task_configs[args.task_config_key]["task_trainer"]
            task_trainer = task_trainer_class(logger, args, task_configs, model_config, device, task_key, task_output_dir, accelerator=accelerator)
            loss_dict, c_model = task_trainer.train(temp_model, task_key)
            if args.fedavg_w_data_amount_weighted:
                client_weight[client_index] = task_trainer.get_num_of_training_data()
            if args.fedavg_label_distribution or args.fedavg_importance_ab_model_wise_w_label_distribution:
                c_models_label_dis.append(task_trainer.get_training_data_label_distribution())
            if args.fedavg_importance_fim_param_wise or args.fedavg_importance_fim_model_wise:
                c_model_fim, c_model_avg_fim_value = task_trainer.compute_fisher_information_matrix(c_model)
                c_model_fim = strip_module_prefix(c_model_fim)
                c_models_fim_list.append(c_model_fim)
                model_avg_fim.append(c_model_avg_fim_value)
            if args.fedavg_importance_wanda_param_wise or args.fedavg_importance_wanda_model_wise:
                c_model_wanda, c_model_avg_wanda_value = task_trainer.compute_wanda_information_matrix(c_model)
                c_model_wanda = strip_module_prefix(c_model_wanda)
                c_models_wanda_list.append(c_model_wanda)
                model_avg_wanda.append(c_model_avg_wanda_value)                


            accelerate_and_free_cache(accelerator)

            # record training information, e.g. loss
            record_training_information(args, logger, tensorboard_writer, loss_dict, task_key, inc_step, client_index, comm_round)
            
            # Store the model parameters for later weight averaging          
            c_model_dict = {}
            for n in c_model.state_dict().keys():
                if n in model.comm_state_dict_names:
                    c_model_dict[n] = c_model.state_dict()[n].data.to(device)
            c_models_list.append(c_model_dict)
            del task_trainer, c_model, temp_model

        accelerate_and_free_cache(accelerator)

        # --------- Average client models to get the global model ---------
        if args.fedavg_importance_ab_param_wise or args.fedavg_importance_ab_model_wise:
            print("------ fedavg_importance_ab ------------")
            c_models_importance_list, model_avg_importances = get_parameter_importance_by_absolution(c_models_list)
            print("model_avg_importances: {0}".format(model_avg_importances))
            if args.fedavg_importance_ab_param_wise:
                model = get_importance_weighted_net_parameter_wise(model, c_models_list, c_models_importance_list, device)
            elif args.fedavg_importance_ab_model_wise:
                model = get_average_net(model, c_models_list, model_avg_importances, device)
        elif args.fedavg_importance_fim_param_wise:
            print("------fedavg_importance_fim_param_wise -----------------")
            model = get_importance_weighted_net_parameter_wise(model, c_models_list, c_models_fim_list, device)
        elif args.fedavg_importance_fim_model_wise:
            print("------fedavg_importance_fim_model_wise -----------------")
            model = get_average_net(model, c_models_list, model_avg_fim, device)

        elif args.fedavg_importance_wanda_param_wise:
            print("------fedavg_importance_wanda_param_wise -----------------")
            print("model_avg_wanda: {0}".format(model_avg_wanda))
            model = get_importance_weighted_net_parameter_wise(model, c_models_list, c_models_wanda_list, device)
        elif args.fedavg_importance_wanda_model_wise:
            print("------fedavg_importance_wanda_model_wise -----------------")
            print("model_avg_wanda: {0}".format(model_avg_wanda))
            model = get_average_net(model, c_models_list, model_avg_wanda, device)
        elif args.fedavg_label_distribution:
                model = get_label_distribution_aware_average_net(model, c_models_list, client_weight, c_models_label_dis, device)
        elif args.fedavg_importance_ab_model_wise_w_label_distribution:
                c_models_importance_list, model_avg_importances = get_parameter_importance_by_absolution(c_models_list)
                print("model_avg_importances: {0}".format(model_avg_importances))
                model = get_label_distribution_aware_average_net(model, c_models_list, model_avg_importances, c_models_label_dis, device)
        else:
            model = get_average_net(model, c_models_list, client_weight, device)
        
        del c_models_list
        model.to(device)
        accelerate_and_free_cache(accelerator)

        # --------- perform evaluation in each communication round ---------
        eval_content, global_cls_avg_f1, result_dict, _, _, _, _ = perform_eval_for_federated_train(inc_step, comm_round, args, logger, tensorboard_writer, model, accelerator, device)
    
        if global_cls_avg_f1 > best_f1:
            best_f1 = global_cls_avg_f1
            model_save_path = os.path.join(task_output_dir, "fedavg_best-global-model_step-{}.pth".format(inc_step))
            torch.save(model.state_dict(), model_save_path)
            for client_index, client_result_dict in enumerate(result_dict):
                plot_radar_chart_figure(0, client_result_dict, client_output_dir_list[client_index], figure_name="best-global-model_client-{}".format(client_index + 1))
            logger.info("--- best class-average global F1 achived: {:.2f}% at round {} ---".format(best_f1 * 100, comm_round))

        if comm_round == (args.comm_rounds - 1):
            model_save_path = os.path.join(task_output_dir, "fedavg_final-model-{}-round_step-{}.pth".format(args.comm_rounds, inc_step))
            torch.save(model.state_dict(), model_save_path)
            for client_index, client_result_dict in enumerate(result_dict):
                plot_radar_chart_figure(0, client_result_dict, client_output_dir_list[client_index], figure_name="final-model_client-{}".format(client_index + 1))
    
    return eval_content
