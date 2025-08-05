import torch
import numpy as np


def get_column_sums(label_distributions):
    """
    Sums each column (label index) across a list of NumPy arrays.

    Args:
        label_distributions (list of np.ndarray): List of 1D arrays with label counts.

    Returns:
        np.ndarray: A 1D array with the total sum for each column/label.
    """
    total = np.sum(label_distributions, axis=0)
    return total


def get_label_distribution_aware_average_net(server, c_models_list, client_weight, c_models_label_dis, device, check_l2_distance=False, epsilon=1e-8):
    total_num_client = sum(client_weight)
    total_label_distribution = get_column_sums(c_models_label_dis)
    """
    print("get_label_distribution_aware_average_net | client_weight: {0}".format(client_weight))
    print("get_label_distribution_aware_average_net | total_num_client: {0}".format(total_num_client))
    for client_index, label_distribution in enumerate(c_models_label_dis):
        print("client_index: {0}".format(client_index))
        print("label_distribution: {0}".format(label_distribution))
    print("get_label_distribution_aware_average_net | total_label_distribution: {0}".format(total_label_distribution))
    """

    total_l2_distance = 0.0  # initialize total L2 distance
    num_models = len(c_models_list)  # number of client models
    
    with torch.no_grad():
        for key in server.comm_state_dict_names:
            temp = torch.zeros_like(server.state_dict()[key]).float().to(device)
            if "class_wise_MLP_layer" in key:
                # print("key: {0}".format(key))
                class_index = int(key.split(".")[-3])
                # rint("class_index: {0}".format(class_index))
                for net, label_dis in zip(c_models_list, c_models_label_dis):
                    temp += net[key] * label_dis[class_index] / total_label_distribution[class_index]
            else:
                for net, weight in zip(c_models_list, client_weight):  
                    temp += net[key] * weight / total_num_client
            server.state_dict()[key].data.copy_(temp)  # update server's state dict
                      
    # Calculate the L2 distance between each client model and the server model
    if check_l2_distance:
        for net in c_models_list:
            for key in server.comm_state_dict_names:
                if key not in net:
                    continue
                if not isinstance(net[key], torch.Tensor):
                    continue
                if net[key].shape != server.state_dict()[key].shape:
                    continue
                net_param = net[key].to(device)
                server_param = server.state_dict()[key].to(device)

                if not (torch.is_floating_point(net_param) and torch.is_floating_point(server_param)):
                    print("not floating point: {0}".format(key))
                    continue  # Only compute L2 distance for floating tensors
                    
                
                l2_distance = torch.norm(net_param - server_param, p=2)
                total_l2_distance += l2_distance.item()

        average_l2_distance = total_l2_distance / (num_models * len(server.comm_state_dict_names))
        print(f'Average L2 Distance between each model and the server model: {average_l2_distance}')

        total_norm = 0
        param_count = 0
        for param in server.parameters():
            if param.requires_grad and torch.is_floating_point(param):
                total_norm += param.norm(p=2).item()
                param_count += 1

        avg_weight_norm = total_norm / param_count
        print(f"Average weight L2 norm: {avg_weight_norm}")

    return server