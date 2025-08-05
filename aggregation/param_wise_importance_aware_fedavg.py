import torch


def get_parameter_importance_by_absolution(c_models):
    """
    Calculate parameter importance using the absolute value of each parameter.

    Args:
        c_models: list of client models (nn.Module or state_dicts)

    Returns:
        c_importances: list of dicts (same structure as model state_dict)
        model_avg_importances: list of floats (average absolute weight per model)
    """
    c_importances = []
    model_avg_importances = []

    for model in c_models:
        # Get state_dict
        state_dict = model.state_dict() if isinstance(model, torch.nn.Module) else model
        importance = {}
        total_abs = 0.0
        count = 0

        for name, param in state_dict.items():
            if torch.is_floating_point(param):
                abs_param = param.abs()
                importance[name] = abs_param
                total_abs += abs_param.sum().item()
                count += param.numel()
            else:
                importance[name] = torch.ones_like(param, dtype=torch.float32)

        avg_importance = total_abs / count if count > 0 else 0.0
        model_avg_importances.append(avg_importance)
        c_importances.append(importance)

    return c_importances, model_avg_importances


def get_importance_weighted_net_parameter_wise(server, c_models, c_importances, device, check_l2_distance=False, epsilon=1e-8):
    """
    Performs importance-weighted federated averaging with equal averaging for 'clf_layer.etf_classifier.weight'.
    
    Args:
        server: the global model (with .state_dict())
        c_models: list of client model state_dicts
        c_importances: list of client parameter-wise importance dicts (same structure as c_models)
        device: torch.device
        check_l2_distance: bool, whether to print L2 distance diagnostics
        epsilon: small value to avoid division by zero

    Returns:
        server: updated global model
    """
    total_l2_distance = 0.0
    num_models = len(c_models)

    with torch.no_grad():
        for key in server.comm_state_dict_names:
            if key == "clf_layer.etf_classifier.weight":
                # Equal averaging
                sum_param = torch.zeros_like(server.state_dict()[key]).float().to(device)
                count = 0
                for c_model in c_models:
                    if key not in c_model:
                        continue
                    sum_param += c_model[key].to(device)
                    count += 1
                if count > 0:
                    avg_param = sum_param / count
                    server.state_dict()[key].data.copy_(avg_param)
            else:
                # Importance-weighted averaging
                weighted_sum = torch.zeros_like(server.state_dict()[key]).float().to(device)
                importance_sum = torch.zeros_like(server.state_dict()[key]).float().to(device)

                for c_model, c_importance in zip(c_models, c_importances):
                    if key not in c_model or key not in c_importance:
                        continue
                    param = c_model[key].to(device)
                    importance = c_importance[key].to(device)
                    weighted_sum += param * importance
                    importance_sum += importance

                avg_param = weighted_sum / (importance_sum + epsilon)
                server.state_dict()[key].data.copy_(avg_param)

    # Optional: Diagnostic for average L2 distance
    if check_l2_distance:
        for net in c_models:
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
                    print(f"not floating point: {key}")
                    continue

                l2_distance = torch.norm(net_param - server_param, p=2)
                total_l2_distance += l2_distance.item()

        average_l2_distance = total_l2_distance / (num_models * len(server.comm_state_dict_names))
        print(f'Average L2 Distance between each model and the server model: {average_l2_distance:.4f}')

        total_norm = 0
        param_count = 0
        for param in server.parameters():
            if param.requires_grad and torch.is_floating_point(param):
                total_norm += param.norm(p=2).item()
                param_count += 1

        avg_weight_norm = total_norm / param_count
        print(f"Average weight L2 norm: {avg_weight_norm:.4f}")

    return server
