from train.visionlanguage_tasks.train_multi_label_cls import MultiLabelTrainer

SUPPORTED_VL_TASKS = [
    "openi_train",
    "mimic_train",
    "mnist_train",
    "cifar10_train",
    "voc2012_train",
    "isic2018_train",
    "xray14_train"
    ]

SUPPORTED_ORDER = ["Order1", "Order2", "Order3", "Order4", "Order5", "Debug_order"]

openi_train_config = {
    "task_name": "openi",
    "images_source": "openi",
    "splits": ["train", "val_small"],
    "classifier_type": "FC_Classifier",
    "num_labels": 15,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

mimic_train_config = {
    "task_name": "mimic",
    "images_source": "mimic",
    "splits": ["train", "val_small"],
    "classifier_type": "FC_Classifier",
    "num_labels": 15,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20, 
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

mnist_train_config = {
    "task_name": "mnist",
    "images_source": "mnist",
    "splits": ["train", "val_small"],
    "classifier_type": "FC_Classifier",
    "num_labels": 10,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

cifar10_train_config = {
    "task_name": "cifar10",
    "images_source": "cifar10",
    "splits": ["train", "val_small"],
    "classifier_type": "FC_Classifier",
    "num_labels": 10,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

voc2012_train_config = {
    "task_name": "voc2012",
    "images_source": "voc2012",
    "splits": ["train", "val"],
    "classifier_type": "FC_Classifier",
    "num_labels": 20,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

isic2018_train_config = {
    "task_name": "isic2018",
    "images_source": "isic2018",
    "splits": ["train", "val"],
    "classifier_type": "FC_Classifier",
    "num_labels": 7,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

xray14_train_config = {
    "task_name": "xray14",
    "images_source": "xray14",
    "splits": ["train", "val"],
    "classifier_type": "FC_Classifier",
    "num_labels": 15,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

medmnist_train_config = {
    "task_name": "medmnist",
    "images_source": "medmnist",
    "splits": ["train", "val_small"],
    "classifier_type": "FC_Classifier",
    "num_labels": 10,
    "num_images": 1,
    "model_type": "classification",
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "task_trainer": MultiLabelTrainer,
    "random_baseline_score": 0.0,
}

task_configs = {
    "openi_train": openi_train_config,
    "mimic_train": mimic_train_config,
    "mnist_train": mnist_train_config,
    "cifar10_train": cifar10_train_config,
    "medmnist_train": medmnist_train_config,
    "voc2012_train": voc2012_train_config,
    "isic2018_train": isic2018_train_config,
    "xray14_train": xray14_train_config
}
