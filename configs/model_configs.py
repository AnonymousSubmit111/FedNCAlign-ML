from modeling.resnet.resnet import convert_batch_to_resnet_input_dict
from modeling.resnet.resnet_encoder import ResNetEncoderWrapper
from modeling.vit.vit import convert_batch_to_vit_input_dict
from modeling.vit.vit_encoder import ViTEncoderWrapper

ALLOWED_CL_ENCODERS = ["resnet18", "resnet50", "vit-t-16", "vit-b-16"]


#### for ResNet ####
resnet18_config = {
    'encoder_dim': 512,
    'visual_input_type': 'pil-image',
    'encoder_class': ResNetEncoderWrapper,  # Assumes you've imported or defined it
    'batch2inputs_converter': convert_batch_to_resnet_input_dict,  # You need to define this
    'encoder_name': 'ResNet'
}

resnet50_config = {
    'encoder_dim': 2048,
    'visual_input_type': 'pil-image',
    'encoder_class': ResNetEncoderWrapper,  # Assumes you've imported or defined it
    'batch2inputs_converter': convert_batch_to_resnet_input_dict,  # You need to define this
    'encoder_name': 'ResNet'
}

#### for ViT ####
vit_t_16_config = {
    'encoder_dim': 192,
    'visual_input_type': 'pil-image',
    'encoder_class': ViTEncoderWrapper,  # Assumes you've imported or defined it
    'batch2inputs_converter': convert_batch_to_vit_input_dict,  # You need to define this
    'encoder_name': 'ResNet'
}

vit_b_16_config = {
    'encoder_dim': 768,
    'visual_input_type': 'pil-image',
    'encoder_class': ViTEncoderWrapper,  # Assumes you've imported or defined it
    'batch2inputs_converter': convert_batch_to_vit_input_dict,  # You need to define this
    'encoder_name': 'ResNet'
}

model_configs = {
    "resnet18": resnet18_config,
    "resnet50": resnet50_config,
    "vit-t-16": vit_t_16_config,
    "vit-b-16": vit_b_16_config
}


