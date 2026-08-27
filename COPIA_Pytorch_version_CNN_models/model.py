import torch
from timm import create_model
import torchvision.models as models



MODEL_NAMES = [
    'vit_base_patch16_224',
    'deit_base_distilled_patch16_224',
    'levit_256',
    'pit_b_224',
    'cait_s24_224',
    'convit_base',
    'tnt_s_patch16_224',
    'visformer_small',
    'vit_large_patch16_224',
    'inception_v3',
    'inception_v4',
    'inception_resnet_v2',
    'resnet50',
    'resnet152',
]

def get_model(model_name):

    CNN_299_MODELS = ['inception_v3', 'inception_v4', 'inception_resnet_v2']
    CNN_224_MODELS = ['resnet50', 'resnet152']

    if model_name in MODEL_NAMES and model_name not in CNN_299_MODELS and model_name not in CNN_224_MODELS:
        model = create_model(model_name, pretrained=True, num_classes=1000, in_chans=3, global_pool=None, scriptable=False)
        model.input_size = 224

    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True, aux_logits=False)
        model.input_size = 299
        model = InceptionWrapper(model)
    elif model_name == 'inception_v4' or model_name == 'inception_resnet_v2':
        model = create_model(model_name, pretrained=True, num_classes=1000)
        model.input_size = 299
        model = InceptionWrapper(model)

    elif model_name == 'resnet50':
        base_model = models.resnet50(pretrained=True)
        base_model.input_size = 224
        model = InceptionWrapper(base_model)

    elif model_name == 'resnet152':
        base_model = models.resnet152(pretrained=True)
        base_model.input_size = 224
        model = InceptionWrapper(base_model)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    print(f'Loaded model: {model_name} (Input Size: {model.input_size})')
    return model


class InceptionWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

        if hasattr(model, 'input_size'):
            self.input_size = model.input_size

    def forward(self, x):

        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out