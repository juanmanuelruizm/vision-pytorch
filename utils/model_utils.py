import torch


def classifier_layer(input_dim: int, output_dim: int) -> torch.nn.Linear:
    '''Capa densa para clasificación.'''
    class_layer = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
    return class_layer

def load_pretrained_model(model: str, dir_pth: str, model_name: str) -> torch.nn.Module:
    '''Carga un modelo preentrenado desde Torch Hub.'''

    if model == 'dinov3':
        # La idea es cargarlo en local pero si no dispones del fichero .pth (o .pt) se puede descargar desde internet
        base_model = torch.hub.load(dir_pth, model_name, pretrained=True, source='local')
        return base_model