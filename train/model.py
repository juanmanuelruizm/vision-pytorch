import torch

from utils.model_utils import classifier_layer, load_pretrained_model


class Model(torch.nn.Module):

    '''Clase base para definir el modelo a entrenar.'''

    def __init__(self, dir_pth: str, model_name: str, num_classes: int, loss_function: torch.nn.Module, emb_dim: int):

        super().__init__()
        # Definimos las capas que conformarán nuestro modelo
        ## Parte convolucional para extracción de características ##
        # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        ## Parte densa para la toma de desición por parte del modelo ##
        # self.fc1 = torch.nn.Linear(in_features=16 * 6 * 6, out_features=120)
        # self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        # self.fc3 = torch.nn.Linear(in_features=84, out_features=10)
        
        # Si se desea cargar un modelo preentrenado y añadirle una capa de clasificación
        # Instanciamos la función para cargar el modelo preentrenado
        self.base = load_pretrained_model(dir_pth, model_name)
        # Añadimos la capa de clasificación
        self.classifier = classifier_layer(emb_dim, num_classes)
        self.loss_function = loss_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Definimos la arquitectura del modelo.'''

        # En mi caso paso solo por base y clasificador porque la idea es  finetunear un modelo preentrenado.
        # Pasamos la entrada por la base del modelo
        x = self.base(x)
        # Pasamos la salida por la capa de clasificación
        x = self.classifier(x)
        return x