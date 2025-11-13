1. Visão geral do código
Esse código implementa um detector de veículos em imagens usando:

PyTorch + Torchvision → modelo de detecção de objetos (Faster R-CNN com ResNet50 e FPN).

PIL (Pillow) → para carregar a imagem e desenhar as bounding boxes.

Matplotlib → para exibir a imagem anotada.

O fluxo é:

Carrega o modelo pré-treinado.

Recebe uma imagem (PIL.Image).

Roda a detecção de objetos.

Filtra apenas veículos (carro, moto, caminhão, etc.).

Desenha as caixas e rótulos.

Retorna a imagem anotada e uma lista com as detecções.

2. Importações principais
import torch, torchvision
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
torch, torchvision: base do PyTorch; torchvision traz modelos de visão computacional já prontos.

to_tensor: converte uma imagem PIL em tensor PyTorch (formato que o modelo entende).

fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights:

Modelo de detecção de objetos (Faster R-CNN) com backbone ResNet50 e FPN.

Weights define os pesos pré-treinados (no dataset COCO).

PIL (Image, ImageDraw, ImageFont): trabalhar com imagens e desenhar retângulos e textos.

matplotlib.pyplot: para mostrar o resultado na tela.

numpy: manipulação de arrays, usado para converter os tensores de saída do modelo.
