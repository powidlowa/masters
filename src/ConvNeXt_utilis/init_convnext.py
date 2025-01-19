import logging
import requests
import torch
from torchvision import models

# loading pretrained ConvNeXT
def download_weights():
    # link to base weights
    url = "https://download.pytorch.org/models/convnext_base-6075fbad.pth"

    # download the weights
    weights_path = "models/ConvNeXt_weights/convnext_base.pth"
    response = requests.get(url)
    with open(weights_path, "wb") as f:
        f.write(response.content)
    logging.info(f'weights saved:{weights_path}')

    return weights_path

def init_convnext_model(weights_path):
    # load the model with the downloaded weights
    model = models.convnext_base()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    logging.info('model loaded')

    return model