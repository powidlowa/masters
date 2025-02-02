import logging
import requests
import torch
from torchvision import models
from datetime import datetime

# loading pretrained ConvNeXT
def download_weights(is_wieghts_file_available):

    weights_path = "models/ConvNeXt_weights/convnext_base.pth"
    
    if not is_wieghts_file_available:
        # link to base weights
        url = "https://download.pytorch.org/models/convnext_base-6075fbad.pth"

        # download the weights
        response = requests.get(url)
        with open(weights_path, "wb") as f:
            f.write(response.content)
        logging.info(f'{datetime.now()}: weights saved: {weights_path}')
        
    return weights_path

def init_convnext_model(weights_path):
    # load the model with the downloaded weights
    model = models.convnext_base()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    logging.info(f'{datetime.now()}: model loaded')

    return model