import logging
import torch
from datetime import datetime

def return_feature_extractor(model):

    # remove final layer of the model; model.children() aka layers
    layers_without_classifier = list(model.children())[:-1]
    # unpack layers and init feature exractor
    feature_extractor = torch.nn.Sequential(*layers_without_classifier)
    # set to evaluation mode
    feature_extractor.eval()
    logging.info(f'{datetime.now()}: feature extractor initialised')

    return feature_extractor
