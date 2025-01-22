import logging
from torchvision import transforms
from datetime import datetime

def preprocess_image():
    # prepare images for embeddings creation
    preprocesser = transforms.Compose([ 
        transforms.Resize(256), # crop the shortes side to 256 pixels ( pixels margin)
        transforms.CenterCrop(224), # center and cop image to 224x224 size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # average pixel values for Red, Green, and Blue channels in the ImageNet dataset
            std=[0.229, 0.224, 0.225]), # e standard deviation of pixel values for each channel in ImageNet
        ])
    logging.info(f'{datetime.now()}: preprocesser initialised')
    return preprocesser