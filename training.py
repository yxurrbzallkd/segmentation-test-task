import numpy as np
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
import cv2

from module.transform import normalize_rescale_blur, means, stds
from module.dataset import AirbusDataset
from module.dice import dice_coeff, dice_loss
from module.unet import make_model

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="model name")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--checkpoint", type=int, default=360, help="interval between checkpoints (seconds)")
    parser.add_argument("--data", type=str, default="./jsons/train_data_ships.json")
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()

    dataset = AirbusDataset(data_path=args.json)
    modelname = args.name
    checkpoint = args.checkpoint # 10 mins
    iterations = args.iterations
    batch = args.batch      # batch size
    
    model = make_model(768//2,768//2,3)
    transform = normalize_rescale_blur

    counter = 0     # how many checlpoints saved so far
    for iteration in range(iterations):
        start = time()
        for i in range(0, len(dataset)//2-batch-1, batch):
            # first half of the dataset is for training
            images, masks = dataset[list(range(i,i+batch))].values()
            images, masks = transform(images, masks)

            # second half of the dataset is for validation
            vali, valm = dataset[list(range(len(dataset)//2+i,len(dataset)//2+i+batch))].values() 
            vali, valm = transform(vali, valm)
            
            model.fit(images, masks, epochs=5, validation_data=(vali,valm), batch_size=10)
            
            if time() - start > 600: # checkpoint
                model.save(f"{modelname}-{counter}.h5")
                counter += 1
                start = time()

        # checkpoint after going over all the training data
        model.save(f"{modelname}-{counter}.h5")
        counter += 1

    # save the finished model
    model.save(f"{modelname}.h5")
