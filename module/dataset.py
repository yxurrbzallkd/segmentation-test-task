import pandas as pd
import numpy as np
from module.make_mask import ships_to_mask
from PIL import Image
import json
import os
import pandas as pd
import numpy as np

DATASET_PATH="C:/Users/Diana/Documents/airbus-ship-detection"
LABELS_PATH = f"{DATASET_PATH}/train_ship_segmentations_v2.csv"
TRAIN_PATH=f"{DATASET_PATH}/train_v2"
TEST_PATH =f"{DATASET_PATH}/test_v2"

def csv_to_dict(csv):
    """
    in the .csv each row corresponds to one ship on one image in run-length encoding
    gather all ships for each image (0 if there are none) into a dictionary
    purpose:
        easier to make masks
    """
    df = pd.read_csv(csv)
    data = dict()
    for i in df.values:
        if i[0] not in data:
            data[i[0]] = dict()
        if isinstance(i[1], str): # if there is a ship and not a NaN
            data[i[0]][len(data[i[0]])] = list(map(int, i[1].split()))
    imgs = list(data.keys())
    return imgs, data


class AirbusDataset():
    def __init__(self, data_path="./train_data_v2.json", root_dir=TRAIN_PATH, transform=None):
        """A Cnvenient Airbus Dataset loader
        
        Args:
            data_path  (str):   path to csv file with labels or to a json file obtained with csv_to_dict
            root_dir (string):  directory with the images.
            transform (callable, optional): optional transform to apply to images
                                            a function that accepts 2 arguments: images and masks
                                            transforms them and return transformed results
        """
        super(AirbusDataset, self).__init__()
        
        if data_path.endswith(".json"):
            with open(data_path) as f: self.dataset = json.load(f)
            self.images = list(self.dataset.keys())
        
        elif data_path.endswith(".csv"):
            self.images, self.dataset = csv_to_dict(data_path)
        
        else:
            raise ValueError(f"filetype of {data_path} not accepted - provide .csv or .json")
        
        self.root_dir = root_dir
        self.ids = list(range(len(self.dataset)))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list): # return multiple images
            img_paths = [os.path.join(self.root_dir,
                                      self.images[i]) for i in idx]
            image = [np.array(Image.open(ip)) for ip in img_paths]
            mask = [ships_to_mask(self.dataset[self.images[i]]) for i in idx]
        
        else: # return one image
            img_path = os.path.join(self.root_dir,
                                    self.images[idx])
            image = np.array(Image.open(img_path))
            mask = ships_to_mask(self.dataset[self.images[idx]])
        
        if self.transform:
            image,mask = self.transform(image,mask)

        return {'image': image, 'mask': mask}
