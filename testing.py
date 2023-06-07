import tensorflow as tf
from keras.models import load_model

from module.dataset import DATASET_PATH, LABELS_PATH, TRAIN_PATH, TEST_PATH, csv_to_dict, AirbusDataset
from module.dice import dice_loss, dice_coeff
from module.transform import normalize_rescale_blur

import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", type=str, default="./Models/ModelS.h5", help="model name")
	parser.add_argument("--csv", type=str, default=f"{LABELS_PATH}")
	parser.add_argument("--img_dir", type=str, default=f"{TRAIN_PATH}")
	parser.add_argument("--batch", type=int, default=100)
	parser.add_argument("--limit", type=int, default=-1)
	args = parser.parse_args()

	csv_path = args.csv
	img_dir = args.img_dir
	name = args.name
	batch = args.batch
	limit = args.limit

	model = load_model(name, custom_objects={"dice_loss":dice_loss})
	dataset = AirbusDataset(csv_path, img_dir, normalize_rescale_blur)

	if limit == -1:
		limit = len(dataset)

	losses = []
	for i in range(0, limit, batch): # evaluate on all data. provide your own indexes
		images, masks = dataset[list(range(i,i+batch))].values()
		prediction = model.predict(images)
		losses.append(dice_loss(tf.constant(masks, tf.float32), prediction))
	
	print("dice_loss", tf.math.reduce_mean(losses))

