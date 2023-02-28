import tensorflow as tf
from model import PatchMatchingModel
import matplotlib.pyplot as plt
from data import PatchTestDataset
import logging

db_test = PatchTestDataset('./data/Bing.png', './data/Yandex.png')
logging.basicConfig(level=logging.DEBUG)
matching_model = PatchMatchingModel()

checkpoint = tf.train.Checkpoint(model=matching_model)

manager = tf.train.CheckpointManager(checkpoint, directory='./saved_models', max_to_keep=1)

for patch, rows, cols in db_test.db:
    patch_1 = patch[0]
    patch_2 = patch[1]
    out = matching_model(patch_1, patch_2, training=False)
    print(out)



