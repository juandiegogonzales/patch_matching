import tensorflow as tf
from model  import PatchMatchingModel
import tensorflow_addons as tfa
from data import PatchTrainDataset
import logging
logging.basicConfig(level=logging.DEBUG)
matching_model = PatchMatchingModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
loss_fn = tfa.losses.ContrastiveLoss()

db = PatchTrainDataset()

NUM_EPOCHS = 100
NUM_STEPS = 0

for epoch in range(NUM_EPOCHS):
    for patch_1, patch_2, labels in db.db:
        with tf.GradientTape() as tape:
            out = matching_model(patch_1, patch_2)
            loss = loss_fn(labels,out)
        grads = tape.gradient(loss, matching_model.trainable_weights)
        grads = [tf.clip_by_value(x, -1.0, 1.0) for x in grads]
        optimizer.apply_gradients(zip(grads,matching_model.trainable_weights))
        NUM_STEPS+=1
        if NUM_STEPS==1 or NUM_STEPS%100 == 0:
            logging.info('Epoch = {}; Loss = {:.4f}'.format(epoch+1, loss))
            
