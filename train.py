import tensorflow as tf
from model  import PatchMatchingModel
import tensorflow_addons as tfa
from data import PatchTrainDataset, PatchValDataset
import logging
logging.basicConfig(level=logging.DEBUG)
matching_model = PatchMatchingModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_fn = tfa.losses.ContrastiveLoss()

db_train = PatchTrainDataset()
db_val = PatchValDataset()

NUM_EPOCHS = 100
NUM_STEPS = 0
STEPS_VAL = 0
checkpoint = tf.train.Checkpoint(model=matching_model)
manager = tf.train.CheckpointManager(
    checkpoint, directory='./saved_models', max_to_keep=1)

@tf.function(jit_compile=True)
def train_step(patch_1, patch_2,labels):
    with tf.GradientTape() as tape:
        out = matching_model(patch_1, patch_2)
        loss = loss_fn(labels, out)
        validation_loss = tf.reduce_sum(matching_model.losses)
        total_loss = validation_loss + loss
    grads = tape.gradient(total_loss, matching_model.trainable_weights)
    grads = [tf.clip_by_value(x, -1.0, 1.0) for x in grads]
    optimizer.apply_gradients(zip(grads,matching_model.trainable_weights))
    return total_loss

@tf.function(jit_compile=True)
def val_step(patch_1, patch_2,labels):
    out = matching_model(patch_1, patch_2,training=False)
    loss = loss_fn(labels, out)
    validation_loss = tf.reduce_sum(matching_model.losses)
    total_loss = loss + validation_loss
    return total_loss


for epoch in range(NUM_EPOCHS):
    tf.keras.backend.set_learning_phase(1)
    for patch_1, patch_2, labels in db_train.db:
        total_loss = train_step(patch_1, patch_2, labels)
        NUM_STEPS+=1    
        if NUM_STEPS==1 or NUM_STEPS%10 == 0:
            logging.info('Training : Epoch = {}; steps = {}; Loss = {:.4f}'.format(epoch+1, NUM_STEPS,total_loss))
    manager.save()
    logging.info('Epoch {} training finished. Now doing validation.'.format(epoch+1))
    tf.keras.backend.set_learning_phase(0)
    for patch_1, patch_2, labels in db_val.db:
        val_step(patch_1,patch_2, labels)
        STEPS_VAL+=1
        if STEPS_VAL%10 == 0:
            logging.info('Validation : Epoch = {}; steps = {};  loss = {:.4f}'.format(epoch+1,STEPS_VAL, total_loss))

    logging.info('Validation finished. Starting epoch {}.'.format(epoch+2))
    STEPS_VAL = 0
            
