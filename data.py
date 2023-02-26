from itertools import combinations_with_replacement
from typing import Literal, List
import tensorflow_io as tfio
import tensorflow as tf
import tensorflow_probability as tfp
import augmentation



class PatchTrainDataset(object):
    DIST = tfp.distributions.Categorical(probs=[0.5,0.5])
    def __init__(self,
                 batch_size : int = 8
                 ):
        self._db = tf.data.Dataset.from_tensor_slices(self._train_files())
        self._db = self._db.map(self._read_fn)
        self._db = self._db.repeat(100)
        self._db = self._db.cache()
        self._db = self._db.map(self._extract_patch)
        self._db = self._db.batch(batch_size)
        self._db = self._db.prefetch(100)
        

    @property
    def db(self):
        return self._db


    def _train_files(self):
        files = [
            './data/Bing.png',
            './data/Yandex.png',
            './data/ESRI.png'
            ]
        files = list(
            combinations_with_replacement(files,2)
            )
        return files

    @staticmethod
    def _read_fn(filenames : List[str]):
        image_1 = tf.io.read_file(filenames[0])
        image_1 = tf.io.decode_image(image_1)
        #image_1 = tfio.experimental.image.decode_image(image_1)
        image_1_shape = tf.shape(image_1)
        image_1 = tf.slice(image_1,[0,0,0],[image_1_shape[0],image_1_shape[1],3])
        
        image_2 = tf.io.read_file(filenames[1])
        image_2 = tf.io.decode_image(image_2)
        #image_2 = tfio.experimental.image.decode_tiff(image_2)
        image_2_shape = tf.shape(image_2)
        image_2 = tf.slice(image_2, [0,0,0],[image_2_shape[0],image_2_shape[1],3])
        return image_1, image_2


    def _extract_patch(self,image_1, image_2):
        boundary_1 = self._get_patch_boundary()
        keep_same = self.DIST.sample()
        boundary_2 = tf.cond(
            tf.equal(keep_same, 1),
            lambda: boundary_1,
            lambda : self._get_patch_boundary())
        patch_1 = tf.slice(image_1,[boundary_1[0],boundary_1[1],0],
                           [512,512,3])
        patch_2 = tf.slice(image_2,[boundary_2[0],boundary_2[1],0],
                           [512,512,3]
                           )
        out = augmentation.flip_left_right(patch_1, patch_2)
        out = augmentation.normalize_patches(out[0],out[1])
        label = tf.cond(
            tf.equal(keep_same,1),
            lambda : 1,
            lambda : 0
            )
        return out[0], out[1], label

    @staticmethod
    def _get_patch_boundary():
        col_start = tf.random.uniform(shape=(), maxval=5604-512,dtype=tf.int32)
        row_start = tf.random.uniform(shape=(), maxval=3509-512,dtype=tf.int32)
        return row_start, col_start






if __name__ == "__main__":
    dataset = PatchTrainDataset()
    db = dataset.db
    for i in db:
        print(i)
