import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 num_filters: int,
                 l2_reg: float = 1E-5
                 ):
        super(ConvBlock, self).__init__()
        self._conv = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(l2=l2_reg)
        )
        self._bnorm = tf.keras.layers.BatchNormalization()

    def call(self, x):
        out = self._conv(x)
        out = self._bnorm(out)
        return out


def get_branch(l2_reg: float = 1E-5):
    model = tf.keras.Sequential(
        [
            ConvBlock(32, l2_reg),
            ConvBlock(64, l2_reg),
            tf.keras.layers.MaxPool2D(
                pool_size=2),
            ConvBlock(64, l2_reg),
            ConvBlock(64, l2_reg),
            tf.keras.layers.MaxPool2D(
                pool_size=2),
            ConvBlock(128, l2_reg),
            ConvBlock(128, l2_reg),
            tf.keras.layers.MaxPool2D(
                pool_size=3),
            ConvBlock(128, l2_reg),
            tf.keras.layers.Flatten()
        ]
    )
    return model


class PatchMatchingModel(tf.keras.Model):
    def __init__(self, l2_reg: float = 1E-5):
        super(PatchMatchingModel, self).__init__()
        self._branch = get_branch(l2_reg=l2_reg)
        self._subtract = tf.keras.layers.Subtract()

    def call(self, patch1, patch2):
        P1 = self._branch(patch1)
        P2 = self._branch(patch2)
        out = self._subtract([P1, P2])
        out = tf.norm(out)
        return out


if __name__ == "__main__":
    model = PatchMatchingModel()
    inp1 = tf.random.uniform(shape=(1, 512, 512, 3))
    inp2 = tf.random.uniform(shape=(1, 512, 512, 3))
    out = model(inp1, inp2)
    print(out)
