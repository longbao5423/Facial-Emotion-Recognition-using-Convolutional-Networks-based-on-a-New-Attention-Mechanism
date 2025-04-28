import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, AlphaDropout, Activation, GlobalAveragePooling2D


class MyModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()

        self.conv2d1_1 = Conv2D(32, 3, strides=1, padding='same', activation='relu')
        self.bn1_1 = BatchNormalization()
        self.conv2d1_2 = Conv2D(64, 7, strides=1, padding='same', activation='relu')
        self.bn1_2 = BatchNormalization()
        self.max_pool1_1 = MaxPooling2D()

        self.conv2d2_1 = Conv2D(64, 3, strides=1, padding='same', activation='relu')
        self.bn2_1 = BatchNormalization()
        self.conv2d2_2 = Conv2D(64, 3, strides=1, padding='same', activation='relu')
        self.bn2_2 = BatchNormalization()
        self.max_pool2_1 = MaxPooling2D()

        self.conv2d3_1 = Conv2D(128, 3, strides=1, padding='same', activation='relu')
        self.bn3_1 = BatchNormalization()
        self.conv2d3_2 = Conv2D(128, 3, strides=1, padding='same', activation='relu')
        self.bn3_2 = BatchNormalization()
        self.max_pool3_1 = MaxPooling2D()

        self.conv2d4_1 = Conv2D(256, 3, strides=1, padding='same', activation='relu')
        self.bn4_1 = BatchNormalization()
        self.conv2d4_2 = Conv2D(256, 3, strides=1, padding='same', activation='relu')
        self.bn4_2 = BatchNormalization()
        self.max_pool4_1 = MaxPooling2D()

        self.flatten1 = Flatten()

        self.dense1 = Dense(512, activation='relu')
        self.bn6_1 = BatchNormalization()
        self.drop6_1 = Dropout(0.5)
        self.dense2 = Dense(256, activation='relu')
        self.bn6_2 = BatchNormalization()
        self.drop6_2 = Dropout(0.1)
        self.dense3 = Dense(7, activation='softmax')
        
    def call(self, inputs, training=None, mask=None):
        x1_1 = self.conv2d1_1(inputs)
        x1_1 = self.bn1_1(x1_1)
        x1_2 = self.conv2d1_2(x1_1)
        x1_2 = self.bn1_2(x1_2)
        x1 = self.max_pool1_1(x1_2)

        x2_1 = self.conv2d2_1(x1)
        x2_1 = self.bn2_1(x2_1)
        x2_2 = self.conv2d2_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x2_2 = self.max_pool2_1(x2_2)


        x3_1 = self.conv2d3_1(x2_2)
        x3_1 = self.bn3_1(x3_1)
        x3_2 = self.conv2d3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)
        x3_2 = self.max_pool3_1(x3_2)


        x4_1 = self.conv2d4_1(x3_2)
        x4_1 = self.bn4_1(x4_1)
        x4_2 = self.conv2d4_2(x4_1)
        x4_2 = self.bn4_2(x4_2)
        x4_2 = self.max_pool4_1(x4_2)


        x = self.flatten1(x4_2)
        x = self.dense1(x)
        x = self.bn6_1(x)
        x = self.drop6_1(x)
        x = self.dense2(x)
        x = self.bn6_2(x)
        x = self.drop6_2(x)
        x = self.dense3(x)
        return x
        

if __name__ == '__main__':
    model = MyModel()
    model.build(input_shape=(None, 48, 48, 1))
    model.summary()
