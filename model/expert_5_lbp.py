import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Concatenate, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, AlphaDropout, Activation, GlobalAveragePooling2D


class SEBlock(tf.keras.models.Model):
    def __init__(self, filters, ratio=16):
        super().__init__()
        self.avg = GlobalAveragePooling2D(keepdims=True)
        self.dense_1 = Dense(filters // ratio, activation='relu')
        self.dense_2 = Dense(filters, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.avg(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
    

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
        self.seblock_2 = SEBlock(64, 16)
        self.max_pool2_1 = MaxPooling2D()
        self.drop2_1 = Dropout(0.3)

        self.conv2d3_1 = Conv2D(128, 3, strides=1, padding='same', activation='relu')
        self.bn3_1 = BatchNormalization()
        self.conv2d3_2 = Conv2D(128, 3, strides=1, padding='same', activation='relu')
        self.bn3_2 = BatchNormalization()
        self.seblock_3 = SEBlock(128, 16)
        self.max_pool3_1 = MaxPooling2D()
        self.drop3_1 = Dropout(0.3)

        self.conv2d4_1 = Conv2D(256, 3, strides=1, padding='same', activation='relu')
        self.bn4_1 = BatchNormalization()
        self.conv2d4_2 = Conv2D(256, 3, strides=1, padding='same', activation='relu')
        self.bn4_2 = BatchNormalization()
        self.seblock_4 = SEBlock(256, 16)
        self.max_pool4_1 = MaxPooling2D()
        self.drop4_1 = Dropout(0.3)

        self.flatten1 = Flatten()

        # lbp特征处理
        self.lbp_conv2d1_1 = Conv2D(32, 3, strides=1, padding='same', activation='relu')
        self.lbp_bn1_1 = BatchNormalization()
        self.lbp_conv2d1_2 = Conv2D(64, 7, strides=1, padding='same', activation='relu')
        self.lbp_bn1_2 = BatchNormalization()
        self.lbp_max_pool1_1 = MaxPooling2D()

        self.lbp_conv2d2_1 = Conv2D(64, 3, strides=1, padding='same', activation='relu')
        self.lbp_bn2_1 = BatchNormalization()
        self.lbp_conv2d2_2 = Conv2D(64, 3, strides=1, padding='same', activation='relu')
        self.lbp_bn2_2 = BatchNormalization()
        self.lbp_seblock_2 = SEBlock(64, 16)
        self.lbp_max_pool2_1 = MaxPooling2D()
        self.lbp_drop2_1 = Dropout(0.3)

        self.lbp_conv2d3_1 = Conv2D(128, 3, strides=1, padding='same', activation='relu')
        self.lbp_bn3_1 = BatchNormalization()
        self.lbp_conv2d3_2 = Conv2D(128, 3, strides=1, padding='same', activation='relu')
        self.lbp_bn3_2 = BatchNormalization()
        self.lbp_seblock_3 = SEBlock(128, 16)
        self.lbp_max_pool3_1 = MaxPooling2D()
        self.lbp_drop3_1 = Dropout(0.3)

        self.lbp_conv2d4_1 = Conv2D(256, 3, strides=1, padding='same', activation='relu')
        self.lbp_bn4_1 = BatchNormalization()
        self.lbp_conv2d4_2 = Conv2D(256, 3, strides=1, padding='same', activation='relu')
        self.lbp_bn4_2 = BatchNormalization()
        self.lbp_seblock_4 = SEBlock(256, 16)
        self.lbp_max_pool4_1 = MaxPooling2D()
        self.lbp_drop4_1 = Dropout(0.3)

        self.lbp_flatten1 = Flatten()

        self.concat_lbp = Concatenate(axis=-1)

        self.seq_out = Sequential(
            layers=[
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.1),
                Dense(7, activation='softmax')
            ]
        )
        
    def call(self, inputs, training=None, mask=None):
        x1_1 = self.conv2d1_1(inputs[0])
        x1_1 = self.bn1_1(x1_1)
        x1_2 = self.conv2d1_2(x1_1)
        x1_2 = self.bn1_2(x1_2)
        x1 = self.max_pool1_1(x1_2)

        x2_1 = self.conv2d2_1(x1)
        x2_1 = self.bn2_1(x2_1)
        x2_2 = self.conv2d2_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x2_2 = self.seblock_2(x2_2) * x2_2
        x2_2 = self.max_pool2_1(x2_2)
        x2_2 = self.drop2_1(x2_2)


        x3_1 = self.conv2d3_1(x2_2)
        x3_1 = self.bn3_1(x3_1)
        x3_2 = self.conv2d3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)
        x3_2 = self.seblock_3(x3_2) * x3_2
        x3_2 = self.max_pool3_1(x3_2)
        x3_2 = self.drop3_1(x3_2)


        x4_1 = self.conv2d4_1(x3_2)
        x4_1 = self.bn4_1(x4_1)
        x4_2 = self.conv2d4_2(x4_1)
        x4_2 = self.bn4_2(x4_2)
        x4_2 = self.seblock_4(x4_2) * x4_2
        x4_2 = self.max_pool4_1(x4_2)
        x4_2 = self.drop4_1(x4_2)


        x = self.flatten1(x4_2)

        # 加入lbp手工特征
        lbp_x1_1 = self.lbp_conv2d1_1(inputs[1])
        lbp_x1_1 = self.lbp_bn1_1(lbp_x1_1)
        lbp_x1_2 = self.lbp_conv2d1_2(lbp_x1_1)
        lbp_x1_2 = self.lbp_bn1_2(lbp_x1_2)
        lbp_x1 = self.lbp_max_pool1_1(lbp_x1_2)

        lbp_x2_1 = self.lbp_conv2d2_1(lbp_x1)
        lbp_x2_1 = self.lbp_bn2_1(lbp_x2_1)
        lbp_x2_2 = self.lbp_conv2d2_2(lbp_x2_1)
        lbp_x2_2 = self.lbp_bn2_2(lbp_x2_2)
        lbp_x2_2 = self.lbp_seblock_2(lbp_x2_2) * lbp_x2_2
        lbp_x2_2 = self.lbp_max_pool2_1(lbp_x2_2)
        lbp_x2_2 = self.lbp_drop2_1(lbp_x2_2)

        lbp_x3_1 = self.lbp_conv2d3_1(lbp_x2_2)
        lbp_x3_1 = self.lbp_bn3_1(lbp_x3_1)
        lbp_x3_2 = self.lbp_conv2d3_2(lbp_x3_1)
        lbp_x3_2 = self.lbp_bn3_2(lbp_x3_2)
        lbp_x3_2 = self.lbp_seblock_3(lbp_x3_2) * lbp_x3_2
        lbp_x3_2 = self.lbp_max_pool3_1(lbp_x3_2)
        lbp_x3_2 = self.lbp_drop3_1(lbp_x3_2)

        lbp_x4_1 = self.lbp_conv2d4_1(lbp_x3_2)
        lbp_x4_1 = self.lbp_bn4_1(lbp_x4_1)
        lbp_x4_2 = self.lbp_conv2d4_2(lbp_x4_1)
        lbp_x4_2 = self.lbp_bn4_2(lbp_x4_2)
        lbp_x4_2 = self.lbp_seblock_4(lbp_x4_2) * lbp_x4_2
        lbp_x4_2 = self.lbp_max_pool4_1(lbp_x4_2)
        lbp_x4_2 = self.lbp_drop4_1(lbp_x4_2)

        lbp_x = self.lbp_flatten1(lbp_x4_2)

        lbp_x = self.concat_lbp([x, lbp_x])

        out = self.seq_out(lbp_x)
        return out


if __name__ == '__main__':
    model = MyModel()
    model.build(input_shape=[(None, 48, 48, 1), (None, 48, 48, 1)])
    model.summary()
