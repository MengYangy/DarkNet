import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import*
from tensorflow.keras.models import Model

def darknetConv2d_BN_Leakr(x, f_size, k_size, strides=1):
    x = Conv2D(filters=f_size,
               kernel_size=k_size,
               strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def res_unit(x, f_size, k_size):
    x1 = x
    x = darknetConv2d_BN_Leakr(x, f_size // 2, k_size)
    x = darknetConv2d_BN_Leakr(x, f_size, k_size)
    return Add()([x, x1])


def resn(x, f_size, k_size=3, n=1, strides=2):
    x = darknetConv2d_BN_Leakr(x, f_size, k_size, strides=strides)
    for i in range(n):
        x = res_unit(x, f_size, k_size)
    return x


def daeknet_body(inputs=(416,416,3), include_fc=True, classes=1000):
    input_tensor = Input(inputs)
    x = darknetConv2d_BN_Leakr(input_tensor, f_size=32, k_size=3)
    x = resn(x, f_size=64, n=1)
    x = resn(x, f_size=128, n=2)
    x = resn(x, f_size=256, n=8)
    x = resn(x, f_size=512, n=8)
    x = resn(x, f_size=1024, n=4)

    if include_fc:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax')(x)
    return Model(input_tensor, x)

if __name__ == '__main__':
    model = daeknet_body()
    model.summary()