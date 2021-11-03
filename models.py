import tensorflow as tf
from tensorflow import keras 

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D


channels = 3
lr_height = 64  # Low resolution height
lr_width = 64  # Low resolution width
lr_shape = (lr_height, lr_width, channels)
hr_height = lr_height * 4  # High resolution height
hr_width = lr_width * 4  # High resolution width
hr_shape = (hr_height, hr_width, channels)

# Number of residual blocks in the generator
n_residual_blocks = 16


def build_vgg():
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    vgg19 = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=hr_shape))
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
        
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block3_conv4').output)
    model.trainable = False
    
    return model


def build_generator(gf=64, layer_half=None):
    def residual_block(layer_input, filters):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d

    def deconv2d(layer_input):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u

    # Low resolution image input
    img_lr = Input(shape=lr_shape)

    # Pre-residual block
    c1 = Conv2D(gf, kernel_size=9, strides=1, padding='same')(img_lr)
    c1 = Activation('relu')(c1)

    # Propogate through residual blocks
    num_residual_blocks = n_residual_blocks
    if layer_half:
        num_residual_blocks = int(n_residual_blocks / 2)
        
    r = residual_block(c1, gf)
    for _ in range(num_residual_blocks - 1):
        r = residual_block(r, gf)

    # Post-residual block
    c2 = Conv2D(gf, kernel_size=3, strides=1, padding='same')(r)
    c2 = BatchNormalization(momentum=0.8)(c2)
    c2 = Add()([c2, c1])

    # Upsampling
    u1 = deconv2d(c2)
    u2 = deconv2d(u1)

    # Generate high resolution output
    gen_hr = Conv2D(channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

    return Model(img_lr, gen_hr)


def build_discriminator(df=64):
    def d_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    # Input img
    d0 = Input(shape=hr_shape)

    d1 = d_block(d0, df, bn=False)
    d2 = d_block(d1, df, strides=2)
    d3 = d_block(d2, df * 2)
    d4 = d_block(d3, df * 2, strides=2)
    d5 = d_block(d4, df * 4)
    d6 = d_block(d5, df * 4, strides=2)
    d7 = d_block(d6, df * 8)
    d8 = d_block(d7, df * 8, strides=2)

    d9 = Dense(df * 16)(d8)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(d0, validity)