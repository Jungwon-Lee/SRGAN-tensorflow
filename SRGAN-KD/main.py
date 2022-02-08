import tensorflow as tf
from tensorflow import keras 

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image_dataset_from_directory

from data_loader import DataLoader
from models import build_generator, build_discriminator, build_vgg

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import load_img

from IPython.display import display
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from preprocessing import scaling, random_flip, process_input
import logging

channels = 3
lr_height = 64  # Low resolution height
lr_width = 64  # Low resolution width
lr_shape = (lr_height, lr_width, channels)
hr_height = lr_height * 4  # High resolution height
hr_width = lr_width * 4  # High resolution width
hr_shape = (hr_height, hr_width, channels)

# Number of residual blocks in the generator
n_residual_blocks = 16

mse = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.BinaryCrossentropy()

patch = int(hr_height / 2**4)
disc_patch = (patch, patch, 1)


def preprocess(ds, train=True):
    ds = ds.map(scaling)
    
    if train:
        ds = ds.map(random_flip)
        
    ds = train_ds.map(
        lambda x: (process_input(x, lr_height), x)
    )
    return ds

def discriminator_loss(real_output, fake_output):
    
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)
         
    real_loss = cross_entropy(valid, real_output)
    fake_loss = cross_entropy(fake, fake_output)
    total_loss = 0.5 * (real_loss + fake_loss)
    return total_loss

def generator_loss(imgs_lr, imgs_hr):
    
    v, img_f = combined([imgs_lr, imgs_hr])
    image_features = vgg(imgs_hr)
    
    valid = np.ones((batch_size,) + disc_patch)
    
    vgg_loss = mse(img_f, image_features)
    gan_loss = 1e-3 * cross_entropy(v, valid)
    vgg_loss = tf.dtypes.cast(vgg_loss, tf.float64)

    loss = tf.add(gan_loss, vgg_loss)

    return loss
   
@tf.function
def train_step(data):

    imgs_lr, imgs_hr = data
        
    with tf.GradientTape() as disc_tape:
        fake_hr = generator(imgs_lr, training=True)

        real_output = discriminator(imgs_hr, training=True)
        fake_output = discriminator(fake_hr, training=True)
            
        psnr = tf.image.psnr(fake_hr, imgs_hr, max_val=1.0)
        ssim = tf.image.ssim(fake_hr, imgs_hr, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        
    with tf.GradientTape() as gen_tape:
        gen_loss = generator_loss(imgs_lr, imgs_hr)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        
    return gen_loss, disc_loss, psnr, ssim

def train(epochs, batch_size=1, sample_interval=50):

    train_iter = iter(train_ds)
    
    start_time = datetime.datetime.now()
    psnr_list = []
    ssim_list = []

    for epoch in range(epochs):
#     for batch in train_ds.take(epochs):
        batch = next(train_iter)
        g_loss, d_loss, psnr, ssim = train_step(batch)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        
        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        if epoch % 10 == 0:
            print ("%d time: %s " % (epoch, elapsed_time), end=' ')
            tf.print(tf.strings.format("g_loss: {}, d_loss: {}, psnr: {}, ssim: {}", (g_loss, d_loss, np.mean(psnr_list), np.mean(ssim_list))), output_stream=sys.stdout)
            psnr_list = []
            ssim_list = []

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images(epoch)
            
def main(args):
    tf.random.set_seed(args.seed) # default 1004
    
    dataset_name = args.dataset_name
    
    current_dir = os.getcwd()
    root_dir = os.path.join(current_dir, 'datasets')
    image_size = (hr_height, hr_width)

    train_ds = image_dataset_from_directory(
        root_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        validation_split=0.2,
        subset="training",
        seed=1333,
        label_mode=None,
    )

    valid_ds = image_dataset_from_directory(
        root_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        validation_split=0.2,
        subset="validation",
        seed=1333,
        label_mode=None,
    )
    
    logging.info("Make dataset "+ dataset_name + " Success")
    
    train_ds = preprocess(train_ds, train=True)
    valid_ds = preprocess(valid_ds, train=False)
    
    train_ds = train_ds.prefetch(buffer_size=32)
    valid_ds = valid_ds.prefetch(buffer_size=32)
    
    logging.info("Prepare dataset "+ dataset_name + " Success")
    
    generator_optimizer = Adam(args.lr, args.mo)
    discriminator_optimizer = Adam(args.lr, args.mo)

    df = 64
    gf = 64

    '''build vgg19'''
    vgg = build_vgg()
    vgg.trainable = False

    '''build the discriminator'''
    patch = int(hr_height / 2**4)
    disc_patch = (patch, patch, 1)

    discriminator = build_discriminator(df)

    '''Build the generator''' 
    generator = build_generator(gf)
    
    '''Build combined'''
    # High res. and low res. images
    img_hr = Input(shape=hr_shape)
    img_lr = Input(shape=lr_shape)

    # Generate high res. version from low res.
    fake_hr = generator(img_lr)

    # Extract image features of the generated img
    fake_features = vgg(fake_hr)

    # Discriminator determines validity of generated high res. images
    validity = discriminator(fake_hr)

    combined = Model([img_lr, img_hr], [validity, fake_features])
    
    logging.info("Build model Success!")
    
    logging.info("Train Start")
    train(epochs = 3000)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-sd', type=int, default=1004,
                            help='Random seed nuber')
    parser.add_argument('--train_mode', '-t', default=True, action='store_false',
                            help='Train mode or not')
    
    parser.add_argument('--dataset_name', '-dn', type=str, required=True,
                            help='Dataset name')

    parser.add_argument('--batch_size', '-b', type=int, default=16,
                            help='Mini batch size')

    parser.add_argument('--checkpoint', '-cp', type=str, default=None,
                            help='Checkpoint for model load')

    parser.add_argument('--initial_lr', '-lr', type=float, default=0.0002,
                            help='Initial learning rate')

    parser.add_argument('--loss_func', '-lf', type=str, default='softmax',
                            help='Loss function type')

    parser.add_argument('--momentum', '-mo', type=float, default=0.5,
                            help='Momentum of optimizer')

    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5,
                            help='Weight decay of optimizer')

    parser.add_argument('--num_gpu', '-g', type=int, default=4,
                            help='Number of available GPU')

    parser.add_argument('--total_epoch', '-te', type=int, default=310,
                            help='Total number of epoch')

    parser.add_argument('--save_path', '-s', type=str,
                            default='/mnt/aitrics_ext/ext01/marcus/',
                            help='Directory of checkpoint saving')

    parser.add_argument('--exp_name', '-en', type=str, default='result_logs',
                            help='Save the result or not')

    parser.add_argument('--info', '-d', default=False, action='store_true',
                            help='Choose level of log level')

    args = parser.parse_args()
    
    if args.info:
        get_logger(print_level='DEBUG', file_level='DEBUG')
    else:
        get_logger(print_level='INFO', file_level='INFO')

    main(args)