import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
#from __future__ import absolute_import, division, print_function, unicode_literals

#tf.enable_eager_execution()
#print(tf.__version__)
import IPython.display as display
import pathlib

IMAGESIZE = 280
BATCH_SIZE = 2
BUFFER = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE

data_root_orig = 'Images'
data_root = pathlib.Path(data_root_orig)
#print(data_root)

#for item in data_root.iterdir():
#  print(item)

import random
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count , 'ImageCount')
all_image_paths[:10]



img_path = all_image_paths[0]
#print(img_path)

img_raw = tf.io.read_file(img_path)
#print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_image(img_raw)

#print(img_tensor.shape)
#print(img_tensor.dtype)

img_final = tf.image.resize(img_tensor, [IMAGESIZE, IMAGESIZE])
img_final = img_final/255.0
#print(img_final.shape)
#print(img_final.numpy().min())
#print(img_final.numpy().max())

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

img_path = all_image_paths[0]


#plt.imshow(load_and_preprocess_image(img_path))
#plt.grid(False)
#plt.xlabel(caption_image(img_path).encode('utf-8'))
#plt.title(label_names[label].title())
#print()

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
#path_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#path_ds
#print('shape: ', repr(path_ds.output_shapes))
#print('type: ', path_ds.output_types)
#print()
print("path_ds" , path_ds)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

print("image_ds" ,  image_ds)

import matplotlib.pyplot as plt

plt.figure(figsize=(4,4))
for n,image in enumerate(image_ds.take(2)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  #plt.grid(False)
  #plt.xticks([])
  #plt.yticks([])
  #plt.xlabel(caption_image(all_image_paths[n]))
  #plt.show()

#image_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

#for label in label_ds.take(10):
#  print(label_names[label.numpy()])

#image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 2
BUFFER = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)


#ds = image_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
#ds = ds.batch(BATCH_SIZE)
#ds = ds.prefetch(buffer_size=AUTOTUNE)
print("DS",ds)

###

#mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
#mobile_net.trainable=False

def change_range(image):
  return 2*image-1

keras_ds = ds.map(change_range)

print("keras_ds",keras_ds)

# The dataset may take a few seconds to start, as it fills its shuffle buffer.
image_batch = next(iter(keras_ds))

#print("image_batch",image_batch)
#feature_map_batch = mobile_net(image_batch)
#print(feature_map_batch.shape)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# SLICE, BATCH AND SHUFFLE DATA
train_images = image_batch  
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(train_dataset)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(70*70*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((70, 70, 256)))
    assert model.output_shape == (None, 70, 70, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 70, 70, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 140, 140, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 280, 280, 3)

    return model
  
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[280, 280, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
  
  
discriminator = make_discriminator_model()

decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
  
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
  
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 1

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 100 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
  
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
      plt.axis('off')

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

EPOCHS = 1
noise_dim = 100
num_examples_to_generate = 1

train(train_dataset, EPOCHS)