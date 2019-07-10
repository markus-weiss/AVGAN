from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
#tf.enable_eager_execution()
#tf.__version__

AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib
data_root_orig = tf.keras.utils.get_file('flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
#data_root_orig = "Images/DATA1"
#data_root_orig = "Images/Data"
data_root = pathlib.Path(data_root_orig)
print(data_root)

for item in data_root.iterdir():
  print(item)

import random
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

import os
#attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
#attributions = [line.split(' CC-BY') for line in attributions]
#attributions = dict(attributions)

import IPython.display as display

#def caption_image(image_path):
#    image_rel = pathlib.Path(image_path).relative_to(data_root)
#    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])

#for n in range(3):
  #image_path = random.choice(all_image_paths)
  #display.display(display.Image(image_path))
  #print(caption_image(image_path))
  #print()

#label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
#label_names

#label_to_index = dict((name, index) for index,name in enumerate(label_names))
#label_to_index

#all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
 #                   for path in all_image_paths]

#print("First 10 labels indices: ", all_image_labels[:10])

img_path = all_image_paths[0]
print(img_path)

img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)

img_final = tf.image.resize(img_tensor, [28, 28])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [28, 28])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

import matplotlib.pyplot as plt

img_path = all_image_paths[0]

#label = all_image_labels[0]

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(tf.compat.v1.data.get_output_shapes(path_ds)))
print('type: ', tf.compat.v1.data.get_output_types(path_ds))
print()
print(path_ds)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

print(image_ds)

#import matplotlib.pyplot as plt

#plt.figure(figsize=(8,8))
#for n,image in enumerate(image_ds.take(4)):
  #plt.subplot(2,2,n+1)
  #plt.imshow(image)
  #plt.grid(False)
  #plt.xticks([])
  #plt.yticks([])
  #plt.xlabel(caption_image(all_image_paths[n]))
  #plt.show()

#print(image_ds)

dataset = tf.data.Dataset.from_tensor_slices((all_image_paths))
#print(dataset)

print("--------------------------------------Readed---------------------------------------------")