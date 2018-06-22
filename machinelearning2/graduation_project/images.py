from imgaug import augmenters as iaa
import imgaug as ia
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator, load_img, img_to_array
import numpy as np
import time

seq = iaa.Sequential([
    iaa.SomeOf((0, 1), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.AdditiveGaussianNoise(scale=0.1*255),
        # iaa.Add(value=45),
        iaa.Add(value=-45),
        iaa.GaussianBlur(sigma=(0, 4.0)) # blur images with a sigma of 0 to 3.0
    ],
    random_order=True, random_state=np.random.RandomState(time.time().as_integer_ratio()[1])),
], random_order=True)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq1 = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Invert(0.05, per_channel=True), # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               second=iaa.ContrastNormalization((0.5, 2.0))
                           )
                       ]),
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

from sklearn.datasets import load_files
import os
import random
import imageio
import matplotlib.pyplot as plt

# files1 = os.listdir("data/imgs/preview")
# file = random.choice(files1)
# while(os.path.isdir(file)):
#     file = random.choice(files1)


class ImageAugGenerator(ImageDataGenerator):
    def __init__(self, sequential=None, rescale=None, preprocessing_function=None):
        self.sequential = sequential
        self.preprocessing_function = preprocessing_function
        self.rescale = rescale
        super(ImageAugGenerator, self).__init__(rescale=rescale, data_format='channels_last')

    def random_transform(self, x):
        x = self.sequential.augment_image(x)
        return x

    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x = x * self.rescale
        return x


generator = ImageAugGenerator(seq1, rescale=1.0/255)
generator1 = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)


img = load_img('data/imgs/train/c0/img_316.jpg')
x = img_to_array(img)
print(x.shape)

# img_aug = seq1.augment_image(x)
# x1 = img_aug * (1.0/255)
# print(x1.shape)
#
# x = x.reshape((1,) + x.shape)
# print(x.shape)
# x2 = generator.flow(x, batch_size=1)
# print(x2.shape)

# i = 0
# for batch in generator1.flow(x, [0], batch_size=1):
#     print(batch)
#     i += 1
#     if i > 20:
#         break

# i = 0
# for batch in generator.flow_from_directory("data/imgs/train", target_size=(150,150), batch_size=2,
#                           save_to_dir='data/imgs/preview', save_prefix='c0', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break

# img = imageio.imread("proposal_img/img_16.jpg")
# # plt.imshow(img)
# img_aug = seq.augment_image(img)
# print(img_aug.shape)
# plt.imshow(img_aug)
# plt.show()

# for batch_idx in range(1000):
#     # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
#     # or a list of 3D numpy arrays, each having shape (height, width, channels).
#     # Grayscale images must have shape (height, width, 1) each.
#     # All images must have numpy's dtype uint8. Values are expected to be in
#     # range 0-255.
#     images = load_batch(batch_idx)
#     images_aug = seq.augment_images(images)
#     train_on_images(images_aug)

from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.layers.core import Dropout
from keras.optimizers import Adam, SGD
from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions, name='inceptionv3')

op = SGD(lr=0.0002, decay=4e-8, momentum=0.9, nesterov=True)
model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = generator1.flow_from_directory(
    "data/imgs/train",
    target_size=(299,299),
    batch_size=1,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    epochs=10,
    validation_steps=1000)