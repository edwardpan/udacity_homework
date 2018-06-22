from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.layers.core import Dropout
from keras.optimizers import Adam, SGD, Adadelta
from keras.losses import categorical_crossentropy
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.utils import plot_model


base_model = Xception(weights=None, include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions, name=base_model.name)
# model.load_weights('saved_weights/xception_0.h5')
#
# #     op = SGD(lr=0.0002, decay=4e-8, momentum=0.9, nesterov=True)
op = Adam(lr=0.001, decay=10e-8)
model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])

# print(model.summary())
# plot_model(model, to_file='Xception.png')

import pandas as pd
# import numpy as np
from keras.preprocessing.image import *

df = pd.read_csv("data/sample_submission.csv")

# df = pd.read_csv("data/driver_imgs_list.csv")
# print(df.head())
# choices = np.random.choice(df["subject"].drop_duplicates(), 2)
# print(choices)
# imgs_pd = df["img"]
# class_pd = df["classname"]
# subject_pd = df["subject"]
#
# val_index = []
# for choice in choices:
#     val_index.extend(df[df["subject"] == choice].index.tolist())
# print(len(val_index))
#
# test_mask = np.zeros(np.alen(df), dtype=np.bool)
# for val_i in val_index:
#     test_mask[val_i] = True
# print(np.alen(subject_pd[np.logical_not(test_mask)]))

import os

test_img_files = os.listdir("data/imgs/test1/test")
print(test_img_files[0])
print(os.path.join("data/imgs/test1/test", test_img_files[0]))

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("data/imgs/test1", (299, 299), shuffle=False,
                                         batch_size=16, class_mode=None)
filenames = test_generator.filenames
i = 0
for img in test_generator:
    print(i)
    # print(img.shape)
    y_preds = model.predict(img, verbose=1)
    y_preds = y_preds.clip(min=0.005, max=0.995)
    # print(y_preds)
    for j, y_pred in enumerate(y_preds):
        index = i * 16 + j
        fname = filenames[index]
        # print("file:", fname, ", index:", index, ", pred:", y_pred)
        for k, c in enumerate(y_pred):
            # print("c"+str(k), "=", c)
            df.at[index, 'c'+str(k)] = c
    i += 1

print(df.head())

df.to_csv('data/pred.csv', index=None)
print("predict done.")
