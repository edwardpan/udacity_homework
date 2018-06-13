from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.AdditiveGaussianNoise(scale=0.1*255),
    # iaa.Add(value=45),
    iaa.Add(value=-45),
    iaa.GaussianBlur(sigma=(0, 4.0)) # blur images with a sigma of 0 to 3.0
])


from sklearn.datasets import load_files
import os
import random
import imageio
import matplotlib.pyplot as plt

# files1 = os.listdir("data/imgs/preview")
# file = random.choice(files1)
# while(os.path.isdir(file)):
#     file = random.choice(files1)

img = imageio.imread("proposal_img/img_16.jpg")
# plt.imshow(img)
img_aug = seq.augment_image(img)
plt.imshow(img_aug)
plt.show()

# for batch_idx in range(1000):
#     # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
#     # or a list of 3D numpy arrays, each having shape (height, width, channels).
#     # Grayscale images must have shape (height, width, 1) each.
#     # All images must have numpy's dtype uint8. Values are expected to be in
#     # range 0-255.
#     images = load_batch(batch_idx)
#     images_aug = seq.augment_images(images)
#     train_on_images(images_aug)