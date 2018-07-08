import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

status = ["safe driving", " texting - right", "phone - right", "texting - left", "phone - left", 
                "operation radio", "drinking", "reaching behind", "hair and makeup", "talking"]

def view_pred(model=None, show_num=10, test_dir=None, out_image_size=(299, 299), preprocess_input=None):
    test_show_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
    test_show_generator = test_show_datagen.flow_from_directory(test_dir, out_image_size, shuffle=False, 
                                                 batch_size=1, class_mode=None)

    plt.figure(figsize=(12, 24))
    for i, x in enumerate(test_show_generator):
        if i >= show_num:
            break
        plt.subplot(5, 2, i+1)
        preds = model.predict(x)
        preds = preds[0]

        max_idx = np.argmax(preds)
        pred = preds[max_idx]

        plt.title('c%d |%s| %.2f%%' % (max_idx , status[max_idx], pred*100))
        plt.axis('off')
        x = x.reshape((x.shape[1], x.shape[2], x.shape[3]))
        img = array_to_img(x)
        plt.imshow(img)