import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pandas


import cv2
import os
import scipy
import numpy as np
import tensorflow as tf


def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

if __name__=="__main__":
    pass
    plot_log('result/log.csv')


    #######################################################################################################




def edge_detector(images, labels):
    new_images = []
    new_labels = []
    for img_, label_ in zip(images, labels):
        #img = np.concatenate((img_, img_, img_), axis=2).astype(np.uint8)
        img = np.uint8(img_)
        try:
            edge_img = cv2.Canny(img, img.shape[0], img.shape[0])
        except:
            print('failed')
            continue
        new_images.append(edge_img.reshape(edge_img.shape[0], edge_img.shape[0], 1))
        new_labels.append(label_)
    return new_images, new_labels
        

def augment(images, labels, num_noisy_imgs=7):
    new_images = []
    new_labels = []
    for img, num in zip(images, labels):
        for _ in range(num_noisy_imgs):
            img_cp = img + np.random.normal(0.0, 6.0, img.shape)
            img_cp = np.clip(img_cp, 0, 255)
            new_images.append(img_cp)
            new_labels.append(num)
            #new_images.append(np.fliplr(img_cp))
            #new_labels.append(num)

        for i in range(-num_noisy_imgs, num_noisy_imgs):
            if i == 0: continue
            img_cp = img + (i*3)
            img_cp = np.clip(img_cp, 0, 255)
            new_images.append(img_cp)
            new_labels.append(num)
            #new_images.append(np.fliplr(img_cp))
            #new_labels.append(num)
    return new_images, new_labels



# path to current file
DIR = os.path.dirname(os.path.realpath(__file__))

def load_hsyaa(resize=True, add_noise=True, vectorize=False, only_test=False):
    #path = DIR + '/../../../HSYAA_training_images/hsyaa_5Class_Tr1000_Te180_60x60_good/'
    #path ='/home/mghafari/github/HSYAA_training_images/hsyaa_5Class_Tr1000_Te180_60x60_good/'
    path = DIR + '/../../../HSYAA_training_images/hsyaa_5Class_Tr1000_Te180_60x60_good/'
   # path ='/home/mghafari/github/HSYAA_training_images/hsyaa_5Class_Tr1000_Te180_60x60_good/'


    for loc in ['training', 'test']:
        if only_test and loc=='training':
            continue
        dirs = [dir for dir in os.listdir(path+'{}-images'.format(loc)) if '_add' not in dir]
       # print(dirs)
        images = []
        labels = []
        for num, label in enumerate(dirs):                                  # this will loop from 0, 1, 2 (i.e. labels)
            folder = path + '/{}-images/{}'.format(loc, label)
            for _file in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, _file))
                if resize:
                    img = cv2.resize(img,(60,60),interpolation=cv2.INTER_AREA)
                if img is not None:
                    try: img = img[:, :, 0].reshape(img.shape[0], img.shape[0], 1).astype(np.float32)       # the z-dimension was 3, but they were just copies so we need to flatten it\
                    except: continue

                    images.append(img)
                    label = np.zeros(len(dirs))
                    label[num] = 1
                    labels.append(num)
                # add noise to dataset
        if loc=='training':
            if add_noise:
                images, labels = augment(images, labels)
            if vectorize:
                images, labels = edge_detector(images)
            train_x = np.array(images) / 255.                           # normalize pixels to be between 0 and 1
            train_y = np.array(labels)
            train_y = train_y.reshape(train_y.shape[0], 1)
        else:
            if vectorize:
                images, labels = edge_detector(images)
            test_x = np.array(images) / 255.
            test_y = np.array(labels)
            test_y = test_y.reshape(test_y.shape[0], 1)
    #print(train_y)
    #print(test_y)
    if only_test:
        return None, None, test_x, test_y
    return (train_x, train_y), (test_x, test_y)





def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_hsyaa(batch_size, is_training)
    else:

        raise Exception('Invalid dataset, please check the name of dataset:', dataset)



    
def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_hsyaa(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)
