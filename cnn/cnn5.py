

from __future__ import absolute_import, division, print_function, unicode_literals



import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
import sys
import keras
import tensorflow  as tf

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import utils
import plot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time 

from datetime import datetime

import sklearn.metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,  precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.client import device_lib

import itertools
K.tensorflow_backend._get_available_gpus()
import seaborn as sns
from matplotlib import rcParams, gridspec

from keras.callbacks import TensorBoard


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#print(device_lib.list_local_devices())


data_augmentation = True
batch_size = 64     #64 ,  128 , 256 , 512 
num_classes = 5
epochs =1

img_rows, img_cols = 60, 60

class_names= [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ]


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cnn_trained_model.h5'



np.set_printoptions(precision=2)
le = LabelEncoder()
encoded_labels = le.fit_transform(class_names)



########################################################
#################### Loading dataset ###################

print('')
print('++++++++++++++++++ Load Data +++++++++++++++')

# the data, split between train# Plotting Configuration and test sets
(x_train, y_train), (x_test, y_test) = utils.load_hsyaa(resize=False, add_noise=True)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# data normalization
#x_train /= 255
#x_test /= 255

print('')

print("Shape: ", x_train.shape)
print("Label: ", class_names)

print('')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


########################################################
################### setup Tensorboard #################

########### visualization of single image ############# 

'''

img = np.reshape(x_train[0], (-1, 60, 60, 1))


# Clear out any prior log data.
#   !rm -rf logs

# Sets up a timestamped log directory.
logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

# Using the file writer, log the reshaped image.
with file_writer.as_default():
  tf.summary.image("Training data", img, step=0)


#   $tensorboard --logdir logs/train_data

#
################################################
###### visualization of multi -  images ######## 

with file_writer.as_default():
  # Don't forget to reshape.
  images = np.reshape(x_train[0:25], (-1, 60, 60, 1))
  tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

#    $ tensorboard --logdir logs/train_data


###############################################
#############  Plot History  ##################


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in

        history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(figsize=(8,6))

    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    from time import timeplt.figure(figsize=(8,6))

    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
'''

########### multiclass or binary report################
## If import seaborn as snsbinary (sigmoid output), set binary parameter to True

def full_multiclass_report(model,
                           x,
                           y_true,
                           class_names,
                           batch_size=batch_size,
                           binary=False):
    
    
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)
    # 3. Print accuracy score
    
    print( [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ])
    print('')
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))

    print("")

    # 4. Print classification report
    
    print("Classification Report")
    print('')
    #print(classification_report(y_true,y_pred,digits=5))
    print(classification_report(y_true,y_pred,target_names=  [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ]))

    print('')
    print( [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ])
    print('')

    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)

    plot_confusion_matrix(cnf_matrix, [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ])

    cm_df = pd.DataFrame(cnf_matrix,
                     index = class_names,
                     columns = class_names)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True)

    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.title('Cnn Model \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return cnf_matrix


####################################################
################ Plot confusion matrix #############


def plot_confusion_matrix(cm,class_names):
  """from sklearn.metrics import confusion_matrix
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)

  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.from sklearn.metrics import confusion_matrix
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

'''

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(test_images)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)



  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure , cm
'''

#######################################################
##### tensorbosrd for confusion matrix (above) ########
'''
# Clear out prior logging data.
#  !rm -rf logs/image

logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Define the basic TensorBoard callback.
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
'''
########################################################
################# Confusion matrix log #################
'''

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(test_images)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

'''
###################### Build model ###################
######################################################

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=x_train[0].shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation=Activation(tf.nn.softmax)))

model.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer=keras.optimizers.Adadelta(),
                                metrics=['accuracy'])

model.summary()

#################### Data Augmentation ################
#######################################################


if not data_augmentation:

    print('>>>>>>> process without data augmentation <<<<<<<<')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=0,
              shuffle=True)


else:
     print('>>>>>>>> Process with data augmentation <<<<<<<<<<<')
     
     datagen = ImageDataGenerator(featurewise_center=True,
             featurewise_std_normalization=True,
             rotation_range=10,
             width_shift_range=0.2,
             height_shift_range=0.2,
             brightness_range=(-1, 1),
             horizontal_flip=True)


     datagen.fit(x_train)

     model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1)

     
########################################################
####### fit model as history with out CM  #############


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = keras.callbacks.TensorBoard(log_dir=logdir)

#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


model.fit(x_train, y_train,batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

history= model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[tensorboard])


#####################################################
############### Apply Tensorboard ##################

    # Clear out prior logging data.
#   !rm -rf logs/image
#   %tensorboard --logdir logs/train_data
#   %tensorboard --logdir logs/plots
#   %tensorboard --logdir logs/image


#################### Results ########################
print("=============  plot history  ===============")
'''
plot_history(history)
'''
print("")
score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test Accuracy:', score[1])
print('')


################## full report ###################
##################################################

print("========= full report Train set ==========")


full_multiclass_report(model,
                       x_train,
                       y_train,
                       le.inverse_transform(np.arange(5)))

print("")
print("========= full report Test set ===========")
print("")

full_multiclass_report(model,
                       x_test,
                       y_test,
                       le.inverse_transform(np.arange(5)))

print("")
print("============ Save Model weight ===========")
print("")

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


print("")
print("==========================================")
print("")

print(' ###################### TensorBoard Output ###################### ')
print(' #           $ !rm -rf logs/image                               # ')
print(' #           $ tensorboard --logdir logs/train_data             # ')
print(' #           $ tensorboard --logdir logs/plots                  # ')
print(' #           $ tensorboard --logdir logs/image                  # ')
print(' ################################################################ ')


print("")
print("==========================================")
print("================Basic Info================")
print("")

print('1) exC', '2) mddC', '3) nC', '4) mC', '5) mduC')
print('Batch Size:',batch_size)
print('Class Numbers:', num_classes)
print('Epochs:',epochs)

print("")
print("=========================================")
print("==================End====================")


