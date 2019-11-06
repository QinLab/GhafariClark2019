
#import keras
from tensorflow.python import keras as keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.backend import backend as K
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import regularizers, optimizers
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.initializers import glorot_normal, RandomNormal, Zeros
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
import tensorflow as tf
import utils

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,  precision_recall_fscore_support


from keras.callbacks import TensorBoard


from time import time

from datetime import datetime
from keras.callbacks import TensorBoard


#################### Configuration ###################
data_augmentation = False
num_classes=5
class_names= [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ]
batch_size = 128
epochs = 1


data_augmentation = False  # True
num_classes=5
class_names= [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ]
batch_size = 128
epochs = 1




np.set_printoptions(precision=2)
le = LabelEncoder()
encoded_labels = le.fit_transform(class_names)


class Softmax(Activation):
    def __init__(self, activation, **kwargs):
        super(Softmax, self).__init__(activation, **kwargs)
        self.__name__ = 'softmax'


print("************ Load dataset *******************")


# Data Retrieval & mean/std preprocess
(x_train, y_train), (x_test, y_test) = utils.load_hsyaa(resize=False, add_noise=True, vectorize=False)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)


y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)


################### Augmentation #####################

if not data_augmentation:
    print('')
    print('>>>>>>>Not using data augmentation <<<<<<<<')
    
    datagen = ImageDataGenerator(featurewise_center=False,
            featurewise_std_normalization=False,
            horizontal_flip=False)
    
    datagen.fit(x_train)

else:

    print('')
    print('>>>>>>>> Using real-time data augmentation <<<<<<<<<<<')

    '''
    datagen = ImageDataGenerator(featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=0,
            width_shift_range=0,
            height_shift_range=0,
            brightness_range=(-0, 0),
            zoom_range=(-0, 0),
            horizontal_flip=False)

    '''

    datagen = ImageDataGenerator(featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(-1, 1),
            horizontal_flip=True)



    datagen.fit(x_train)

################ Full Report ########################

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
    '''
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
    '''
    return cnf_matrix

########################################################
############# Define Model architecture ################
def create_model(s = 2, weight_decay = 1e-2, act="relu"):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=glorot_normal(), input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    #Block 2
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 3
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    #Block 4
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    # First Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))


    # Block 5
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 6
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 7
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # Second Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))


    # Block 8
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 9
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
# Third Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))


    # Block 10
    model.add(Conv2D(512, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 11  
    model.add(Conv2D(2048, (1,1), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 12  
    model.add(Conv2D(256, (1,1), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fourth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))


    # Block 13
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fifth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))

    # Final Classifier
    model.add(Flatten())
    model.add(Dense(num_classes, activation=Softmax(tf.nn.softmax)))

    return model



if __name__ == "__main__":
# Prepare for training 
    model = create_model(act="relu")
    batch_size = 128
    epochs = 1
    train = {}


#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = keras.callbacks.TensorBoard(log_dir=logdir)


# First training for 50 epochs - (0-50)
    opt_adm = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_1"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                            steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                            verbose=1,validation_data=(x_test,y_test),
                                            callbacks=[tensorboard])
    model.save("simplenet_generic_first.h5")
    print(train["part_1"].history)

# Training for 25 epochs more - (50-75)
    opt_adm = keras.optimizers.Adadelta(lr=0.7, rho=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_2"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                            steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                            verbose=1,validation_data=(x_test,y_test),
                                            callbacks=[tensorboard])
    model.save("simplenet_generic_second.h5")
    print(train["part_2"].history)

# Training for 25 epochs more - (75-100)
    opt_adm = keras.optimizers.Adadelta(lr=0.5, rho=0.85)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_3"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                            steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                            verbose=1,validation_data=(x_test,y_test),
                                            callbacks=[tensorboard])
    model.save("simplenet_generic_third.h5")
    print(train["part_3"].history)

    # Training for 25 epochs more  - (100-125)
    opt_adm = keras.optimizers.Adadelta(lr=0.3, rho=0.75)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_4"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                            steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                            verbose=1,validation_data=(x_test,y_test),
                                            callbacks=[tensorboard])
    model.save("simplenet_generic_fourth.h5")
    print(train["part_4"].history)

    print("\n \n Final Logs: ", train)


    model.summary()


    print("========= full report Train set ==========")
    print("")

    full_multiclass_report(model,
                       x_train,
                       y_train,
                       le.inverse_transform(np.arange(5)))


    print("========= full report Test set ===========")
    print("")
    full_multiclass_report(model,
                       x_test,
                       y_test,
                       le.inverse_transform(np.arange(5)))
    
    print('BATCH SIZE:', batch_size )
    print('EPOCHS:',epochs)


    print("===========    End REPORT   =============")

