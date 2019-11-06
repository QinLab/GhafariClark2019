


"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage: 

       python capsulenet.py -h     ( help ) 
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
       run on more than one GPU
       python capsulenet-multi-gpu.py --gpus 2
       

       
       
       Test a pre-trained CapsNet model
       Suppose you have trained a model using the above command, then the trained
       model will be saved to result/trained_model.h5. Now just launch the following
       command to get test results.


       python capsulenet.py -t -w result/trained_model.h5



        Test Errors
        CapsNet classification test error on MNIST. Average and standard deviation results
        are reported by 3 trials. The results can be reproduced by launching the following commands.

        python capsulenet.py --routings 1 --lam_recon 0.0    #CapsNet-v1   
        python capsulenet.py --routings 1 --lam_recon 0.392  #CapsNet-v2
        python capsulenet.py --routings 3 --lam_recon 0.0    #CapsNet-v3 
        python capsulenet.py --routings 3 --lam_recon 0.392  #CapsNet-v4


        
        Reconstruction result
        The result of CapsNet-v4 by launching

        python capsulenet.py -t -w result/trained_model.h5
        

        Manipulate latent code
        Digits at top 5 rows are real images from MNIST and digits at bottom are corresponding reconstructed images.

        python capsulenet.py -t --digit 5 -w result/trained_model.h5



"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,  precision_recall_fscore_support


from utils import combine_images

from tensorflow.python.keras.utils import np_utils





data_augmentation = True
num_classes=5
class_names= [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ]
batch_size = 128
epochs = 20


np.set_printoptions(precision=2)
le = LabelEncoder()
encoded_labels = le.fit_transform(class_names)


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
/home/ubuntu/github/HSYAA/model_playground
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)

    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})


    if not data_augmentation:

        print('')
        print('>>>>>>> Not using data augmentation <<<<<<<<')

        model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    else:


        print('')
        print('>>>>>>>> Using real-time data augmentation <<<<<<<<<<<')


        def train_generator(x, y, batch_size, shift_fraction=0.):
            train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)

            generator = train_datagen.flow(x, y, batch_size=batch_size)
            while 1:
                x_batch, y_batch = generator.next()
                yield ([x_batch, y_batch], [y_batch, x_batch])


        model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])


    model.save_weights(args.save_dir + '/trained_model.h5')

    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)


    return model



def test(model, data, batch_size,args):

    x_test, y_test = data
    #y_test = np.argmax(y_test,axis=1)

    y_pred, x_recon = model.predict(x_test, batch_size=batch_size)

    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


    y_pred = np.argmax(y_pred,axis=1)
    y_test = np.argmax(y_test,axis=1)


     # Print classification report

    print("Classification Report")

    print( [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ])
    print('')
    print(" Test Accuracy : "+ str(accuracy_score(y_test,y_pred)))

    print("")



    print('')
    #y_test = np.argmax(y_test,axis=1)
    #y_pred = np.argmax(y_pred, axis=1)
    #print(classification_report(y_true,y_pred,digits=5))
    print(classification_report(y_test,y_pred,target_names=  [ 'exC' , 'mddC' , 'nC ', 'mC' , 'mduC' ]))

    print('')

    cnf_matrix = confusion_matrix(y_test,y_pred)
    print(cnf_matrix)

    print('')


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])



################ Full Report ########################

def full_multiclass_report(model,
                           x_test,
                           y_true,
                           class_names,
                           batch_size=batch_size,
                           binary=False):


    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.predict(x_test,batch_size=batch_size )

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
    
    return cnf_matrix
    

if __name__ == "__main__":

    
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    epoch=20
    batch_size=128

    
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network  HSYAA.")
    parser.add_argument('--epochs', default=epoch, type=int)

    parser.add_argument('--batch_size', default=batch_size, type=int)

    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')

    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")

    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    print("************ Load dataset *******************")
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


    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()
    model.save("model_weight.h5")
    print('save model weight : DONE !!!')

     # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
            
        #manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test),batch_size=batch_size, args=args)



    print('1) exC', '2) mddC', '3) nC', '4) mC', '5) mduC')
    print('Batch Size:',batch_size)
    print('Class Numbers:', num_classes)
    print('Epochs:',epochs)

    print("===========    End REPORT   =============")

