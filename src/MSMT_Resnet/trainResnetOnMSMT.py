import keras
from PIL import Image
from keras import optimizers, Input, Model
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, \
    Dense
from keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.resnet50 import conv_block, identity_block
from keras.callbacks import ModelCheckpoint

NUM_OF_CHARACTERS_OF_ID = 4
NUM_OF_ID_TRAIN = 1041
NORMALIZING_COSTANTS = [103.939, 116.779, 123.68]
GPU_FRACTION = 0.5
NUM_LAYERS_TO_FREEZE = 0
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
SHAPE_INPUT_NN = (224, 224, 3)
BATCH_SIZE = 16
PATH_LIST_OF_TRAINFILES = '/media/data/dataset/MSMT17_V1/list_train.txt'
TRAINDATA_PATH = '/media/data/dataset/MSMT17_V1/train/'
WEIGHTS_PATH = '/home/jansaldi/Progetto-tesi/weights/resnet50_tf_weights_imagenet.h5'
NAME_MODEL_TO_SAVE = 'Resnet_MSMT_config3'


def halfGPU():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
    set_session(tf.Session(config=config))


def count_images(trainlist_file):
    # @input : path of train images
    # @output : num of images in the directory
    num_imgs = 0
    for line in trainlist_file:
        num_imgs +=1
    return num_imgs


def read_line(line_of_file):
    splitted_line = line_of_file.split()
    id = splitted_line[1]
    img_path = splitted_line[0]
    return id, img_path


def image_read(img_path):
    image = Image.open(img_path).convert('RGB')
    return image


def prepare_x_train(image):
    x = img_to_array(image)
    x = x[:, :, ::-1]
    x[:, :, 0] -= NORMALIZING_COSTANTS[0]
    x[:, :, 1] -= NORMALIZING_COSTANTS[1]
    x[:, :, 2] -= NORMALIZING_COSTANTS[2]
    return x


def prepare_y_train(id, num_id):
    y = keras.utils.to_categorical(id, num_id)
    return y


def create_trainData(shape_input_nn, trainlist_file, num_imgs, num_id):
    # @input : size of images, path of training images,dict and num of id
    # @output : X_train and Y_train
    X_train = np.empty((num_imgs, shape_input_nn[0], shape_input_nn[1], shape_input_nn[2]), 'float32')
    Y_train = np.empty((num_imgs, num_id), 'float32')
    index = 0
    for fileline in trainlist_file:
        [identity, img_path] = read_line(fileline)

         # create x_train
        image = image_read(TRAINDATA_PATH + img_path).resize(shape_input_nn[0:2])
        X_train[index, :, :, :] = prepare_x_train(image)

        # create Y_train
        Y_train[index, :] = prepare_y_train(int(identity), num_id)
        index += 1

    print "dimensione x_train: " + str(X_train.shape)
    print "dimensione y_train: " + str(Y_train.shape)
    return X_train, Y_train


def flip_image_horizontally(image):
    return np.flip(image, 1)


def create_trainData_data_augmentation(shape_input_nn, trainlist_file, num_imgs, num_id):
    # @input : size of images, path of training images,dict and num of id
    # @output : X_train and Y_train
    X_train = np.empty((2*num_imgs, shape_input_nn[0], shape_input_nn[1], shape_input_nn[2]), 'float32')
    Y_train = np.empty((2*num_imgs, num_id), 'float32')
    index = 0
    for fileline in trainlist_file:
        [identity, img_path] = read_line(fileline)

         # create x_train
        image = image_read(TRAINDATA_PATH + img_path).resize(shape_input_nn[0:2])
        X_train[index, :, :, :] = prepare_x_train(image)

        # create Y_train
        y_label = prepare_y_train(int(identity), num_id)
        Y_train[index, :] = y_label
        index += 1

        flipped_image = flip_image_horizontally(image)
        X_train[index, :, :, :] = prepare_x_train(flipped_image)
        Y_train[index, :] = y_label
        index += 1

        print str(index)

    print "dimensione x_train: " + str(X_train.shape)
    print "dimensione y_train: " + str(Y_train.shape)
    return X_train, Y_train


def freeze_layers(model, n_layers_to_freeze):
    # freeze the weights of firsts layers
    for layer in model.layers[:n_layers_to_freeze]:
        layer.trainable = False
    return model


def create_resnet_model(num_classes, shape_input_nn):
    # @input: num of classes of the new final softmax layer, num layers to freeze
    # @output: Resnet final model with new softmax layer at the end

    #creating Resnet network
    img_input = Input(shape= shape_input_nn)
    bn_axis = 3
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fcnew')(x_fc)

    resnet_model = Model(img_input, x_fc)

    # load weights
    resnet_model.load_weights(WEIGHTS_PATH)

    #creating new last softmax layer
    x_new_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_new_fc = Flatten()(x_new_fc)
    x_new_fc = Dense(num_classes, activation='softmax', name='fcnew')(x_new_fc)

    #creating the new model
    resnet_model = Model(img_input, x_new_fc)

    return resnet_model


def compile_model(model, learning_rate):
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def fine_tune_model(model_to_fine_tune, nb_epoch, batch_size, traindata):
    save_checkpoint = ModelCheckpoint(filepath='/home/jansaldi/Progetto-tesi/models/' + NAME_MODEL_TO_SAVE + ".{epoch:02d}.h5",
                                      monitor='val_loss', verbose=0, save_best_only=False,
                                      save_weights_only=False, period=5)
    model_to_fine_tune.fit(traindata[0], traindata[1], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
                           callbacks=[save_checkpoint])
    return model_to_fine_tune


halfGPU()

trainlist_file = open(PATH_LIST_OF_TRAINFILES, 'r')
num_imgs = count_images(trainlist_file)
trainlist_file.seek(0)

traindata = create_trainData_data_augmentation(SHAPE_INPUT_NN, trainlist_file, num_imgs, NUM_OF_ID_TRAIN)
trainlist_file.close()


model = create_resnet_model(NUM_OF_ID_TRAIN, SHAPE_INPUT_NN)
model = freeze_layers(model, NUM_LAYERS_TO_FREEZE)
model = compile_model(model, LEARNING_RATE)
print model.summary()
model = fine_tune_model(model, NUM_EPOCHS, BATCH_SIZE, traindata)



