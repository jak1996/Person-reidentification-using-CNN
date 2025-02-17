import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.layers import Dense
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt


NUM_OF_CHARACTERS_OF_ID = 4
GPU_FRACTION = 0.5
NORMALIZING_COSTANTS = [103.939, 116.779, 123.68]
NUM_LAYERS_TO_FREEZE = 0
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
SHAPE_INPUT_NN = (224, 224, 3)
BATCH_SIZE = 16
TRAINDATA_PATH ='/media/data/dataset/Market-1501-v15.09.15/bounding_box_train/'
NAME_MODEL_TO_SAVE = 'VGG_Market_config3'


def halfGPU():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
    set_session(tf.Session(config=config))


def count_id(path):
    # @input : path of Market
    # @output : dictionary of ID-integers to build keras input labels
    listing = os.listdir(path)
    #dictionary = dictionary for conversion from ID to continuous mapping output (key :ID to integer from 0 to 750)
    dictionary = {}
    index = 0
    for filename in listing:
        if filename.endswith('.jpg'):
            # ID indice ID immagine
            ID = filename[:NUM_OF_CHARACTERS_OF_ID]
            if index == 0:
                dictionary[ID] = index
                index += 1
            else:
                is_not_already_listed = True
                for keys in dictionary:
                    if keys == ID:
                        is_not_already_listed = False
                if is_not_already_listed:
                    dictionary[ID] = index
                    index += 1
    return dictionary


def count_images(path_traindata):
    # @input : path of train images
    # @output : num of images in the directory
    listing = os.listdir(path_traindata)
    num_imgs = 0
    for filename in listing:
        if filename.endswith(".jpg"):
            num_imgs += 1
    return num_imgs


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


def prepare_y_train(id_int_dictionary, filename, num_id):
    ID = filename[:NUM_OF_CHARACTERS_OF_ID]
    y = id_int_dictionary[ID]
    y = keras.utils.to_categorical(y, num_id)
    return y


def create_trainData(shape_input_nn, path_train, id_int_dict, num_id):
    # @input : size of images, path of training images
    # @output : X_train and Y_train
    listing = os.listdir(path_train)
    num_imgs = count_images(path_train)
    X_train = np.empty((num_imgs, shape_input_nn[0], shape_input_nn[1], shape_input_nn[2]), 'float32')
    Y_train = np.empty((num_imgs, num_id), 'float32')
    index = 0
    for filename in listing:
        if filename.endswith(".jpg"):
            # create x_train
            image = image_read(path_train + filename).resize(shape_input_nn[0:2])
            X_train[index, :, :, :] = prepare_x_train(image)
            # create Y_train
            Y_train[index, :] = prepare_y_train(id_int_dict, filename, num_id)
            index += 1

    print "dimensione x_train: " + str(X_train.shape)
    print "dimensione y_train: " + str(Y_train.shape)
    return X_train, Y_train


def flip_image_horizontally(image):
    return np.flip(image, 1)


def create_trainData_flippedImages(shape_input_nn, path_train, id_int_dict, num_id):
    # @input : size of images, path of training images, dict and num of id
    # @output : X_train and Y_train
    listing = os.listdir(path_train)
    num_imgs = count_images(path_train)
    X_train = np.empty((2*num_imgs, shape_input_nn[0], shape_input_nn[1], shape_input_nn[2]), 'float32')
    Y_train = np.empty((2*num_imgs, num_id), 'float32')
    index = 0
    for filename in listing:
        if filename.endswith(".jpg"):
            # create x_train image
            image = image_read(path_train + filename).resize(shape_input_nn[0:2])
            X_train[index, :, :, :] = prepare_x_train(image)

            # create Y_train
            y = prepare_y_train(id_int_dict, filename, num_id)
            Y_train[index, :] = y
            index += 1

            #create x_train flipped image
            flipped_image = flip_image_horizontally(image)
            X_train[index, :, :, :] = prepare_x_train(flipped_image)
            Y_train[index, :] = y
            index += 1

    print "dimensione x_train: " + str(X_train.shape)
    print "dimensione y_train: " + str(Y_train.shape)
    return X_train, Y_train


def print_layers(nn_model):
    for i, layer in enumerate(nn_model.layers):
        print str(i) + layer.name


def freeze_layers(model_to_freeze, num_layers_to_freeze):
    # freeze the weights of first layers
    for layer in model_to_freeze.layers[:num_layers_to_freeze]:
        layer.trainable = False
    return model_to_freeze


def create_vgg_model(num_classes,shape_input_nn):
    # @input: num of classes of the new final softmax layer, num of layers to freeze
    # @output: VGG final model with new softmax layer at the end

    # i use a sequential model because the VGG16 keras model doesn't have an "add" method to add new layers
    vgg_model = VGG16(include_top=True, weights='imagenet', input_shape=shape_input_nn)
    model = Sequential()
    for layer in vgg_model.layers[:-1]:
       model.add(layer)

    # don't use model.pop cause when you pop a layer directly from the list model.layers,
    #  the topology of this model is not updated accordingly.
    #  So all following operations would be wrong, given a wrong model definition.
    #  As a result, the next added layer will be called with a wrong input tensor
    # to solve the problem i don't add the last layer to the sequential model
    # so i don't have to pop it after
    model.add(Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, learning_rate):
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def fine_tune_model(model_to_fine_tune, nb_epoch, batch_size, traindata):
    # compile the model with SGD and a very slow learning rate
    save_checkpoint = ModelCheckpoint(
        filepath='/home/jansaldi/Progetto-tesi/models/' + NAME_MODEL_TO_SAVE + ".{epoch:02d}.h5",
        monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=False, period=10)
    history = model_to_fine_tune.fit(traindata[0], traindata[1], nb_epoch=nb_epoch, shuffle=True, batch_size=batch_size,
                                     verbose=2, callbacks=[save_checkpoint])
    return history


def plot_train_loss(history):
    plt.plot(history.history['loss'])
    plt.title('model train loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('/home/jansaldi/Progetto-tesi/Market_VGG/config1.jpg', dpi=500)
    plt.show()


halfGPU()

# path of market
dictionary = count_id(TRAINDATA_PATH)

num_ID = len(dictionary)
print "num of identities: " + str(num_ID)

traindata = create_trainData_flippedImages(SHAPE_INPUT_NN, TRAINDATA_PATH, dictionary, num_ID)

vgg_model = create_vgg_model(num_ID, SHAPE_INPUT_NN)
vgg_model = freeze_layers(vgg_model, NUM_LAYERS_TO_FREEZE)
vgg_model = compile_model(vgg_model, LEARNING_RATE)
print_layers(vgg_model)
print vgg_model.summary()
train_loss_history = fine_tune_model(vgg_model, NUM_EPOCHS, BATCH_SIZE, traindata)
plot_train_loss(train_loss_history)

