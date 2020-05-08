from PIL import Image
from keras.backend import set_session
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.models import Model
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os

GPU_FRACTION = 0.5
NORMALIZING_COSTANTS = [103.939, 116.779, 123.68]
SHAPE_INPUT_NN = [224, 224, 3]
DIM_OUTPUT_FEATURE_LAYER = 4096
NAME_FEATURE_EXTRACTION_LAYER = 'fc2'
NAME_MODEL_TO_LOAD = 'VGG_Duke_config2.20.h5'
PATH_QUERY_IMAGES = '/media/data/dataset/Duke_online/query/'
PATH_GALLERY_IMAGES = '/media/data/dataset/Duke_online/bounding_box_test/'
PATH_IN_WHICH_SAVE_GALLERY_FEATURES = '/home/jansaldi/Progetto-tesi/DukeMTMC_VGG/features/gallery_feature_config2.20.mat'
PATH_IN_WHICH_SAVE_PROB_FEATURES = '/home/jansaldi/Progetto-tesi/DukeMTMC_VGG/features/prob_feature_config2.20.mat'


def halfGPU():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
    set_session(tf.Session(config=config))


def get_model_for_feature_extraction(model_path, name_last_layer):
    # @input: path of model to load
    # @output: model with last layer truncated, ready to feature extraction
    model = load_model(model_path)
    model = Model(model.input, model.get_layer(name_last_layer).output)
    return model


def count_images(path_data):
    # @input : path of images
    # @output : num of images in the directory
    listing = os.listdir(path_data)
    num_imgs = 0
    for filename in listing:
        if filename.endswith(".jpg"):
            num_imgs += 1
    return num_imgs


def image_read(img_path):
    image = Image.open(img_path).convert('RGB')
    return image


def create_input_to_predict(img_filename, path_of_images):
    # @input: string that identifies the image, path of the directory in which search for it
    # @output : array ready to be predicted
    X_test = np.empty((1, SHAPE_INPUT_NN[0], SHAPE_INPUT_NN[1], SHAPE_INPUT_NN[2]), 'float32')
    image = image_read(path_of_images + img_filename).resize(SHAPE_INPUT_NN[0:2])
    x = img_to_array(image)
    x = x[:, :, ::-1]
    x[:, :, 0] -= NORMALIZING_COSTANTS[0]
    x[:, :, 1] -= NORMALIZING_COSTANTS[1]
    x[:, :, 2] -= NORMALIZING_COSTANTS[2]
    X_test[0, :, :, :] = x
    return X_test


def print_percentage(index, num_tot_iteration):
    index = index * 1.0
    percentage = (index / num_tot_iteration) * 100.0
    print '%01d' % percentage + '%'


def fill_feature_matrix(feature_matrix, model, path_of_images):
    index = 0
    listing = os.listdir(path_of_images)
    listing.sort()
    num_tot_iteration = len(listing)
    for filename in listing:
        print_percentage(index, num_tot_iteration)
        prediction = model.predict(create_input_to_predict(filename, path_of_images))
        feature_matrix[index, :] = prediction.squeeze()
        index += 1
    return feature_matrix


#test trained networks on test data

halfGPU()

model = get_model_for_feature_extraction('/home/jansaldi/Progetto-tesi/models/' + NAME_MODEL_TO_LOAD, NAME_FEATURE_EXTRACTION_LAYER)

prob_feature = np.empty((count_images(PATH_QUERY_IMAGES), DIM_OUTPUT_FEATURE_LAYER))
gallery_feature = np.empty((count_images(PATH_GALLERY_IMAGES),DIM_OUTPUT_FEATURE_LAYER))

print('PROB_FEATURE_FILL:')
prob_feature = fill_feature_matrix(prob_feature, model, PATH_QUERY_IMAGES)
print('GALLERY_FEATURE_FILL:')
gallery_feature = fill_feature_matrix(gallery_feature, model, PATH_GALLERY_IMAGES)


print " dim gallery_feature: " + str(gallery_feature.shape)
print " dim prob_feature: " + str(prob_feature.shape)

sio.savemat(PATH_IN_WHICH_SAVE_GALLERY_FEATURES, mdict={'gal': gallery_feature})
sio.savemat(PATH_IN_WHICH_SAVE_PROB_FEATURES, mdict={'prob': prob_feature})