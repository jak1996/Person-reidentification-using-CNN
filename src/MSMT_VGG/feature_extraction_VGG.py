from PIL import Image
from keras.backend import set_session
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.models import Model
import tensorflow as tf
import numpy as np
import scipy.io as sio

GPU_FRACTION = 0.5
NORMALIZING_COSTANTS = [103.939, 116.779, 123.68]
SHAPE_INPUT_NN = [224, 224, 3]
DIM_OUTPUT_FEATURE_LAYER = 4096
NAME_FEATURE_EXTRACTION_LAYER = 'fc2'
NAME_MODEL_TO_LOAD = 'VGG_MSMT_config3.10.h5'
PATH_LIST_OF_QUERYFILES = '/media/data/dataset/MSMT17_V1/list_query.txt'
PATH_LIST_OF_GALLERYFILES = '/media/data/dataset/MSMT17_V1/list_gallery.txt'
PATH_TEST_IMAGES = '/media/data/dataset/MSMT17_V1/test/'


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


def count_images(trainlist_file):
    # @input : path of train images
    # @output : num of images in the directory
    num_imgs = 0
    for line in trainlist_file:
        num_imgs +=1
    return num_imgs


def read_line(line_of_file):
    splitted_line = line_of_file.split()
    img_path = splitted_line[0]
    return img_path


def image_read(img_path):
    image = Image.open(img_path).convert('RGB')
    return image


def create_input_to_predict(path_of_image):
    # @input: path of the image
    # @output : array ready to be predicted
    X_test = np.empty((1, SHAPE_INPUT_NN[0], SHAPE_INPUT_NN[1], SHAPE_INPUT_NN[2]), 'float32')
    image = image_read(path_of_image).resize(SHAPE_INPUT_NN[0:2])
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


def fill_feature_matrix(feature_matrix, model, datalist_file, num_imgs):
    index = 0
    for line in datalist_file:
        print_percentage(index, num_imgs)
        img_path = read_line(line)
        prediction = model.predict(create_input_to_predict( PATH_TEST_IMAGES + img_path))
        prediction = prediction.transpose()
        feature_matrix[:, index] = prediction.squeeze()
        index += 1
    return feature_matrix


#test trained networks on test data

halfGPU()

model = get_model_for_feature_extraction('/home/jansaldi/Progetto-tesi/models/' + NAME_MODEL_TO_LOAD, NAME_FEATURE_EXTRACTION_LAYER)

querylist_file = open(PATH_LIST_OF_QUERYFILES, 'r')
gallerylist_file = open(PATH_LIST_OF_GALLERYFILES, 'r')

num_query_imgs = count_images(querylist_file)
prob_feature = np.empty((DIM_OUTPUT_FEATURE_LAYER, num_query_imgs))
querylist_file.seek(0)

num_gallery_imgs = count_images(gallerylist_file)
gallery_feature = np.empty((DIM_OUTPUT_FEATURE_LAYER, num_gallery_imgs))
gallerylist_file.seek(0)

print ('NAME MODEL: ' + NAME_MODEL_TO_LOAD)
print('PROB_FEATURE_FILL:')
prob_feature = fill_feature_matrix(prob_feature, model, querylist_file, num_query_imgs)
querylist_file.close()
print('GALLERY_FEATURE_FILL:')
gallery_feature = fill_feature_matrix(gallery_feature, model, gallerylist_file, num_gallery_imgs)
gallerylist_file.close()


print " dim gallery_feature: " + str(gallery_feature.shape)
print " dim prob_feature: " + str(prob_feature.shape)

sio.savemat('/home/jansaldi/Progetto-tesi/MSMT_VGG/features/gallery_feature_config3.10.mat', mdict={'galFea': gallery_feature})
sio.savemat('/home/jansaldi/Progetto-tesi/MSMT_VGG/features/prob_feature_config3.10.mat', mdict={'probFea': prob_feature})

