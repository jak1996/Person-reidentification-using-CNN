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
NAME_MODEL_TO_LOAD = 'VGG_Market_config3.20.h5'
PATH_IN_WHICH_SAVE_GALLERY_FEATURE = '/home/jansaldi/Progetto-tesi/Market_VGG/features/gallery_feature_config3.20.mat'
PATH_IN_WHICH_SAVE_PROB_FEATURE = '/home/jansaldi/Progetto-tesi/Market_VGG/features/prob_feature_config3.20.mat'


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


def build_idcam_string_vector(id_vector, cam_vector):
    # @input: vector of id and vector of cam that identify every image
    # @output: vector of the same dim of the inputs that contains the strings to identify every image in the correct format
    string_vector = np.empty(id_vector.shape, dtype=np.object_)
    index = 0
    for id_num, cam_num in zip(id_vector, cam_vector):
        substring = "%04d_c%s" % (id_num, cam_num[0])
        if id_num == -1:
            substring = "%s_c%s" % (id_num[0], cam_num[0])
        string_vector[index, :] = substring
        index += 1
    return string_vector


def find_complete_filename(path_of_images, id_cam_string):
    # @input: path in which search for the image, string idcam that identifies it
    # @output: complete filename of the image
    listing = os.listdir(path_of_images)
    img_filename = ""
    for filename in listing:
        if id_cam_string in filename:
            img_filename = filename
            break
    return img_filename


def image_read(img_path):
    image = Image.open(img_path).convert('RGB')
    return image


def create_input_to_predict(idcam_string, path_of_images):
    # @input: string that identifies the image, path of the directory in which search for it
    # @output : array ready to be predicted
    X_test = np.empty((1, SHAPE_INPUT_NN[0], SHAPE_INPUT_NN[1], SHAPE_INPUT_NN[2]), 'float32')
    img_filename = find_complete_filename(path_of_images, idcam_string)
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
    print '{0:.2f}%'.format(percentage)


def fill_feature_matrix(feature_matrix, string_idcam_vector_, model, path_of_images):
    index = 0
    for string in string_idcam_vector_:
        print_percentage(index, len(string_idcam_vector_))
        prediction = model.predict(create_input_to_predict(string[0], path_of_images))
        prediction = prediction.transpose()
        feature_matrix[:, index] = prediction.squeeze()
        index += 1
    return feature_matrix

#test trained networks on test data

halfGPU()

model = get_model_for_feature_extraction('/home/jansaldi/Progetto-tesi/models/' + NAME_MODEL_TO_LOAD, NAME_FEATURE_EXTRACTION_LAYER)

query_id = sio.loadmat('/home/jansaldi/Progetto-tesi/utils/Market/queryID.mat')
query_cam = sio.loadmat('/home/jansaldi/Progetto-tesi/utils/Market/queryCam.mat')
test_id = sio.loadmat('/home/jansaldi/Progetto-tesi/utils/Market/testID.mat')
test_cam = sio.loadmat('/home/jansaldi/Progetto-tesi/utils/Market/testCam.mat')

path_query = '/media/data/dataset/Market-1501-v15.09.15/query/'
path_gallery = '/media/data/dataset/Market-1501-v15.09.15/bounding_box_test/'

string_query_vector = build_idcam_string_vector(query_id['queryID'], query_cam['queryCAM'])
string_gallery_vector = build_idcam_string_vector(test_id["testID"], test_cam['testCAM'])

prob_feature = np.empty((DIM_OUTPUT_FEATURE_LAYER, count_images(path_query)))
gallery_feature = np.empty((DIM_OUTPUT_FEATURE_LAYER, count_images(path_gallery)))

print('PROB_FEATURE_FILL:')
prob_feature = fill_feature_matrix(prob_feature, string_query_vector, model, path_query)
print('GALLERY_FEATURE_FILL:')
gallery_feature = fill_feature_matrix(gallery_feature, string_gallery_vector, model, path_gallery)


print " dim gallery_feature: " + str(gallery_feature.shape)
print " dim prob_feature: " + str(prob_feature.shape)

sio.savemat(PATH_IN_WHICH_SAVE_GALLERY_FEATURE, mdict={'gal': gallery_feature})
sio.savemat(PATH_IN_WHICH_SAVE_PROB_FEATURE, mdict={'prob': prob_feature})










