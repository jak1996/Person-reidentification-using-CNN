import numpy as np
import scipy.io as sio
PATH_GALLERY_FEATURE = '/home/jansaldi/Progetto-tesi/Market_Resnet/features/gallery_feature.mat'
PATH_PROB_FEATURE = '/home/jansaldi/Progetto-tesi/Market_Resnet/features/prob_feature.mat'


def normalize(feature_vector):
    # @input: vector to be normalized
    # @output: normalized vector
    sum_val = np.sqrt(sum(np.square(feature_vector)))
    # check if sum is on columns
    for i in range(len(feature_vector[:, 0])):
        feature_vector[i, :] = feature_vector[i, :] / sum_val
    return feature_vector


def my_pdist(galFea, probFea):
    # @input: 2 matrixes nxd and mxd
    # #output: euclidean distance of 2 matrixes point to point nxm
    squared_galFea = np.square(galFea)
    squared_probFea = np.square(probFea)
    row_sum_galFea = np.sum(squared_galFea, axis=1)
    row_sum_probFea = np.sum(squared_probFea, axis=1)

    # code to make them row and column vectors
    row_sum_galFea = row_sum_galFea[:, np.newaxis]
    row_sum_probFea = row_sum_probFea[np.newaxis, :]

    double_product = 2 * np.dot(galFea, np.transpose(probFea))

    eucliden_distance = np.sqrt(row_sum_galFea + row_sum_probFea - double_product)
    return eucliden_distance

#def evaluation(dist, test_ID, query_ID, test_cam, query_cam):


query_id = sio.loadmat('/home/jansaldi/Progetto-tesi/utils/Market/queryID.mat')
query_cam = sio.loadmat('/home/jansaldi/Progetto-tesi/utils/Market/queryCam.mat')
test_id = sio.loadmat('/home/jansaldi/Progetto-tesi/utils/Market/testID.mat')
test_cam = sio.loadmat('/home/jansaldi/Progetto-tesi/utils/Market/testCam.mat')

gallery_feature = sio.loadmat(PATH_GALLERY_FEATURE)
prob_feature = sio.loadmat(PATH_PROB_FEATURE)

gallery_feature = normalize(gallery_feature)
prob_feature = normalize(prob_feature)

dist = my_pdist(gallery_feature, prob_feature)