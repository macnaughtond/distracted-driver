
import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

import daveutils
from daveutils import *
import davenet
from davenet import *

from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from scipy.misc import toimage

np.random.seed(2016)
use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 3
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1)) #These values are for cats and dogs. Change or remove for other data sets


def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model

def load_hat_weights(model, fc_layers, index, cross=''):
    # iterate through model from num_conv_layers until end, setting the layer weights
    #for index, layer in enumerate(alllayers[layer_idx+1:]):
    read_model(4, cross='old')  #read_model(index, cross='')  
    # can't I just go: model.add(fc_layers)
    model.add(fc_layers)
    return model


def get_gen(directory, shift_h, shift_v, rot, shear, chan_shift):
    datagen = ImageDataGenerator(width_shift_range=shift_h, height_shift_range=shift_v, rotation_range=rot, shear_range=.1, channel_shift_range=20)
    generator = datagen.flow_from_directory(directory,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle = False)
    return generator


def augment_save_ims(features_array):
    generator = get_gen(directory=directory, shift_h=0.1, shift_v=0.5, rot=4, shear=2, chan_shift=20)
    #print("len(features_array) = "+str(len(features_array)))
    #scipy.misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')
    for i,feat in enumerate(features_array): #dtype=np.int8
        print("features_array[i].shape = "+str(feat.shape))
        feat.astype(np.uint8)
        scipy.misc.toimage(feat, cmin=0.0, cmax=255).save(os.join(directory,'outfile_'+str(i)+'.jpg'),rgb)
        #im = Image.fromarray(image)
        #im.save(os.join(directory,'outfile_'+str(i)+'.jpg'))
        #toimage(arr, high, low, cmin, cmax, pal, mode, channel_axis)
    return


def flow_resize_pred_save_feats(directory, conv_model, nsamples): #samples_per_epoch, nb_epoch
    generator = get_gen(directory=directory, shift_h=0.1, shift_v=0.5, rot=4, shear=2, chan_shift=20)
    #predict_generator(self, generator, val_samples, max_q_size=10, nb_worker=1, pickle_safe=False)
    features_array = conv_model.predict_generator(generator, nsamples) #fit_generator samples_per_epoch, nb_epoch, verbose=1, validation_data=None
    

    

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

   
def my_cross_validation(model, train_data, train_target, filenames, nworst=40, ifolds=10, nb_epoch=10, modelStr='new', sample_weights=None, augment=False):
    # Now it loads color image
    # input image dimensions
    
    n_view = int(round(nworst/ifolds))
    batch_size = 64
    random_state = 20
    worst_img_nums = []
    worst_img_ids = []
    worst_pred_class = []
    filenames = np.array(filenames)
    num_fold = 0
    kf = KFold(n_splits=ifolds, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(train_data, train_target):
        print("X, y, sample_weights[train_index] = " + str(train_data[train_index].shape) +" "+ str(train_target[train_index].shape)+" "+str(sample_weights[train_index].shape) )
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, ifolds))
        if augment:
            #fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=None, validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10, nb_worker=1, pickle_safe=False, initial_epoch=0)
            model.fit_generator(generator, samples_per_epoch, nb_epoch, verbose=1, validation_data=None)
            #generator: a generator. The output of the generator must be either
                # tuple (inputs, targets)
                # tuple (inputs, targets, sample_weights). All arrays should contain the same number of samples. The generator is expected to loop over its data indefinitely. An epoch finishes when samples_per_epoch samples have been seen by the model.
            # samples_per_epoch: integer, number of samples to process before going to the next epoch.
        else:
            model.fit(train_data[train_index], train_target[train_index], batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, \
                      validation_data=(train_data[test_index], train_target[test_index]), shuffle=True, class_weight=None, sample_weight=sample_weights[train_index])
                      #callbacks=None, validation_split=0.0, class_weight=None, sample_weight=None)

        save_model(model, num_fold, modelStr)
        np.set_printoptions(precision=3)
        
        # Find and plot the worst fitting validation cases for this model:
        # Step 1: Determine predictions for the validation data
        print("Predicting classes for validation fold:")
        preds = model.predict_classes(train_data[test_index], batch_size=batch_size) # predict(self, x, batch_size=32, verbose=0)

        # Step 2:
        # find the confidence had in these predictions (probabilities) of being in class c
        print("Predicting probabilities for validation fold:")
        probs = model.predict_proba(train_data[test_index], batch_size=batch_size)
        
        # Step 3:
        # Find indexes in the validation data array for the most misplaced confidence
        # 3a. Find the incorrectly classified cases
        print("Finding the incorrectly predicted cases in the validation fold:")
        actual_classes = np.argmax(train_target[test_index], axis=1)       
        incorrect = np.nonzero(actual_classes!=preds)[0] #[0] is important  #incorrect = np.where(actual_classes!=preds)[0] # no different to nonzero result
        
        # 3b: Find the estimated probabilies corresponding to the correct class in each case     
        print("Finding the estimated probabilities associated with incorrectly predicted cases in the validation fold:")
        estimatedProbForActualClass = np.choose(actual_classes[incorrect], probs[incorrect].T) # Does this work for train_target as opposed to val_classes??
        
        # 3c. Find the worst estimates of probabilities for each case in the validation sample
        print("Sorting to find the worst offenders:")
        test_incorrect_ids_sorted_by_prob = np.argsort(estimatedProbForActualClass)
        numMostIncorrect = min(n_view, len(incorrect))
        most_incorrect = test_incorrect_ids_sorted_by_prob[:n_view]
        #shortlist = np.array(filenames[test_index]) 
        #image_num = np.array([vfile[7:][:-4] for vfile in shortlist[incorrect][most_incorrect]]) #filenames[test_index][most_incorrect]])
        image_num = np.array([vfile[7:][:-4] for vfile in filenames[test_index][incorrect][most_incorrect]]) #filenames[test_index][most_incorrect]])        
        #known_class_name = np.array([cat_names[int(vfile[1:2])] for vfile in filenames[test_index][most_incorrect]])
        #known_class_num = np.array([vfile[1:2] for vfile in filenames[test_index][most_incorrect]])  
        #title_con = np.core.defchararray.add(assumed_class_name,' ')#,known_class_name
        #title_con = np.core.defchararray.add(title_con,known_class_num)
        #title_con = np.core.defchararray.add(title_con,' ')
        #title_con = np.core.defchararray.add(title_con,image_num)
        worst_img_nums.append(image_num)
        worst_img_ids.append(test_index[incorrect][most_incorrect])
        worst_pred_class.append(preds[incorrect][most_incorrect])

    worst_img_nums = np.concatenate((worst_img_nums), axis=0)
    worst_img_ids = np.concatenate((worst_img_ids), axis=0)
    worst_pred_class = np.concatenate((worst_pred_class), axis=0)
    return (worst_img_ids, worst_img_nums, worst_pred_class)
        





def knowledge_distill():
    # conv_feat: training data output feature map of base model
    #            shape: (num_samples, height, width, depth)
    # conv_test_feat: test data out feature map of base model
    #            shape: (num_samples, height, width, depth)
    # bn_model: model for fine-tunning, its input is the output feature map
    #           of base model, and its output is predicted label
    # preds: Predicted test data label (soft-target)
    #        shape: (num_samples, num_classes)
    preds = bn_model.predict(conv_test_feat, batch_size=batch_size)

    i_trn = 0
    i_test = 0
    # iterate through 800 mini-batch
    num_iter = 800
    # mini-batch size
    size_trn = 48
    size_test = 16
    # we must know how many batches are there per epoch to decide 
    # when to shuffle index array
    num_batch_per_epoch_trn = int(conv_feat.shape[0]/size_trn)
    num_batch_per_epoch_test = int(conv_test_feat.shape[0]/size_test)
    # Create an shuffled index array
    index_trn = np.random.permutation(num_batch_per_epoch_trn)
    index_test = np.random.permutation(num_batch_per_epoch_test)

    for i in range(num_iter):
        # Get an index number from the shuffled index array for the current loop i
        i_trn = index_trn[i%num_batch_per_epoch_trn]
        i_test = index_test[i%num_batch_per_epoch_test]    

        """
        Combine training data and pseudo-labeled test data
        For example, when i == 0:
        comb_features = np.concatenate((conv_feat[0:48], conv_test_feat[0:16], axis=0))
        comb_feature contains 48 data from training set and 16 data from pseudo-labeled test set
        similarly, when i == 10:
        comb_labels = np.concatenate((trn_labels[48*10:48*11],preds[16*10:16*11]), axis=0)   
        """    
        comb_features = np.concatenate((conv_feat[(size_trn*i_trn):size_trn*(i_trn+1)],
                                       conv_test_feat[(size_test*i_test):size_test*(i_test+1)]),axis=0)
        comb_labels = np.concatenate((trn_labels[(size_trn*i_trn):size_trn*(i_trn+1)],
                                     preds[(size_test*i_test):size_test*(i_test+1)]), axis=0)

        bn_model.train_on_batch(comb_features, comb_labels)
        
        # Shuffle index array after model had trained on the last mini-batch.
        if (i+1)%num_batch_per_epoch_trn == 0:
            index_trn = np.random.permutation(num_batch_per_epoch_trn)
        if (i+1)%num_batch_per_epoch_test == 0:
            index_test = np.random.permutation(num_batch_per_epoch_test)
