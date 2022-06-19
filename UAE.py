import struct
import time
import os
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sklearn
from sklearn.preprocessing import LabelBinarizer
# model training and test
from contextlib import redirect_stdout
from keras import layers
import math
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import core, Flatten, Dense, Input, Dropout, Activation, Conv1D, MaxPooling1D, Reshape, AveragePooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import to_categorical
from keras.models import load_model
def ae_model(encoding_dim = 60,node=64,layer_nb=1,N_POI=2000):
    de_node = node
    encoder_input = x = keras.Input(shape=(N_POI, ), name='encoder_input')
    #node = node * math.pow(2,(layer_nb-1))
    for i in range(0,layer_nb):
        node = de_node
        node = node*math.pow(2,(layer_nb-i-1))
        print(node)
        x = layers.Dense(node, activation='relu')(x)
    encoder_output = layers.Dense(encoding_dim)(x)
    encoder = keras.Model(encoder_input, encoder_output, name='encoder')
    encoder.summary()
    decoder_input = x = keras.Input(shape=(encoding_dim, ), name='decoder_input')
#     node = de_node
#     x = layers.Dense(node, activation='relu')(x)
    decoder_output = layers.Dense(N_POI)(x)
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    decoder.summary()
    autoencoder_input = keras.Input(shape=(N_POI), name='autoencoder_input')
    encoded = encoder(autoencoder_input)
    autoencoder_output = decoder(encoded)
    autoencoder = keras.Model(
        autoencoder_input,
        autoencoder_output,
        name='autoencoder',
    )
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    modelSummary = './asymetric_AE_'+str(node)+'_node_'+str(encoding_dim)+'encodingDim_'+str(layer_nb)+'layers_model_summary.txt'
    print(modelSummary)
    with open(modelSummary, 'w') as f:
        with redirect_stdout(f):
            encoder.summary()
            decoder.summary()
            autoencoder.summary()
    return encoder,autoencoder,decoder

def load_data(file, load_metadata=False):
    #check_file_exists(file)
    # Open the database HDF5 for reading
    try:
        in_file  = h5py.File(file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    
    print(X_profiling.shape[0])
    print(X_profiling.shape[1])
    
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], 
                                                                  in_file['Attack_traces/metadata'])
labeled_file='./raw_dataset/ASCAD_Sbox3_3000_test.h5'
(X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_data(labeled_file, load_metadata=True)
print(X_profiling.shape)
print(X_attack.shape)

sbox = 'sbox3' 
file_path = './trained_models/AE_model' 
if (os.path.exists(file_path)==False):
    os.makedirs(file_path)
node_list = [256]
dim_list = [300] #10倍

for layer_nb in range(1,2):
    for node in node_list:
        for encoding_dim in dim_list:
            start = time.time() # time
            model_weights = file_path + '/ASCAD_'+str(encoding_dim*10)+'_AE_('+sbox+')_'+str(node)+'node_'+str(encoding_dim)+'encoding_dim_'+str(layer_nb)+'layers_model_weights.h5'  
            callbacks_list = [
                keras.callbacks.ModelCheckpoint(
                    filepath=model_weights, 
                    monitor='val_loss',
                    save_best_only=True,
                )
            ]
            encoder,autoencoder,decoder = ae_model(encoding_dim,node,layer_nb,encoding_dim*10)
            history=autoencoder.fit(X_profiling, X_profiling,
                            epochs=30,
                            batch_size=512,
                            shuffle=False,
                            validation_data=(X_attack, X_attack),
                           callbacks=callbacks_list)
            end=time.time() # time
            print('Temps execution = %d'%(end-start))
            
            lossy = history.history['loss']
            val_lossy = history.history['val_loss']
            np_lossy =np.array(lossy).reshape((1,len(lossy))) 
            np_val_lossy =np.array(val_lossy).reshape((1,len(val_lossy))) 

            np_out = np.concatenate([np_lossy,np_val_lossy],axis=0)
            loss_file = file_path +'/loss_ASCAD_AE(MLP)_'+str(node)+'node_'+str(encoding_dim)+'encoding_dim_'+str(layer_nb)+'layers.txt'
            np.savetxt(loss_file,np_out)    
            print("保存文件成功")
            #divide trace set
            subfile_path = './dataset'
            #sbox1~sbox16
            subfile_name = subfile_path+'/ASCAD_Sbox3_'+str(encoding_dim*10)+'_AE_'+str(node)+'node_'+str(encoding_dim)+'encoding_dim_'+str(layer_nb)+'layers.h5'

            labels_profiling = []
            labels_attack = []


            raw_traces_profiling = encoder.predict(X_profiling)
            raw_traces_attack = encoder.predict(X_attack)
            labels_profiling = Y_profiling
            labels_attack = Y_attack


            with h5py.File(subfile_name,'w') as f:
                #f.create_dataset('test_numpy',data=x)
                profiling_traces_group = f.create_group('Profiling_traces')
                attack_traces_group = f.create_group("Attack_traces")
                # Datasets in the groups
                profiling_traces_group.create_dataset(name="traces", data=raw_traces_profiling)
                attack_traces_group.create_dataset(name="traces", data=raw_traces_attack)
                # Labels in the groups
                profiling_traces_group.create_dataset(name="labels", data=labels_profiling)
                attack_traces_group.create_dataset(name="labels", data=labels_attack)

                # meta data in the groups
                profiling_traces_group.create_dataset(name="metadata",data = Metadata_profiling)
                attack_traces_group.create_dataset(name="metadata", data=Metadata_attack)

