from __future__ import print_function
import keras
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, concatenate
import numpy as np
np.random.seed(10000)


def model_user(input_shape,labels_dim):
    inputs=Input(shape=input_shape)
    middle_layer=Dense(1024,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs)
    middle_layer=Dense(512,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    middle_layer=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(middle_layer)
    outputs_logits=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(middle_layer)
    outputs=Activation('softmax')(outputs_logits)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def model_defense(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


def model_defense_optimize(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Activation('softmax')(inputs_b)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model


def model_attack_nn(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(512,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(inputs_b)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model   