import numpy as np
import imp
np.random.seed(1000)
import input_data_class
import keras
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import tensorflow as tf
import os
import configparser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dataset',default='location')
parser.add_argument('-adv',default='adv1')
args = parser.parse_args()
dataset=args.dataset 
input_data=input_data_class.InputData(dataset=dataset)
config = configparser.ConfigParser()
config.read('config.ini')


user_label_dim=int(config[dataset]["num_classes"])
num_classes=int(config[dataset]["num_classes"])
save_model=True

user_epochs=int(config[dataset]["user_epochs"])
batch_size=int(config[dataset]["attack_shallow_model_batch_size"])
defense_train_testing_ratio=float(config[dataset]["defense_training_ratio"])
result_folder=config[dataset]["result_folder"]
network_architecture=str(config[dataset]["network_architecture"])
fccnet=imp.load_source(str(config[dataset]["network_name"]),network_architecture)


config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
config_gpu.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config_gpu))


(x_train,y_train),(x_test,y_test) =input_data.input_data_attacker_shallow_model_adv1()


y_train=y_train.astype(int)
y_test=y_test.astype(int)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape=x_train.shape[1:]


####### we assume the attacker know the architecture of the target model. 
model=fccnet.model_user(input_shape=input_shape,labels_dim=num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
model.summary()

print(x_train.shape)
print(x_test.shape)
index_array=np.arange(x_train.shape[0])
batch_num=np.int(np.ceil(x_train.shape[0]/batch_size))
for i in np.arange(user_epochs):
    np.random.shuffle(index_array)
    for j in np.arange(batch_num):
        x_batch=x_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,x_train.shape[0])],:]
        y_batch=y_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,x_train.shape[0])],:]
        model.train_on_batch(x_batch,y_batch)   
    if (i+1)%150==0:
        K.set_value(model.optimizer.lr,K.eval(model.optimizer.lr*0.1))
        print("Learning rate: {}".format(K.eval(model.optimizer.lr)))
    if (i+1)%100==0:
        print("Epochs: {}".format(i))
        scores_test = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', scores_test[0])
        print('Test accuracy:', scores_test[1])  
        scores_train = model.evaluate(x_train, y_train, verbose=0)
        print('Train loss:', scores_train[0])
        print('Train accuracy:', scores_train[1])  


if save_model:
    weights=model.get_weights()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(result_folder+"/models"):
        os.makedirs(result_folder+"/models")
    np.savez(result_folder+"/models/"+"epoch_{}_weights_attack_shallow_model_{}.npz".format(user_epochs,args.adv),x=weights)