import numpy as np
np.random.seed(10000)
import imp
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
parser.add_argument('-scenario',default='full')
parser.add_argument('-adv',default='adv1')
parser.add_argument('-version',default='v0')
args = parser.parse_args()
dataset=args.dataset 
input_data=input_data_class.InputData(dataset=dataset)
config = configparser.ConfigParser()
config.read('config.ini')


user_label_dim=int(config[dataset]["num_classes"])
num_classes=1
save_model=True
epochs=int(config[dataset]["attack_epochs"])
user_epochs=int(config[dataset]["user_epochs"])
attack_epochs=int(config[dataset]["attack_shallow_model_epochs"])
batch_size=int(config[dataset]["defense_batch_size"])
defense_train_testing_ratio=float(config[dataset]["defense_training_ratio"])
result_folder=config[dataset]["result_folder"]
network_architecture=str(config[dataset]["network_architecture"])
fccnet=imp.load_source(str(config[dataset]["network_name"]),network_architecture)

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
config_gpu.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config_gpu))




#########loading defense data###################
(x_evaluate,y_evaluate,l_evaluate)=input_data.input_data_attacker_evaluate()
evaluation_noise_filepath=result_folder+"/attack/"+"noise_data_evaluation.npz"
print(evaluation_noise_filepath)
if not os.path.isfile(evaluation_noise_filepath):
    raise FileNotFoundError
npz_defense=np.load(evaluation_noise_filepath)
f_evaluate_noise=npz_defense['defense_output']
f_evaluate_origin=npz_defense['tc_output']



f_evaluate_defense=np.zeros(f_evaluate_noise.shape,dtype=np.float)
np.random.seed(100)  #one time randomness, fix the seed
for i in np.arange(f_evaluate_defense.shape[0]):
    if np.random.rand(1)<0.5:
        f_evaluate_defense[i,:]=f_evaluate_noise[i,:]
    else:
        f_evaluate_defense[i,:]=f_evaluate_noise[i,:]



##########load attacker's shadow model#################

(x_train,y_train,l_train) =input_data.input_data_attacker_adv1()
y_train=keras.utils.to_categorical(y_train,user_label_dim)
npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_attack_shallow_model_{}.npz".format(user_epochs,args.adv))
weights=npzdata['x']
input_shape=x_train.shape[1:]
model=fccnet.model_user(input_shape=input_shape,labels_dim=user_label_dim)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
model.set_weights(weights)
f_train=model.predict(x_train)
del model


f_train=np.sort(f_train,axis=1)
f_evaluate_defense=np.sort(f_evaluate_defense,axis=1)
f_evaluate_origin=np.sort(f_evaluate_origin,axis=1)


##########################

if args.scenario=='full':
    b_train=f_train[:,:]
    b_test=f_evaluate_defense[:,:]
    b_test_origin=f_evaluate_origin[:,:]
else:
    raise NotImplementedError
label_train=l_train
label_test=l_evaluate
############define attack model#####################
input_shape=b_train.shape[1:]
model=fccnet.model_attack_nn(input_shape=input_shape,labels_dim=num_classes)
model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
model.summary()



index_array=np.arange(b_train.shape[0])
batch_num=np.int(np.ceil(b_train.shape[0]/batch_size))
for i in np.arange(epochs):
    np.random.shuffle(index_array)
    for j in np.arange(batch_num):
        b_batch=b_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,b_train.shape[0])],:]
        y_batch=label_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,label_train.shape[0])]]
        model.train_on_batch(b_batch,y_batch)   

    if (i+1)%300==0:
        K.set_value(model.optimizer.lr,K.eval(model.optimizer.lr*0.1))
        print("Learning rate: {}".format(K.eval(model.optimizer.lr)))
    if (i+1)%100==0:
        print("Epochs: {}".format(i))
        scores_test_defense = model.evaluate(b_test, label_test, verbose=0)
        print('Test loss defense:', scores_test_defense[0])
        print('Test accuracy defense:', scores_test_defense[1])  
        scores_test_nodefense = model.evaluate(b_test_origin, label_test, verbose=0)
        print('Test loss no defense:', scores_test_nodefense[0])
        print('Test accuracy no defense:', scores_test_nodefense[1])  
        scores_train = model.evaluate(b_train, label_train, verbose=0)
        print('Train loss:', scores_train[0])
        print('Train accuracy:', scores_train[1])  

result_filepath=result_folder+"/"+config[dataset]["result_file_publish"]

print(result_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
if not os.path.isfile(result_filepath):
    fp=open(result_filepath,'w+')
    fp.close()
###########evaluate the attack###################

evaluation_noise_filepath=result_folder+"/attack/"+"noise_data_evaluation.npz"
if not os.path.isfile(evaluation_noise_filepath) :
    raise FileNotFoundError
npz_defense=np.load(evaluation_noise_filepath)
f_evaluate_noise=npz_defense['defense_output']
f_evaluate_origin=npz_defense['tc_output']
f_evaluate_origin_score=npz_defense['predict_origin']
f_evaluate_defense_score=npz_defense['predict_modified']


if args.scenario=='full':
    predict_result_origin=np.where(model.predict(np.sort(f_evaluate_origin,axis=1)) > 0.5, 1, 0)
    predict_result_defense=np.where(model.predict(np.sort(f_evaluate_noise,axis=1)) > 0.5, 1, 0)
else:
    raise NotImplementedError

epsilon_value_list=["1.0","0.7","0.5","0.3","0.1","0.0"]
epsilon_value_list=[float(t) for t in epsilon_value_list]

inference_accuracy_list=[]

for epsilon_value in epsilon_value_list: 
    inference_accuracy=0.0

    np.random.seed(100)  
    for i in np.arange(f_evaluate_origin.shape[0]):
        distortion_noise=np.sum(np.abs(f_evaluate_origin[i,:]-f_evaluate_noise[i,:]))
        p_value=0.0
        if np.abs(f_evaluate_origin_score[i]-0.5)<=np.abs(f_evaluate_defense_score[i]-0.5):
            p_value=0.0
        else:
            p_value=min(epsilon_value/distortion_noise,1.0)

        if predict_result_origin[i]==label_test[i]:
            inference_accuracy+=1.0-p_value
        if predict_result_defense[i]==label_test[i]:
            inference_accuracy+=p_value
    inference_accuracy_list.append(inference_accuracy/(float(f_evaluate_origin.shape[0])))


print("Budget list: {}".format(epsilon_value_list))
print("inference accuracy list: {}".format(inference_accuracy_list))       
