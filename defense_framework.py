import numpy as np
np.random.seed(1000)
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
from scipy.special import softmax



config = configparser.ConfigParser()
parser = argparse.ArgumentParser()
parser.add_argument('-qt',type=str,default='evaluation')
parser.add_argument('-dataset',default='location')
args = parser.parse_args()
dataset=args.dataset 
input_data=input_data_class.InputData(dataset=dataset)
config = configparser.ConfigParser()
config.read('config.ini')


user_label_dim=int(config[dataset]["num_classes"])
num_classes=1


user_epochs=int(config[dataset]["user_epochs"])
defense_epochs=int(config[dataset]["defense_epochs"])
result_folder=config[dataset]["result_folder"]
network_architecture=str(config[dataset]["network_architecture"])
fccnet=imp.load_source(str(config[dataset]["network_name"]),network_architecture)

print("Config: ")
print("dataset: {}".format(dataset))
print("result folder: {}".format(result_folder))
print("network architecture: {}".format(network_architecture))


config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
config_gpu.gpu_options.visible_device_list = "0"
#set_session(tf.Session(config=config))

sess = tf.InteractiveSession(config=config_gpu)
sess.run(tf.global_variables_initializer())



print("Loading Evaluation dataset...")
(x_evaluate,y_evaluate,l_evaluate) =input_data.input_data_attacker_evaluate()




print("Loading target model...")
npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_user.npz".format(user_epochs))

######load target model##############
weights=npzdata['x']
input_shape=x_evaluate.shape[1:]
model=fccnet.model_user(input_shape=input_shape,labels_dim=user_label_dim)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
model.set_weights(weights)
output_logits=model.layers[-2].output
f_evaluate=model.predict(x_evaluate)   #confidence score result of target model on evaluation dataset
f_evaluate_logits=np.zeros([1,user_label_dim],dtype=np.float)
batch_predict=100
batch_num=np.ceil(x_evaluate.shape[0]/float(batch_predict))
for i in np.arange(batch_num):
    f_evaluate_logits_temp=sess.run(output_logits,feed_dict={model.input:x_evaluate[int(i*batch_predict):int(min((i+1)*batch_predict,x_evaluate.shape[0])),:]})
    f_evaluate_logits=np.concatenate((f_evaluate_logits,f_evaluate_logits_temp),axis=0)
f_evaluate_logits=f_evaluate_logits[1:,:]  #logits of target model on evaluation dataset
del model

f_evaluate_origin=np.copy(f_evaluate)  #keep a copy of original one
f_evaluate_logits_origin=np.copy(f_evaluate_logits)
#############as we sort the prediction sscores, back_index is used to get back original scores#############
sort_index=np.argsort(f_evaluate,axis=1)
back_index=np.copy(sort_index)
for i in np.arange(back_index.shape[0]):
    back_index[i,sort_index[i,:]]=np.arange(back_index.shape[1])
f_evaluate=np.sort(f_evaluate,axis=1)
f_evaluate_logits=np.sort(f_evaluate_logits,axis=1)



print("f evaluate shape: {}".format(f_evaluate.shape))
print("f evaluate logits shape: {}".format(f_evaluate_logits.shape))

##########loading defense model


input_shape=f_evaluate.shape[1:]
print("Loading defense model...")
npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_defense.npz".format(defense_epochs))
model=fccnet.model_defense_optimize(input_shape=input_shape,labels_dim=num_classes)
model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
#model.summary()
weights=npzdata['x']
model.set_weights(weights)
model.trainable=False

########evaluate the performance of defense's attack model on undefended data########
scores_evaluate = model.evaluate(f_evaluate_logits, l_evaluate, verbose=0)
print('evaluate loss on model:', scores_evaluate[0])
print('evaluate accuracy on model:', scores_evaluate[1])  

output=model.layers[-2].output[:,0]
c1=1.0  #used to find adversarial examples 
c2=10.0    #penalty such that the index of max score is keeped
c3=0.1
#alpha_value=0.0 

origin_value_placeholder=tf.placeholder(tf.float32,shape=(1,user_label_dim)) #placeholder with original confidence score values (not logit)
label_mask=tf.placeholder(tf.float32,shape=(1,user_label_dim))  # one-hot encode that encodes the predicted label 
c1_placeholder=tf.placeholder(tf.float32)
c2_placeholder=tf.placeholder(tf.float32)
c3_placeholder=tf.placeholder(tf.float32)

correct_label = tf.reduce_sum(label_mask * model.input, axis=1)
wrong_label = tf.reduce_max((1-label_mask) * model.input - 1e8*label_mask, axis=1)


loss1=tf.abs(output)
loss2=tf.nn.relu(wrong_label-correct_label)
loss3=tf.reduce_sum(tf.abs(tf.nn.softmax(model.input)-origin_value_placeholder)) #L-1 norm

loss=c1_placeholder*loss1+c2_placeholder*loss2+c3_placeholder*loss3
gradient_targetlabel=K.gradients(loss,model.input)
label_mask_array=np.zeros([1,user_label_dim],dtype=np.float)
##########################################################
result_array=np.zeros(f_evaluate.shape,dtype=np.float)
result_array_logits=np.zeros(f_evaluate.shape,dtype=np.float)
success_fraction=0.0
max_iteration=300   #max iteration if can't find adversarial example that satisfies requirements
np.random.seed(1000)
for test_sample_id in np.arange(0,f_evaluate.shape[0]):
    if test_sample_id%100==0:
        print("test sample id: {}".format(test_sample_id))
    max_label=np.argmax(f_evaluate[test_sample_id,:])
    origin_value=np.copy(f_evaluate[test_sample_id,:]).reshape(1,user_label_dim)
    origin_value_logits=np.copy(f_evaluate_logits[test_sample_id,:]).reshape(1,user_label_dim)
    label_mask_array[0,:]=0.0
    label_mask_array[0,max_label]=1.0
    sample_f=np.copy(origin_value_logits)
    result_predict_scores_initial=model.predict(sample_f)
    ########## if the output score is already very close to 0.5, we can just use it for numerical reason
    if np.abs(result_predict_scores_initial-0.5)<=1e-5:
        success_fraction+=1.0
        result_array[test_sample_id,:]=origin_value[0,back_index[test_sample_id,:]]
        result_array_logits[test_sample_id,:]=origin_value_logits[0,back_index[test_sample_id,:]]
        continue
    last_iteration_result=np.copy(origin_value)[0,back_index[test_sample_id,:]]
    last_iteration_result_logits=np.copy(origin_value_logits)[0,back_index[test_sample_id,:]]
    success=True
    c3=0.1
    iterate_time=1
    while success==True: 
        sample_f=np.copy(origin_value_logits)
        j=1
        result_max_label=-1
        result_predict_scores=result_predict_scores_initial
        while j<max_iteration and (max_label!=result_max_label or (result_predict_scores-0.5)*(result_predict_scores_initial-0.5)>0):
            gradient_values=sess.run(gradient_targetlabel,feed_dict={model.input:sample_f,origin_value_placeholder:origin_value,label_mask:label_mask_array,c3_placeholder:c3,c1_placeholder:c1,c2_placeholder:c2})[0][0]
            gradient_values=gradient_values/np.linalg.norm(gradient_values)
            sample_f=sample_f-0.1*gradient_values
            result_predict_scores=model.predict(sample_f)
            result_max_label=np.argmax(sample_f)
            j+=1        
        if max_label!=result_max_label:
            if iterate_time==1:
                print("failed sample for label not same for id: {},c3:{} not add noise".format(test_sample_id,c3))
                success_fraction-=1.0
            break                
        if ((model.predict(sample_f)-0.5)*(result_predict_scores_initial-0.5))>0:
            if iterate_time==1:
                print("max iteration reached with id: {}, max score: {}, prediction_score: {}, c3: {}, not add noise".format(test_sample_id,np.amax(softmax(sample_f)),result_predict_scores,c3))
            break
        last_iteration_result[:]=softmax(sample_f)[0,back_index[test_sample_id,:]]
        last_iteration_result_logits[:]=sample_f[0,back_index[test_sample_id,:]]
        iterate_time+=1 
        c3=c3*10
        if c3>100000:
            break
    success_fraction+=1.0
    result_array[test_sample_id,:]=last_iteration_result[:]
    result_array_logits[test_sample_id,:]=last_iteration_result_logits[:]
print("Success fraction: {}".format(success_fraction/float(f_evaluate.shape[0])))

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
if not os.path.exists(result_folder+"/attack"):
    os.makedirs(result_folder+"/attack")
del model 


input_shape=f_evaluate.shape[1:]
print("Loading defense model...")
npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_defense.npz".format(defense_epochs))
model=fccnet.model_defense(input_shape=input_shape,labels_dim=num_classes)

weights=npzdata['x']
model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
model.set_weights(weights)
model.trainable=False
predict_origin=model.predict(np.sort(f_evaluate_origin,axis=1))
predict_modified=model.predict(np.sort(result_array,axis=1))
np.savez(result_folder+"/attack/"+"noise_data_{}.npz".format(args.qt),defense_output=result_array,defense_output_logits=result_array_logits,tc_output=f_evaluate_origin,tc_output_logits=f_evaluate_logits_origin,predict_origin=predict_origin,predict_modified=predict_modified)
