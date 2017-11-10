import cv2
import numpy as np
import os
import json

files=os.listdir('./res_json')
files.sort()

input_matrix=[]
output_matrix=[]
test_data=[]

for target in files:
    x_vector=json.load(open('./'+'res_json/'+target,'r'))
    position=files.index(target)
    out_vector_temp=np.zeros((1,34))
    out_vector_temp[0,position]=10000
    ot=out_vector_temp
    t_data=np.zeros((1,48))
    t_data[0]=x_vector[0] #0
    test_data.append(t_data)
    print(len(x_vector))
    for _ in range(len(x_vector)-1):
        out_vector_temp=np.vstack((out_vector_temp,ot))
        
    if len(input_matrix)==0:
        input_matrix=x_vector
        output_matrix=out_vector_temp
    else:
        input_matrix=np.vstack((input_matrix,x_vector))
        output_matrix=np.vstack((output_matrix,out_vector_temp))

input_matrix = np.array(input_matrix, dtype='float32')
output_matrix = np.array(output_matrix, dtype='float32')
ann = cv2.ANN_MLP()
ann.create(np.array([48,60,60,60, 34]))

#ann.load('./ann_trained')
test0_data=np.zeros((1,48))
test0_data[0]=input_matrix[0] #0
test1_data=np.zeros((1,48))
test1_data[0]=input_matrix[50] #0
test2_data=np.zeros((1,48))
test2_data[0]=input_matrix[100] #0
test3_data=np.zeros((1,48))
test3_data[0]=input_matrix[150] #0

params = dict( term_crit = ( cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 100000, 0.0001),  
	       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,   
	       bp_dw_scale = 0.00002,  
	       bp_moment_scale = 0.00002 )  

#ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)  
#ann.setBackpropWeightScale(0.1)  
#ann.setBackpropMomentumScale(0.1)  

