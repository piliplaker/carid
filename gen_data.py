import json
import os
import numpy as np
import cv2
import matplotlib  
import matplotlib.pyplot as plt  



x_list=[]
y_list=[]

for folder in os.listdir('./charSamples'):
    zero_json=[]
    datafilename='./'+'res_json/'+folder
    for filename in os.listdir('./charSamples/'+folder):
        img=cv2.imread('./charSamples/'+folder+'/'+filename,0)
        bak_img=img.copy()
#        cv2.imshow("test",img)
#        cv2.waitKey (0)
        img=cv2.resize(img, (8,16), interpolation=cv2.INTER_CUBIC)
#        cv2.imshow("test",img)
#        cv2.waitKey (0)
        x_sobel=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        x_sobel=abs(x_sobel)
        #x_sobel=cv2.resize(x_sobel, (100,200), interpolation=cv2.INTER_CUBIC)
#        cv2.imshow("test",x_sobel)
#        cv2.waitKey (0)
        x=np.sum(x_sobel)
        #x_list.append(x)
        
        y_sobel=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        y_sobel=abs(y_sobel)
#        cv2.imshow("test",y_sobel)
#        cv2.waitKey (0)
        y=np.sum(y_sobel)
        #y_list.append(x)

        ttt=0
        img_frag=[]
        sum_frag=[]
        feature_vector=[]
        for x_num in range(1,5):
            for y_num in range(1,3):
                x_num_pre=x_num-1
                y_num_pre=y_num-1
                ttt=ttt+1
                img_frag.append(img[x_num_pre*4:x_num*4,y_num_pre*4:y_num*4])
                sum_frag.append(np.sum(img[x_num_pre*4:x_num*4,y_num_pre*4:y_num*4]))
                feature_vector.append(np.sum(x_sobel[x_num_pre*4:x_num*4,y_num_pre*4:y_num*4])/x)
                feature_vector.append(np.sum(y_sobel[x_num_pre*4:x_num*4,y_num_pre*4:y_num*4])/y)
        for ele in np.array(cv2.resize(bak_img, (4,8), interpolation=cv2.INTER_CUBIC),dtype='float32').flat:
            feature_vector.append(ele/1000)
        zero_json.append(feature_vector)

    datafile=open(datafilename,'w')
    json.dump(zero_json,datafile)


