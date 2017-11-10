import cv2 
import numpy as np
import os
import json

ann=cv2.ANN_MLP()
ann.load('./ann_trained3')

def guess(_img):
    img=cv2.resize(_img, (8,16), interpolation=cv2.INTER_CUBIC)
    x_sobel=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    x_sobel=abs(x_sobel)
    x=np.sum(x_sobel)
    y_sobel=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    y_sobel=abs(y_sobel)
    y=np.sum(y_sobel)
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
    for ele in np.array(cv2.resize(img, (4,8), interpolation=cv2.INTER_CUBIC),dtype='float32').flat:
        feature_vector.append(ele/1000)

    input_data=np.zeros((1,48))
    input_data[0]=feature_vector

    temp=ann.predict(input_data)[1][0]
    temp=np.ndarray.tolist(temp)

    return (temp.index(max(temp)),temp)

if __name__ == '__main__':
    zero=cv2.imread('./charSamples/0/24_0.876098_gray_12240_4810_step5_recog_6_0_0.960593_0.841573.png', cv2.IMREAD_GRAYSCALE)
    one=cv2.imread('./charSamples/1/47_0.810739_gray_5529_2125_step5_recog_6_1_0.975752_0.791080.png', cv2.IMREAD_GRAYSCALE)
    two=cv2.imread('./charSamples/2/43_0.918763_gray_623_414_step5_recog_6_2_0.989849_0.909437.png', cv2.IMREAD_GRAYSCALE)
    three=cv2.imread('./charSamples/3/30_0.988141_gray_12674_5689_step5_recog_5_3_0.998670_0.986827.png', cv2.IMREAD_GRAYSCALE)
    four=cv2.imread('./charSamples/4/21_0.953300_gray_2313_1192_step5_recog_3_4_0.994085_0.947662.png', cv2.IMREAD_GRAYSCALE)
    five=cv2.imread('./charSamples/5/21_0.978021_gray_4921_2072_step5_recog_4_5_0.996531_0.974628.png', cv2.IMREAD_GRAYSCALE)
    six=cv2.imread('./charSamples/6/2_0.934481_gray_11876_5333_step5_recog_6_6_0.992466_0.927441.png', cv2.IMREAD_GRAYSCALE)
    seven=cv2.imread('./charSamples/7/30_0.958893_gray_7385_4736_step5_recog_6_7_0.992492_0.951694.png', cv2.IMREAD_GRAYSCALE)
    eight=cv2.imread('./charSamples/8/26_0.945086_gray_34068_14407_step5_recog_2_8_0.980481_0.926639.png', cv2.IMREAD_GRAYSCALE)
    nine=cv2.imread('./charSamples/9/26_0.973152_gray_33964_14350_step5_recog_4_9_0.994884_0.968173.png', cv2.IMREAD_GRAYSCALE)
    letterA=cv2.imread('./charSamples/A/18_0.918068_gray_14532_5759_step5_recog_3_A_0.982697_0.902183.png', cv2.IMREAD_GRAYSCALE)
    print(guess(zero)[0])
    print(guess(one)[0])
    print(guess(two)[0])
    print(guess(three)[0])
    print(guess(four)[0])
    print(guess(five)[0])
    print(guess(six)[0])
    print(guess(seven)[0])
    print(guess(eight)[0])
    print(guess(nine)[0])
    print(guess(letterA)[0])
    
