#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np
import cv2
from matplotlib import pyplot as plt
import logconfig
import logging
from shower import shower
from logconfig import timelog
from tools import *
from math import *

logger = logging.getLogger("state_track")


@timelog
def get_outer_frame(_img):
    this_img = _img.copy()

    contours, hierarchy = cv2.findContours(this_img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    c_area = [cv2.contourArea(c) for c in contours]
    max_area_c = contours[c_area.index(max(c_area))]

    if max_area_c is None:
        logger.error("can not find a contours,try to change the args")

    cv2.drawContours(this_img, [max_area_c], -1, (0, 255, 0), 9)
    return max_area_c


@timelog
def get_pole_point(_frame):
    _c = _frame
    _poles = {}

    _poles['left'] = tuple(_c[_c[:, :, 0].argmin()][0])
    _poles['right'] = tuple(_c[_c[:, :, 0].argmax()][0])
    _poles['top'] = tuple(_c[_c[:, :, 1].argmin()][0])
    _poles['bot'] = tuple(_c[_c[:, :, 1].argmax()][0])

    logger.info('poles ' + str(_poles))
    return _poles


@timelog
def perspective(_img, _poles, _x, _y):
    pts1 = np.float32(
        [_poles['top'], _poles['right'], _poles['left'], _poles['bot']])
    pts2 = np.float32([[0, 0], [_x, 0], [0, _y], [_x, _y]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(_img, M, (_x, _y))

    return dst


@timelog
def color_pos(_img, color_name):
    logger.info("finding the " + color_name + "object's position")
    c = get_outer_frame(threshold(gray(pick_color(_img, color_name)), 30))
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


@timelog
def grid_map(tgray_img, size):
    if len(tgray_img.shape) == 3:
        logger.error(
            "in grid_map, maybe you put a color image as the gray_img")
    return tgray_img[::size, ::size]


def pplength((x1,y1),(x2,y2)):
    return sqrt(pow(x1-x2,2)+pow(y1-y2,2))




if __name__ == '__main__':
    import res
    import sys

    if len(sys.argv)!=1:
        target = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
    else:
        target=res.xie

    s = shower()

    res_target=target.copy()
    shrink = cv2.resize(res_target, (600,450), interpolation=cv2.INTER_CUBIC)
    s.add_img(shrink,'shrink')

    target=shrink
    #轮廓分析

    g = gray(pick_color(target,'carid_blue'))
    fg = filt(g, 10)
    tfg = threshold(fg,15)


    s.add_img(g,'g')
    s.add_img(fg,'fg')
    s.add_img(tfg,'tfg')

    frame = get_outer_frame(tfg)
    poles = get_pole_point(frame)

    cv2.drawContours(target, frame, cv2.CHAIN_APPROX_SIMPLE, (0,255,0), 9)

    s.add_img(target,'draw')

    #车牌方向分析

    rows,cols = target.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(frame, cv2.cv.CV_DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    line_target=target.copy()
    cv2.line(line_target,(cols-1,righty),(0,lefty),(0,255,0),2)
    s.add_img(line_target)
    mytan=abs(float(righty-lefty)/(cols-1))

    if mytan < 0.75:
        s.show()
        s.clear_imgs()
        (h,w) = target.shape[:2]
        center = (w / 2,h / 2)

        #旋转45度，缩放0.75
        _M = cv2.getRotationMatrix2D(center,45,0.65)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
        target = cv2.warpAffine(target,_M,(w,h))

        #重新进行轮廓分析

        s.add_img(target)

        g = gray(pick_color(target,'carid_blue'))
        fg = filt(g, 8)
        tfg = threshold(fg,15)


        s.add_img(g,'g')
        s.add_img(fg,'fg')
        s.add_img(tfg,'tfg')

        frame = get_outer_frame(tfg)
        poles = get_pole_point(frame)

    s.add_img(
        draw_point(
            target,
            [poles['top'], poles['right'], poles['left'], poles['bot']]),
        'draw_poels')

    #让图像翻转成正常样子

    poles2={}
    poles2['right'] = poles['top']
    poles2['top'] = poles['left']
    poles2['left'] = poles['bot']
    poles2['bot'] = poles['right']

    std = perspective(target, poles2, 220,70)
    s.add_img(std, 'std')


        
    s.show()
    s.clear_imgs()

    #彩色车牌图像识别

    g=gray(std)
    fg=filt(g,4)
    tg=threshold(g,175)
    ftg=filt(tg,4)
    tftg=threshold(ftg,100)

    s.add_img(g,'g')
    s.add_img(fg,'ft')
    s.add_img(tg,'tg')
    s.add_img(ftg,'ftg')
    s.add_img(tftg,'tftg')
    s.add_img(tftg[20:50,64:76],'point')
    s.show()
    s.clear_imgs()

    """
    _c = get_outer_frame(tftg[20:50,64:76])
    _M = cv2.moments(_c)
    cX = int(_M["m10"] / _M["m00"])
    cY = int(_M["m01"] / _M["m00"])
    Pcenter= (cX, cY)
    Xoffset=cX-6
    Yoffset=int((35-cY)/4.5)
    """

    def up_limit(num,limit):
        if num>limit:
            return limit
        else: 
            return num

    def low_limit(num,limit):
        if num<limit:
            return limit
        else: 
            return num

    def cal_diff(hu1,hu2):
        return np.linalg.norm(hu1-hu2)

    Xoffset=0
    Yoffset=0
    if Xoffset <0:
        Xoffset=Xoffset-1
    if Xoffset >0:
        Xoffset=Xoffset+1

    first_word_img=fg[low_limit(6+Yoffset,0):up_limit(64+Yoffset,70),0:34+Xoffset]
    second_word_img=fg[low_limit(6+Yoffset,0):up_limit(64+Yoffset,70),34+Xoffset:64+Xoffset]
    third_word_img=fg[low_limit(6+Yoffset,0):up_limit(64+Yoffset,70),76+Xoffset:106+Xoffset]
    forth_word_img=fg[low_limit(6+Yoffset,0):up_limit(64+Yoffset,70),106+Xoffset:135+Xoffset]
    fifth_word_img=fg[low_limit(6+Yoffset,0):up_limit(64+Yoffset,70),135+Xoffset:164+Xoffset]
    sixth_word_img=fg[low_limit(6+Yoffset,0):up_limit(64+Yoffset,70),164+Xoffset:193+Xoffset]
    seventh_word_img=fg[low_limit(6+Yoffset,0):up_limit(64+Yoffset,70),193+Xoffset:]


    img_string=[first_word_img,second_word_img,third_word_img,forth_word_img,fifth_word_img,sixth_word_img,seventh_word_img]
    hu_string=[]
    for _img in img_string:
       hu_string.append(np.log10(np.abs(cv2.HuMoments(cv2.moments(_img),True).flatten()))) 


    s.add_img(first_word_img,'1')
    s.add_img(second_word_img,'2')
    s.add_img(third_word_img,'3')
    s.add_img(forth_word_img,'4')
    s.add_img(fifth_word_img,'5')
    s.add_img(sixth_word_img,'6')
    s.add_img(seventh_word_img,'7')
#    s.add_img(tftg[20:50,62:78],'2')




    s.show()

"""



    side_top=pplength(poles['left'],poles['top'])
    side_bot=pplength(poles['bot'],poles['right'])
    side_left=pplength(poles['left'],poles['bot'])
    side_right=pplength(poles['top'],poles['right'])

    closety=sqrt(pow(side_top-side_bot,2)+pow(side_left-side_right,2))/pplength(target.shape[:2],(0,0))

        id_long=poles['right'][0]-poles['left'][0]
        id_wide=poles['bot'][1]-poles['top'][1]

        s.add_img(
            draw_point(target, [color_pos(target, 'carid_blue')]),
            'center')

        minLineLength = 100
        maxLineGap =8 
        edges=cv2.Canny(tfg,10,20)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,70,minLineLength,maxLineGap)

        line_target=target.copy()
        for x1,y1,x2,y2 in lines[0]:
                cv2.line(line_target,(x1,y1),(x2,y2),(0,0,255),25)

        s.add_img(line_target,'lines')
    #tg = threshold(g, 70)
    #ftg = filt(tg, 10)
    #tftg = threshold(ftg, 100)

    s = shower()
    s.add_img(edges,'edges')
    
    zheng2=res.zheng.copy()

    zheng3=res.zheng.copy()
    zheng4=res.zheng.copy()
    #lines = cv2.HoughLines(edges,1,np.pi/180,180)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 4000*(-b))
        y1 = int(y0 + 4000*(a))
        x2 = int(x0 - 4000*(-b))
        y2 = int(y0 - 4000*(a))
        cv2.line(zheng2,(x1,y1),(x2,y2),(0,0,255),2)


    x,y,w,h = cv2.boundingRect(frame)
    zheng4 = cv2.rectangle(zheng4,(x,y),(x+w,y+h),(0,255,0),2)







    #s.add_img(ftg,'ftg')
    #s.add_img(tftg,'tftg')

    s.show()
    s.clear_imgs()
    s.add_img(zheng2)
    s.show()

    frame = get_outer_frame(tftg)
    poles = get_pole_point(frame)
    s.add_img(
        draw_point(
            res.xie,
            [poles['top'], poles['right'], poles['left'], poles['bot']]),
        'draw_poels')

    std = perspective(res.xie, poles, 150, 200)
    s.add_img(
        draw_point(res.xie, [color_pos(res.xie, 'red')]),
        'find the red spot position')

    s.add_img(std, 'std')
    s.add_img(tg, 'tg')
    s.add_img(res.xie, 'original')
    s.add_img(tftg, 'tftg')

    row = 0
    col = 0
    scale = 15
    waypoint = []


    gmap = grid_map(threshold(gray(res.xie), 15), scale)
    x_num = len(gmap[0])
    y_num = len(gmap)
    print(str(gmap))
    for xx in range(x_num):
        for yy in range(y_num):
            if gmap[yy,xx] > 250:
                waypoint.append((scale * xx, scale * yy))
                logger.info(str((scale*xx,scale*yy)))

    s.add_img(draw_point(res.xie, waypoint), 'waypoint')

    s.show()
    """
