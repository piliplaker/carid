# -*- coding: utf-8 -*
"""
cv2提供的处理函数使用起来太过繁琐,为了方便较高层次的操作,对一些简单的图像处理进行一些封装
draw_grid 画线,方便
draw_point 图片上画点
threshold 灰度图操作,
filt
rgray
ggray
bgray
gray
pick_color 在图像上只显示指定颜色的像素点,比如blue,yellow,red,green等,注意hsv中后两项的范围,这应该在不同的情况中有不同的指定

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import logconfig
import logging
from logconfig import timelog

logger = logging.getLogger("state_track")


@timelog
def draw_grid(_img, side):
    y, x, _ = _img.shape

    this_img = np.zeros((y, x, 3), dtype=np.uint8)

    for _x in range(0, x, side):
        cv2.line(this_img, (_x, 0), (_x, y), [120, 120, 120], thickness=1)
    for _y in range(0, y, side):
        cv2.line(this_img, (0, _y), (x, _y), [120, 120, 120], thickness=1)

    logger.info("draw grid with " + str(side) + " side")
    return cv2.subtract(_img, this_img)


@timelog
def draw_point(_img, points):
    this_img = _img.copy()
    for _p in points:
        cv2.circle(this_img, _p, _img.shape[0] / 350, (120, 0, 255), -1)
        logger.info("draw a point at " + str(_p))
    return this_img


@timelog
def threshold(_img, th):
    _, res = cv2.threshold(_img, th, 255, cv2.THRESH_BINARY)
    logger.info("threshold with " + str(th))
    return res


@timelog
def filt(_img, a):
    kernel = np.ones((a, a), np.float32) / (a * a)
    dst = cv2.filter2D(_img, -1, kernel)
    logger.info("filt with " + str(a) + " side")
    return dst


@timelog
def rgray(_img):
    _, _, r = cv2.split(_img)
    return r


@timelog
def ggray(_img):
    _, g, _ = cv2.split(_img)
    return g


@timelog
def bgray(_img):
    b, _, _ = cv2.split(_img)
    return b


@timelog
def gray(_img):
    return cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)


@timelog
def pick_color(_img, color_name):
    if color_name == 'blue':
        color_bgr = [255, 0, 0]
        myrange=20
    if color_name == 'red':
        color_bgr = [0, 0, 255]
        myrange=20
    if color_name == 'green':
        color_bgr = [0, 255, 0]
        myrange=20
    if color_name == 'yellow':
        color_bgr = [255, 255, 0]
        myrange=20
    if color_name == 'carid_blue':
        color_bgr = [133, 66, 5]
        myrange=19
    if color_name == 'carid_white':
        color_bgr = [205, 205, 205]
        myrange=30


    object_color = np.uint8([[color_bgr]])
    hsv_object_color = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)

    hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)

    _range=myrange
    if hsv_object_color[0][0][0]< myrange:
        _range = hsv_object_color[0][0][0]
    lower = np.array([hsv_object_color[0][0][0] - _range, 215, 20])
    _range=myrange
    if hsv_object_color[0][0][0]+myrange>255:
        _range = 255-hsv_object_color[0][0][0]
    upper = np.array([hsv_object_color[0][0][0] + _range, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(_img, _img, mask=mask)

    return res


def erode(_img, size):
    return cv2.erode(_img, np.ones((size, size)))


if __name__ == '__main__':
    from shower import shower
    import res

    def pick_show(color_bgr):
        object_color = np.uint8([[color_bgr]])
        hsv_object_color = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)

        return hsv_object_color 


    show = shower()

    show.add_img(res.zheng, 'original')
    show.add_img(rgray(res.zheng), 'r channel')
    show.add_img(ggray(res.zheng), 'g channel')
    show.add_img(bgray(res.zheng), 'b channel')
    show.add_img(gray(res.zheng), 'gray')

    show.show()
    show.clear_imgs()

    show.add_img(res.zheng, 'original')
    show.add_img(threshold(bgray(res.zheng), 120), 'th b chan')
    show.add_img(filt(res.zheng, 10), 'f origin')
    show.add_img(draw_point(res.zheng, [(150, 150)]), 'point')
    show.add_img(draw_grid(res.zheng, 10), 'grid')
    show.add_img(pick_color(res.xie, 'carid_blue'), 'pick_color')
    show.add_img(erode(gray(res.xie), 10))

    show.show()
    show.clear_imgs()

    show.add_img(res.one, 'one')
    show.add_img(res.two, 'two')
    show.add_img(res.three, 'three')
    show.add_img(res.four, 'four')
    show.add_img(res.five, 'five')
    show.show()

    
