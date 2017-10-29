# -*- coding: utf-8 -*
"""
提供了一个类shower,本质上是个列表,可以往里面添加图片add_img(img,name),还可以将列表清空clear_imgs
提供了2个显示函数,一个是全部在一个窗口show(),一个是用类似ppt的方式来显示show_ppt()
依赖于logconfig模块
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import logconfig
import logging
from logconfig import timelog

logger = logging.getLogger("state_track")


class shower:
    def __init__(self):
        self.img_array = []

    def clear_imgs(self):
        self.img_array = []
        return self

    @timelog
    def add_img(self, img, img_name='noname'):
        self.img = img
        if len(img.shape) == 3:
            logger.info(img_name + " added as color image.")
            self.img_array.append({
                'name': img_name,
                'img': img,
                'type': 'color'
            })
            return self
        elif len(img.shape) == 2:
            logger.info(img_name + " added as gray image.")
            self.img_array.append({
                'name': img_name,
                'img': img,
                'type': 'gray'
            })
            return self
        else:
            logger.error(img_name + " can not handle correctly")
            return 1

    def numOfImages(self):
        return len(self.img_array)

    @timelog
    def show(self):
        num = self.numOfImages()

        if num < 4:
            row = 1
            col = num
        elif num < 9:
            row = 2
            col = num / 2
            if col * 2 != num:
                col = col + 1
        elif num == None:
            logger.error("num is None ")
        else:
            logger.error("too many pictures ,maybe you forget clear ")

        counter =1
        for data in self.img_array:
            plt.subplot(row, col, counter)
            plt.title(data['name'])
            if data['type'] == 'gray':
                plt.imshow(data['img'], 'gray')

            else:
                b, g, r = cv2.split(data['img'])
                plt.imshow(cv2.merge([r, g, b]))
            logger.info(data['name'] + " loaded")
            counter = counter+1
        plt.show()

    def show_ppt(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        for data in self.img_array:
            cv2.imshow('image',data['img'])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import res
    import unittest

    class Testshower(unittest.TestCase):
        def setUp(self):
            self.o = shower()
            self.o.add_img(res.zheng, 'zheng')
            self.o.add_img(res.xie, 'xie')

        def tearDown(self):
            # self.o = None
            pass

        def test_numOfImages(self):
            self.assertEqual(self.o.numOfImages(), 2)

        def test_add_img(self):
            self.assertEqual(self.o.img_array[1]['name'], 'xie')
            self.assertEqual(self.o.img_array[1]['type'], 'color')
            self.assertEqual(
                self.o.add_img(res.xie, 'xie self test'), self.o)

        def test_img_clear(self):
            self.assertEqual(self.o.clear_imgs(), self.o)
            self.assertEqual(len(self.o.img_array), 0)

        def test_show(self):
            ''' you need to see if the pictures show as you wish '''
            self.o.show()
        def test_show_ppt(self):
            self.o.show_ppt()

    unittest.main()
