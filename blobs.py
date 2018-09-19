import cv2
import numpy as np
import os

class DataExtractor():

    def __init__(self, infolder, outfolder, dataname):
        self.supported_files = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tif')
        self.datapath = os.path.join(outfolder, dataname)
        self.__dict__.update(locals())
        self.index = 0

    def get_rois(self):
        for root, dirs, files in os.walk(self.infolder):
            image_paths = [file for file in files if file.lower().endswith(self.supported_files)]
            for impath in image_paths:
                self.get_rois_from_img(impath)

    def get_rois_from_img(self, path):
        img = cv2.imread(path)
        rois = []
        rect = cv2.selectROI('Select Region', img, showCrosshair=False, fromCenter=False)
        while rect != (0, 0, 0, 0):
            x, y, w, h = rect
            roi = img[x:x+w, y:y+h]
            cv2.imwrite(self.datapath+'_{}.png'.format(self.index), roi)
            self.index += 1
            rect = cv2.selectROI('Select Region', img, showCrosshair=False, fromCenter=False)



if __name__ == '__main__':
    img = cv2.imread('scanned_books/pngs/turbo_left/turbo_left (4).png')
    img2 = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rect = cv2.selectROI(img2)
    print(rect)

    cv2.imshow('gray', img2)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()

    ret, thresh = cv2.threshold(img2, 235, 255, 0)
    cv2.imshow('threshold', thresh)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()
    
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    # print(len(contours))
    print(len(cnt))
    cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
    cv2.imshow('contours', img)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()
    # print(contours)

    
