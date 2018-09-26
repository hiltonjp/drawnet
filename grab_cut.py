import cv2
import numpy as np
import os

class GrabCutter():

    def __init__(self, infolder, outfolder, dataname):
        self.__dict__.update(locals())
        self.supported_files = ('.png','.jpg','.jpeg','.bmp','.gif','tif')



    def cut_data(self):
        for root, dirs, files in os.walk(self.outfolder):
            image_paths = [os.path.join(root, file) for file in files if file.lower().endswith(self.supported_files)]

            for impath in image_paths:
                self.grabcut_image(impath)


    def grabcut_image(self, path):
        img = cv2.imread(path)


def mouse_event(event, x, y, flags, param):
    actions = {
        cv2.EVENT_LBUTTONDOWN: draw_foreground,
        cv2.EVENT_RBUTTONDOWN: draw_background,

    }

def draw_foreground():


def draw_background():


if __name__ == '__main__':
    img =
    mask = np.zeros(512,512,3)