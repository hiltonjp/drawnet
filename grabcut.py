import cv2
import numpy as np
import os

class GrabCutPainter():
    
    def __init__(self, infolder, outfolder):
        self.supported_files = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')
        self.datapath = os.path.join(outfolder, dataname)
        self.__dict__.update(locals())
        self.index = 0

        # figure out where you left off on the save
        for root, dirs, files in os.walk(self.outfolder):
            image_paths = [file for file in files if file.lower().endswith(self.supported_files)]
            for impath in image_paths:
                self.index += 1


    def grabcut(self):
        for root, dirs, files in os.walk(self.infolder):
            image_paths = [file for file in files if file.lower().endswith(self.supported_files)]
            for impath in image_paths:
                self.cut(impath)

    def cut(self, path):
        image = cv2.imread(path)

fg = False  # draw foreground hints
bg = False  # draw background hints
drawing = False # toss eventually
mode = False
ix, iy = -1, -1

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x,y), 1, -1)
            else:
                cv2.circle(img, (x,y), 5, 1, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x,y), 1, -1)
        else:
            cv2.circle(img, (x,y), 5, 1, -1)

if __name__=='__main__':
    img = np.full((512,512,1), 0.5, dtype=np.float32)
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while(True):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
