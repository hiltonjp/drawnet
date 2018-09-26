import cv2
import numpy as np
import os

class GrabCutPainter():
    
    # static vars
    drawing = True

    def __init__(self, infolder, outfolder):
        self.supported_files = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')
        # self.datapath = os.path.join(outfolder, dataname)
        self.__dict__.update(locals())
        self.index = 0

        if not os.path.exists(self.outfolder):
            os.makedirs(self.outfolder)

        print(os.path.exists(self.infolder))

        # figure out where you left off on the save
        for root, dirs, files in os.walk(self.outfolder):
            image_paths = [file for file in files if file.lower().endswith(self.supported_files)]
            for impath in image_paths:
                self.index += 1

    def grabcut(self):

        for root, dirs, files in os.walk(self.infolder):
            # print(files)
            image_paths = [os.path.join(root, file) for file in files if file.lower().endswith(self.supported_files)]
            for impath in image_paths:
                self.cut(impath)

    def cut(self, path):
        image = cv2.imread(path)
        
        rect = cv2.selectROI('Select ROI', image, False, False)
        cv2.destroyAllWindows()
        
        orig = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        image = orig.copy()
        self.drawing = True

        # grabcut initialization
        rect = (0, 0, *image.shape[:2])
        mask = np.zeros(image.shape[:2])
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros_like(bgdModel)
        iterations = 5

        mask, bgdModel, fgdModel = cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)

        while self.drawing:
            m = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            print(np.array_equal(m, np.zeros_like(m)))           
            prev = image*m[:, :, np.newaxis]
            canvas = np.full((prev.shape), 128, dtype=np.uint8)

            layout = np.concatenate((canvas, image, prev), axis=1)
            cv2.imshow('Preview', layout)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()


    @staticmethod
    def draw(event, x, y, flags, param):
       pass 

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

    gcp = GrabCutPainter(
        infolder='/home/hiltonjp/sample',
        outfolder='/home/hiltonjp/drawnet/cut'
    )

    gcp.grabcut()

#     img = np.full((512,512,1), 0.5, dtype=np.float32)
    
#     cv2.namedWindow('image')
#     cv2.setMouseCallback('image', draw_circle)

#     while(True):
#         cv2.imshow('image', img)
#         if cv2.waitKey(20) & 0xFF == 27:
#             break

#     cv2.destroyAllWindows()
