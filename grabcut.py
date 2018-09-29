import cv2
import numpy as np
import os

class GrabCutPainter():
    
    # static vars
    drawing = False
    bg = True # True, bg drawing; fg otherwise
    canvas = None
    image = None
    scale = 1

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
                self.handle(impath)

    def handle(self, path):
        # TODO: squash square mask bug
        # TODO: squash mask not updating bug

        GrabCutPainter.scale = 1 # scale for later

        image = cv2.imread(path)
        
        # prompt user to select a Region of the Image
        rect = cv2.selectROI('Select Region of Interest', image, False, False)
        if rect == (0, 0, 0, 0):
            return # No Selection made

        cv2.destroyAllWindows()
        
        # crop image to region
        orig = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        image = orig.copy()
        
        # initialize "potential FG" rectangle, mask, and model arrays for grabcut alg
        rect = (0, 0, *image.shape[:2])
        mask = np.zeros(image.shape[:2])
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros_like(bgdModel)
        iterations = 5

        print(rect)
        print(image.shape)
        print(mask.shape)

        mask, bgdModel, fgdModel = cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)

        cv2.namedWindow('Draw FG/BG Hints')
        cv2.setMouseCallback('Draw FG/BG Hints', GrabCutPainter.draw)

        # reset canvas for drawing fg/bg hints
        GrabCutPainter.canvas = np.full(image.shape[:2], 128, dtype=np.uint8)
        GrabCutPainter.image = image.copy()

        finished = False
        while not finished:
                        
            # overlay grabcut results onto a clean background
            alpha_fg = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            alpha_bg = 1 - alpha_fg
            preview = np.full_like(image, 255)
            preview = alpha_fg[:,:,np.newaxis]*image + alpha_bg[:,:,np.newaxis]*preview   
            
            # print(image.shape)
            # print(preview.shape) 
            # display drawing canvas, original image, and grabcut preview side by side
            scale = GrabCutPainter.scale
            layout = np.concatenate((GrabCutPainter.image[::scale, ::scale], preview[::scale, ::scale]), axis=1)
            cv2.imshow('Draw FG/BG Hints', layout)
            key = cv2.waitKey(20)
             
            if key == 13: # Enter Key
                cv2.destroyAllWindows()

                # pre update 
                cv2.imshow('Mask', mask*255)
                cv2.waitKey(20000)
                cv2.destroyAllWindows()

                # update grabcut FG/BG mask with painted hints
                mask[GrabCutPainter.canvas == 0] = 0
                mask[GrabCutPainter.canvas == 255] = 1
                
                # pre cut
                cv2.imshow('Mask', mask*255)
                cv2.waitKey(20000)
                cv2.destroyAllWindows()
                
                mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)
                # post cut
                cv2.imshow('Mask', mask*255)
                cv2.waitKey(20000)
                cv2.destroyAllWindows()
                # Reset display image for easier UX
                GrabCutPainter.image = image.copy()
                    
                # restart window
                cv2.namedWindow('Draw FG/BG Hints')
                cv2.setMouseCallback('Draw FG/BG Hints', GrabCutPainter.draw)

            elif key == 32: # Space Key
                finished = True

            elif key >= 49 and key <= 57: # 1-9 keys
                GrabCutPainter.scale = key - 48 # To obtain valid scale range

            elif key == 109: # 'M' key, for "mode," either bg or fg
                GrabCutPainter.bg = False if GrabCutPainter.bg else True
 
            elif key == 27: # Escape
                pass   


     
        cv2.destroyAllWindows()


    
    @staticmethod
    def draw(event, x, y, flags, param):
        scale = GrabCutPainter.scale
        x, y = x*scale, y*scale

        if event == cv2.EVENT_LBUTTONDOWN:
            GrabCutPainter.drawing = True
        
        elif GrabCutPainter.drawing and event == cv2.EVENT_MOUSEMOVE:
            if GrabCutPainter.bg:
                cv2.circle(GrabCutPainter.image, (x,y), 16, (0,0,0), -1)
                cv2.circle(GrabCutPainter.canvas, (x,y), 16, -1)
            else:
                cv2.circle(GrabCutPainter.image, (x,y), 16, (255,0,0), -1)
                cv2.circle(GrabCutPainter.canvas, (x,y), 16, 255 -1)

        elif event == cv2.EVENT_LBUTTONUP:
            GrabCutPainter.drawing = False
                

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
        infolder='/media/hiltonjp/DATA/drawnet/sample',
        outfolder='/media/hiltonjp/DATA/drawnet/cut'
        # infolder='/media/jeff/DATA/drawnet/sample',
        # outfolder='/media/jeff/DATA/drawnet/cut'
    )

    gcp.grabcut()

