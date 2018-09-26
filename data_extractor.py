import cv2
import os

class DataExtractor():

    def __init__(self, infolder, outfolder, dataname):
        self.__dict__.update(locals())
        self.supported_files = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tif')
        self.datapath = os.path.join(outfolder, dataname)
        print(self.datapath)
        self.index = 0

        # count up previously gathered data
        if os.path.exists(self.outfolder):
            for root, dirs, files in os.walk(self.outfolder):
                for file in files:
                    self.index += 1 if file.lower().endswith(self.supported_files) else 0
        else:
            os.makedirs(self.outfolder)


    def extract(self):
        for root, dirs, files in os.walk(self.infolder):
            image_paths = [file for file in files if file.lower().endswith(self.supported_files)]
            for impath in image_paths:
                filepath = os.path.join(root, impath)
                self.extract_from_image(filepath)


    def extract_from_image(self, path):
        # change if images still too large for display.
        scale = 5

        img = cv2.imread(path)
        impreview = img[::scale, ::scale]

        rect = cv2.selectROI('Select Region', impreview, showCrosshair=False, fromCenter=False)

        while rect != (0, 0, 0, 0):
            x, y, w, h = rect
            x, y, w, h = x*scale, y*scale, w*scale, h*scale # accounts for image resizing
            roi = img[y:y+h, x:x+w]

            cv2.imwrite(self.datapath + f'_{self.index:03}.png', roi)
            self.index += 1

            rect = cv2.selectROI('Select Region', impreview, showCrosshair=False, fromCenter=False)


if __name__ == '__main__':
    extractor = DataExtractor(
        infolder="/media/jeff/DATA/drawnet/animation_program/nokbak_characters",
        outfolder="/media/jeff/DATA/drawnet/extracted_art/nokbak",
        dataname="nokbak_concepts"
    )

    extractor.extract()
    
