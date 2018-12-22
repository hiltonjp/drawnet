import cv2
import os
from argparse import ArgumentParser


class DataExtractor(object):

    supported_files = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tif')
    scale = 1

    def extract(self, src, dst, dataname):
        """Extract regions of interest from every image in a folder.

        Args:
            src (str): The source folder full of images. The function will
                search through all sub-folders.
            dst (str): The destination folder where all regions of interest
                will be saved.
            dataname (str): The prefix used when saving each image.
        """

        index = 0
        if os.path.exists(dst):
            for root, dirs, files in os.walk(dst):
                for file in files:
                    index += 1 if file.lower().endswith(self.supported_files) \
                               else 0
        else:
            os.makedirs(dst)

        for root, dirs, files in os.walk(src):
            image_paths = [file for file in files
                           if file.lower().endswith(self.supported_files)]

            for impath in image_paths:
                filepath = os.path.join(root, impath)
                img = cv2.imread(filepath)
                index = self.extract_from_image(img, dst, dataname, index)

    def extract_from_image(self, img, dst, dataname, index=0):
        """User interface to extract regions of interest from a single image.

        Args:
            img (numpy.ndarray): A numpy array of shape (h, w, 3). The
                image to extract regions of interest from.
            dst (str): The destination folder where all regions of interest
                will be saved.
            dataname (str): The prefix used when saving each image.
            index (int): The index to start counting from (if you know several
                images from the same dataset are already in the destination
                folder).

        Returns:
            index (int): The starting index plus the number of regions extracted
        """
        datapath = os.path.join(dst, dataname)

        scale = DataExtractor.scale
        img_preview = img[::scale, ::scale]
        rect = cv2.selectROI('Select Region',
                             img_preview,
                             showCrosshair=False,
                             fromCenter=False)

        while rect != (0, 0, 0, 0):
            x, y, w, h = rect

            # Re-apply scale to account for scaled display.
            x, y, w, h = x*scale, y*scale, w*scale, h*scale

            roi = img[y:y+h, x:x+w]
            cv2.imwrite(datapath + f'_{index:03}.png', roi)

            index += 1

            rect = cv2.selectROI('Select Region',
                                 img_preview,
                                 showCrosshair=False,
                                 fromCenter=False)

        return index


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('src',
                        type=str,
                        help='The source folder for images')

    parser.add_argument('dst',
                        type=str,
                        help='The destination folder to save cropped images')

    parser.add_argument('--name', '-n',
                        type=str,
                        default='concepts',
                        help='The name each extracted file is given')

    args = parser.parse_args()

    extractor = DataExtractor()
    extractor.extract(args.src, args.dst, args.name)

