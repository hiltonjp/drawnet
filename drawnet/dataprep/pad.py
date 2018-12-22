import cv2
import argparse
import os


def square_pad(img):
    shape = img.shape

    # shape[0] is height, shape[1] is width
    if shape[0] > shape[1]:
        dif = shape[0] - shape[1]
        top, bot = 0, 0
        left, right = dif // 2, dif // 2

        if dif % 2 != 0:
            left += 1

    elif shape[1] > shape[0]:
        dif = shape[1] - shape[0]
        top, bot = dif//2, dif//2
        left, right = 0, 0

        if dif % 2 != 0:
            bot += 1
    else:
        return img

    return cv2.copyMakeBorder(img,
                              top, bot, left, right,
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])


def square_all(src, dst):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tif')

    if not os.path.exists(dst):
        os.mkdir(dst)

    for root, dirs, files in os.walk(src):
        image_paths = [file for file in files if file.lower().endswith(exts)]

        for impath in image_paths:
            print(impath)

            filepath = os.path.join(root, impath)
            filename, ext = os.path.splitext(impath)

            img = cv2.imread(filepath)

            try:
                img = square_pad(img)
            except AttributeError:
                continue

            dstpath = os.path.join(dst, filename)
            while os.path.exists(dstpath+ext):
                dstpath += '-dup'
            dstpath += ext

            cv2.imwrite(dstpath, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('src')
    parser.add_argument('dst')

    args = parser.parse_args()

    square_all(args.src, args.dst)




