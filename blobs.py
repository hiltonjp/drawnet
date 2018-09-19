import cv2
import numpy as np

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

    
