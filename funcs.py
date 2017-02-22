import cv2
import numpy as np

#ensures any picture in any format is shown, keep this    
def show(img):
    img2 = img*1
    if img.dtype == np.uint8:    
        cv2.imshow( 'image',img)
    elif np.max(img) <=1 and np.min(img) >= 0:
        img = np.uint8(img*255)
        cv2.imshow( 'image',img)
    elif np.max(img) <= 255 and np.min(img) >=0:
        img = np.uint8(img)
        cv2.imshow('image', img)
    else:
        img = img*1.0
        img -=np.min(img)
        img/=np.max(img)
        img*=255.99999
        img = np.uint8(img)
        cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#gets dimensions
def getwidth(img):
    return img.shape[1]

def getheight(img):
    return img.shape[0]

def read(fname):
    return cv2.imread("input/%s"%fname)

def write(fname,img):
    cv2.imwrite("output/%s"%fname,img)

#saves
def saveimage(name,img):
     img2 = img*1
     if img.dtype == np.uint8:
         cv2.imwrite( name,img)
     elif np.max(img) <=1 and np.min(img) >= 0:
         img = np.uint8(img*255)
         cv2.imwrite( name,img)
     elif np.max(img) <= 255 and np.min(img) >=0:
         img = np.uint8(img)
         cv2.imwrite(name, img)
     else:
         img -=np.min(img)
         img/=np.max(img)
         img*=255.99999
         img = np.uint8(img)
         cv2.imwrite(name, img)
