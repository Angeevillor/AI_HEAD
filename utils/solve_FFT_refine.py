import cv2
import numpy as np
global img
global point1, point2
def on_mouse(event, x, y, flags,param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 0, 255), 3)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 3)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 3)
        cv2.imshow('image', img2)
        a = open("./Solve_FFT_Refine/"+"logfile.txt", "a")
        print("X_range:",(point1[0],point2[0]),"Y_range:",(point1[1],point2[1]))
        print("X_range:", (point1[0], point2[0]), "Y_range:", (point1[1], point2[1]),flush=True,file=a)
def show(img_info):
    global img
    img = cv2.imread(img_info)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)
def fast_detect(image,Threshold,mode=2,nonmax=0):
    point_list=[]
    img = cv2.imread(image)
    fast = cv2.FastFeatureDetector_create()
    fast.setType(mode)
    fast.setNonmaxSuppression(nonmax)
    fast.setThreshold(Threshold)
    kp = fast.detect(img, None)
    for p in kp:
        coord = np.array([int(np.array(p.pt)[0]), int(np.array(p.pt[1]))])
        point_list.append(coord)
    return point_list
def fast_show(img,Threshold,kp,mode=0,nonmax=2):
    img = cv2.imread(img)
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(Threshold)
    fast.setType(mode)
    fast.setNonmaxSuppression(nonmax)
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255))
    cv2.imshow("Feature extraction", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()