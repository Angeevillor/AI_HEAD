import cv2
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
        a = open("./Image_information/"+"logfile.txt", "a")
        print("X_range:",(point1[0],point2[0]),"Y_range:",(point1[1],point2[1]))
        print("X_range:", (point1[0], point2[0]), "Y_range:", (point1[1], point2[1]),flush=True,file=a)
def show(img_info):
    global img
    img = cv2.imread(img_info)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    while cv2.waitKey(100) != 27: # loop if not get ESC
      if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) <= 0:
          break
    cv2.destroyAllWindows()
def center(event, x, y, flags,param):
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
        a = open("./Image_information/"+"center.txt", "a")
        # print("selected left zone  coordinate:", point1, "selected right zone  coordinate:", point2)
        # print("selected left zone  coordinate:", point1, "selected right zone  coordinate:", point2,flush=True,file=a)
        print("X_range:",(point1[0],point2[0]),"Y_range:",(point1[1],point2[1]))
        print("X_range:", (point1[0], point2[0]), "Y_range:", (point1[1], point2[1]),flush=True,file=a)
def show(img_info):
    global img
    img = cv2.imread(img_info)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)    
    while cv2.waitKey(100) != 27: # loop if not get ESC
      if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) <= 0:
          break
    cv2.destroyAllWindows()
def show_center(img_info):
    global img
    img = cv2.imread(img_info,0)
    h,w=img.shape
    a = open("./Image_information/" + "center.txt", "a")
    print("X_range:",(w//2,w//2),"Y_range:",(h//2,h//2))
    print("X_range:",(w//2,w//2),"Y_range:",(h//2,h//2), flush=True, file=a)
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    # cv2.setMouseCallback('image', center)
    # cv2.imshow('image', img)
    # while cv2.waitKey(100) != 27: # loop if not get ESC
    #   if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) <= 0:
    #       break
    # cv2.destroyAllWindows()
def contour_detect(image,filter_num,filter_intensity,low,high=cv2.THRESH_OTSU):
    img = cv2.imread(image)
    l=img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1=gray.copy()
    gray2=gray.copy()
    gray1[:,0:l//2]=0
    gray2[:,l//2:l]=0
    edges1 = cv2.Canny(gray1, low, high)
    edges2=cv2.Canny(gray2, low, high)
    # edges1=cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #         cv2.THRESH_BINARY,11,0)
    # edges2=cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #         cv2.THRESH_BINARY,11,0)
    contours1, hierarchy1 = cv2.findContours(edges1,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(edges2,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=contours1+contours2
    # cv2.drawContours(img, contours, -1, (255, 0, 255), 2)

    for i, c in enumerate(contours):

        x, y, w, h = cv2.boundingRect(c)
        if x<l/2 and x+w>l/2:
            pass
        else:
            if max(w,h)<filter_num:
                pass
            elif w<h:
                pass
            elif gray[y:y+h,x:x+w].max()<=filter_intensity:
                pass
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                a = open("./Image_information/"+"logfile.txt", "a")
                print("X_range:" ,(x, x+w), "Y_range:",(y, y+h),flush=True)
                print("X_range:" ,(x, x+w), "Y_range:",(y, y+h),file=a,flush=True)
        cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image", img)
    while cv2.waitKey(100) != 27: # loop if not get ESC
      if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) <= 0:
          break
    cv2.destroyAllWindows()
    cv2.imwrite("./Image_information/"+"contour_detect.jpg",img)
def fast_detect(image,Threshold,mode=2,nonmax=0):
    img = cv2.imread(image)
    fast = cv2.FastFeatureDetector_create()  
    fast.setType(mode)
    fast.setNonmaxSuppression(nonmax)
    fast.setThreshold(Threshold)
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255))
    cv2.namedWindow('Feature extraction', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Feature extraction", img2)
    while cv2.waitKey(100) != 27: # loop if not get ESC
      if cv2.getWindowProperty("Feature extraction", cv2.WND_PROP_VISIBLE) <= 0:
          break
    cv2.destroyAllWindows()
    cv2.imwrite("./Image_information/"+"fast_detect.jpg",img2)

