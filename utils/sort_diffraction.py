from PIL import Image,ImageDraw
import re
import cv2
def extract_number(filename):
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group())
    else:
        return 0

def converge(img1,img2):
    w1,h1=img1.size
    img2=cv2.resize(img2,(h1,w1))

    img2=Image.fromarray(img2)

    w2,h2=img2.size

    W=(w1+w2)/2
    H=max(h1,h2)
    new_img=Image.new("RGB",(int(W),int(H)))
    left=img1.crop((0,0,w1//2,h1))
    right=img2.crop((w2//2,0,w2,h2))

    new_img.paste(left,(0,0))
    new_img.paste(right,(int(W//2),0))

    return new_img
def convert(img,point,X,PZ):
    dic={}
    a,b=img.size
    if 0<= point[0]<=a//2 and 0<=point[1]<=b//2:
        dic["coordinate"]=point
        dic["layer order"]=1
    elif 0 <= point[0] <= a and point[1] <0:
        count=1
        while point[1]<0:
            point[1]+=PZ
            point[0]-=X
            count+=1
            if (point[0]<=0 or point[1]>=b//2):
                break
        if 0 <= point[0] <= a // 2 and 0 <= point[1] < b // 2:
            dic["coordinate"] = point
            dic["layer order"] =count
        else:
            dic["coordinate"] = []
            dic["layer order"] = -1
    else:
        dic["coordinate"] = []
        dic["layer order"] = -1
    return dic
def generate_lattice(region,X,PZ):
    dic={}
    a,b=region,region
    center_x,center_y=a//2,b//2
    X_step=round(center_x/abs(X))
    Y_step=round(center_y/abs(PZ))
    step=min(X_step,Y_step)
    if step>0:
        for i in range(-step,step+1):
            x=center_x-i*X
            y=center_y-i*PZ
            dic["coordinate_order({})".format(i)]=[x,y]
            dic["step"]=step
    else:
        dic["coordinate_order({})".format(0)]=[center_x,center_y]
        dic["step"]=0
    return dic
def shift_lattice(img,region,X,PZ,Z,n):
    dic={}
    dic_1=generate_lattice(region,X,PZ)
    a,b=img.size
    center_x,center_y=a//2,b//2
    step=dic_1["step"]
    if step>0:
        for j in range(-n,n+1):
            for i in range(-step,step+1):
                x=center_x-i*X
                y=center_y-i*PZ+j*Z
                if 0<=x<=a and 0<=y<=b:
                    dic["coordinate_order({})_pattern({})".format(i,-j)]=[x,y]
                    dic["step"]=step
                    dic["shift"]=n
                else:
                    pass
    else:
        dic["coordinate_order({})_pattern(0)".format(0)]=[center_x,center_y]
        dic["step"]=0
    return dic


