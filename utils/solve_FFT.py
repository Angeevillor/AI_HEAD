import cv2
import numpy as np
import re
def transform(L,f):
    L2=[]
    for i in range(len(L)):
        L2.append(f(L[i]))
    return L2
def get_region(file):
    data=[]
    with open(file) as f:
        for line in f.readlines():
            line2= re.findall(r"\d+\.?\d*",line)
            data.append(transform(line2,int))
    return data
def get_center(file):
    data=[]
    bin_center=[]
    with open(file) as f:
        for line in f.readlines():
            line2= re.findall(r"\d+\.?\d*",line)
            data.append(transform(line2,int))
    for i in data:
        x=(i[0]+i[1])/2
        y=(i[2]+i[3])/2
        bin_center.append([x,y])
    return bin_center
def get_near(list1,list2):
    left=[]
    right=[]
    for i in list1:
        left.append(i[0])
    for j in list2:
        right.append(j[0])
    return [max(left),min(right)]
def check_meridian(center,region):
    data=[]
    for i in region:
        if (center[0][0]-i[0])*(center[0][0]-i[1])<0:
            data.append(i)
        else:
            pass
    return data
def has_intersection(a, b):
    if a[1] < b[0] or b[1] < a[0]:
        return False
    else:
        return True
def union(a, b):
    left = min(a[0], b[0])
    right = max(a[1], b[1])
    return [left, right]
def merge(list):
    if len(list) == 0 or len(list) == 1:
        return list
    list.sort(key=lambda x: x[0])
    result = []
    current = list[0]
    for i in range(1, len(list)):
        next = list[i]
        if has_intersection(current, next):
            current = union(current, next)
        else:
            result.append(current)
            current = next
    result.append(current)
    return result
def get_layerline(file):
    data=[]
    data2=[]
    with open(file) as f:
        for line in f.readlines():
            line2= re.findall(r"\d+\.?\d*",line)
            data.append(transform(line2,int))
    for i in data:
        data2.append((i[2],i[3]))
    data3=list(set(data2))
    data3=merge(data3)
    return data3
def vector(center,data):
    vector=[]
    center=np.array(center)
    data=np.array(data)
    for i in data:
        vec=i-center
        vector.append(vec)
    return vector
def pair(vector1,vector2):
    pair=[]
    for i in vector1:
        for j in vector2:
            pair.append([i[0],j[0]])
    return pair
def point_pair(center,vector1,vector2):
    point_pair=[]
    point1=center+vector1
    point2=center+vector2
    point_pair.append(point1)
    point_pair.append(point2)
    return point_pair
def lattice(center,pair,a,b):
    lattice=[]
    v1_list=[]
    v2_list=[]
    X1_step=a/abs(pair[0][0])
    Y1_step=b/abs(pair[0][1])
    X2_step=a/abs(pair[1][0])
    Y2_step=b/abs(pair[1][0])
    s1 = int(max(X1_step, Y1_step))
    s2 = int(max(X2_step, Y2_step))
    for k in range(-s1,s1):
        a1=k*pair[0]
        v1_list.append(a1)
    for j in range(-s2,s2):
        a2=j*pair[1]
        v2_list.append(a2)
    for x in v1_list:
        for y in v2_list:
            lat=x+y+center
            if lat[0][0]<0 or lat[0][0]>=a or lat[0][1]<0 or lat[0][1]>=b:
                continue
            else:
                lattice.append(lat[0])
    return lattice
def sym_lattice(lattice,center,a,b):
    sym_lattice=[]
    for i in lattice:
        sym=[2*center[0]-i[0],i[1]]
        if sym[0]<0 or sym[0]>=a or sym[1]<0 or sym[1]>=b:
                continue
        else:
            sym_lattice.append(sym)
    return sym_lattice
def mask(img,region):
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask=np.zeros_like(gray)
    for i in region:
        mask[i[2]:i[3],i[0]:i[1]]=255
    return mask
def line_mask(img,region):
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    line_mask=np.zeros_like(gray)
    for i in region:
        line_mask[i[0]:i[1],:]=255
    return line_mask
def filter(kp,mask):
    point_list=[]
    for i in kp:
        if mask[int(i[1]),int(i[0])]==255:
            point_list.append(i)
        else:
            pass
    return point_list
def count(lattice,sym_lattice,mask):
    point_list=[]
    for i in lattice:
        if mask[int(i[1]),int(i[0])]==255:
            point_list.append(i)
        else:
            pass
    for j in sym_lattice:
        if mask[int(j[1]),int(j[0])]==255:
            point_list.append(j)
        else:
            pass
    return point_list
def check_lattice(mu,lattice,x,y):
    list=[]
    for i in lattice:
        if (i[0]-mu[0])*(i[0]-mu[1])<0:
            if i[1]!=y:
                list.append(i)
            else:
                pass
        else:
            pass
    return list
def check_point(list,mask):
    score=1
    for i in list:
        check=mask[int(i[1]),int(i[0])]
        if check==0:
            score-=1
        else:
            continue
    return score
def check_layerline(region,lattice):
    m=len(region)
    data=[0]*m
    for i in lattice:
        for j in range(m):
            if  (i[1]-region[j][0])*(i[1]-region[j][1])<0:
                data[j]+=1
            else:
                pass
    return data

