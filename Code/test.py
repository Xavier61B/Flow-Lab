# Author @xavier_barneclo
# Automated image based contact angle measuring device code
# Goal is to simply run this script while connected to the camera
# Things to look into: opencv, otsu's method

import cv2
import matplotlib.pyplot as plt
import numpy as np
from list import list
import os

# Variables
white = 255
black = 0

# Max sized array function for countour detection
#def max(c):
#    max = np.array([])
#    for i in c:
#        if i.size > max.size:
#            max = i
#    return i

# Distance function for points
def distance(p_0,p_1):
    return abs(p_0[1]-p_1[1])+abs(p_0[0]-p_1[0])

# Swap two indices
def swap(point):
    new = [point[1],point[0]]
    return new

# Minimize distance to point
def min(c,l,r):
    rhs = 0
    lhs = 0
    for q in range(c.shape[0]):
        i = c[q][0]
        dr = distance(i,r)
        dl = distance(i,l)
        if distance(i,c[rhs][0]) > dr:
            rhs = q

        if distance(i,c[lhs][0]) > dl:
            lhs = q
    c = [c[rhs],c[lhs]]
    
    return c

#def normalize(data,x,y):
#    new = data
#    for i in range(data.shape[1]):
#       new[0][i] = new[0][i]-x
#       new[1][i] = new[1][i]-y
#    return new

    

#def new_poly(data,order):
#    A = np.zeros((data.shape[1],order))
#    for i in range(order):
#        for q in range(data.shape[1]):
#            A[q][i] = data[0][q]**(order-i)
#    AT = np.transpose(A)
#    inv = np.linalg.inv(np.matmul(AT,A))
#    ls = np.matmul(inv,AT)
#    return np.matmul(ls,np.transpose(data[1][:]))

def y_val(x,c,order):
    out = 0
    for i in range(order+1):
        out += c[i]*x**(order-i)
    return out


# Image reading
image = cv2.imread("2.png",0)
row = image.shape[0] - 1
column = image.shape[1] - 1

# Output image to write to
blank = np.zeros((row,column,3), np.uint8)

# Otsu's method applied
otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

# Canny edge detection
edge = cv2.Canny(image_result, otsu_threshold/2,otsu_threshold)

# Boundary detection
lhs = None
rhs = None
for i in range(column):
    for q in range(row):
        if edge[q][i] == white and not lhs:
            lhs = [q,i]
        if edge[q][column-i] == white and not rhs:
            rhs = [q,column-i]

    if lhs and rhs:
        if lhs[0] > rhs[0]:
            out = rhs[0]
        else:
            out = lhs[0]
        break

# Boundary points double-linked list
bound = list(swap(lhs))
a = bound.add(edge,white)
while a:
    a = bound.add(edge,white)

# Crude boundary finder
#refine_edge = edge[:out][:]

# Detect corners with the goodFeaturesToTrack function.
corners = cv2.goodFeaturesToTrack(edge, 10, 0.01, 10)
corners = min(corners, swap(lhs), swap(rhs))
corners = np.int0(corners)
print(corners)
# drawing circles
for i in corners:
    x,y = i.ravel()
    cv2.circle(edge, (x, y), 3, 255, -1)

c = []
for i in corners:
    c.append(i.ravel())
lpp = bound.find(c[1])
rpp = bound.find(c[0])

delta = int(((rhs[0] - lhs[0])**2 + (rhs[1]-lhs[1])**2)**.5/2)
le = np.array(bound.get_list(lpp,delta))
ri = np.array(bound.get_list_back(rpp,delta))

zero_y1 = le[1,0]
zero_y2 = ri[1,0]
zero_x1 = le[0,0]
zero_x2 = ri[0,0]
#le = normalize(le,zero_x,zero_y)
#lpol = new_poly(np.array(swap(le)),4)
lpol = np.poly1d(np.polyfit(le[1,:],le[0,:],4))
rpol = np.poly1d(np.polyfit(ri[1,:],ri[0,:],4))

dl = np.polyder(lpol)
dr = np.polyder(rpol)

dl = dl(zero_y1)
dr = dr(zero_y2)

contact_right = np.degrees(np.arctan(1/dr))
contact_left = np.degrees(np.arctan(1/dl))

print([contact_right,contact_left])

# drawing circles
for i in range(zero_y1,20,-1):
    y = i
    x = int(lpol(y))
    cv2.circle(edge, (x, y), 3, 255, -1)

# Attempt drawing on blank
a = bound.start.next
for i in range(bound.get_size()-1):
    x = a.get_x()
    y = a.get_y()
    blank[y,x] = [0,255,255]
    a = a.get_next()
    

#for i in range(ri.shape[1]):
#    x = ri[0,i]
#    y = int(rpol[0]*x**5 + rpol[1]*x**4 + rpol[2]*x**3 + rpol[3]*x**2 + rpol[4]*x + rpol[5])
#    print(y)
#    cv2.circle(edge, (x, y), 3, 255, -1)

# Countour finding code
#contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#con = max(contours)
#cv2.drawContours(blank, [con], 0, (255,255,0),3)

# Creating convex hull for contours
#hull = cv2.convexHull(con)
#cv2.drawContours(blank, [hull], 0, (0, 255, 0), 2)

# Writing to txt file
files = os.listdir()
if "output.txt" in files:
    file = open("output.txt", "a") 
else:
    file = open("output.txt","w")
file.write(str(contact_left) + "," + str(contact_right) +"\n")



# Image writing
cv2.imwrite("refine.png",blank)
cv2.imwrite("edge.png",edge)