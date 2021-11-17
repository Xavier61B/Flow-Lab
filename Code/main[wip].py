# Author @xavier_barneclo
# Automated image based contact angle measuring device code
# Goal is to simply run this script while connected to the camera
# Things to improve: user interface, output images

import cv2
import matplotlib.pyplot as plt
import numpy as np
from list import list
import os

#
## TO-DO: add user interface section here
#

#
# Variable declaration section
#
white = 255
black = 0
order = 2
img_file = "trial_2_cam2_cropped.jpg"
d_factor = 4

#
# Helper functions section
#

# Distance function for points
def distance(p_0,p_1):
    return ((p_0[1]-p_1[1])**2+(p_0[0]-p_1[0])**2)**.5

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

# Drawing helper function
def draw(x_0,y_0,blank,m):
    len = int(blank.shape[0]/5)
    y = y_0
    x = int(m*(y-y_0)+x_0)
    for i in range(len):
        if not ((x >= 0 and x < blank.shape[1]) and (y >= 0 and y < blank.shape[0])):
            break
        blank[y,x] = [255,0,0]
        y -= 1
        x = int(m*(y-y_0)+x_0)
    return blank

# Max sized array function for countour detection
def max(c):
    max = np.array([])
    for i in c:
        if i.size > max.size:
            max = i
    return i

#
# Contact angle measurement main body
#

# Greyscale image reading
image = cv2.imread(img_file,0)
row = image.shape[0] - 1
column = image.shape[1] - 1

# Output image to write to
blank = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

# Otsu's method applied
otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

# Canny edge detection (using Otsu's thresholds)
edge = cv2.Canny(image_result, otsu_threshold/2,otsu_threshold)

cv2.imwrite("otsu.png",image_result)

# Countour finding code
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
con = max(contours)
cv2.drawContours(edge, contours, -1, (255,255,255), 3)

# Creating convex hull for contours
#hull = cv2.convexHull(con)
#cv2.drawContours(edge, [hull], 0, (0, 255, 0), 2)

cv2.imwrite("cont.png",edge)

# Right and left most edge detection
lhs = None
rhs = None
for i in range(column):
    for q in range(row):
        if edge[q][i] == white and not lhs:
            lhs = [q,i]
        if edge[q][column-i] == white and not rhs:
            rhs = [q,column-i]

# Generate boundary points list
bound = list(swap(lhs))
a = bound.add(edge,white)
while a:
    a = bound.add(edge,white)

# Detect pinning points by minimzing distance from edges to corner points
corners = cv2.goodFeaturesToTrack(edge, 20, 0.01, 10)
#corners = min(corners, swap(lhs), swap(rhs))
corners = np.int0(corners)

c = []
for i in corners:
    a = i.ravel()
    x,y = a
    c.append(a)
    cv2.circle(edge, (x, y), 4, 255, -1)

cv2.imwrite("canny.png",edge)

# Find pinning points in boundary list
print(c)
lpp = bound.find(c[1])
rpp = bound.find(c[0])

# Define delta (number of pixels used in polyfit), and get points for polyfitting
delta = int(distance(rhs,lhs)/d_factor)
le = np.array(bound.get_list(lpp,delta))
ri = np.array(bound.get_list_back(rpp,delta))

# Polyfit
zero_y1 = le[1,0]
zero_y2 = ri[1,0]
zero_x1 = le[0,0]
zero_x2 = ri[0,0]

print(zero_x2)
print(zero_y2)

print(zero_x1)
print(zero_y1)

lpol = np.poly1d(np.polyfit(le[1,:],le[0,:],4))
rpol = np.poly1d(np.polyfit(ri[1,:],ri[0,:],4))

#for i in range(zero_y1,0,-1):
#    x = int(lpol(i))
#    y = i
#    cv2.circle(edge, (x, y), 3, 255, -1)

#for i in range(zero_y2,0,-1):
#    x = int(rpol(i))
#    y = i
#    cv2.circle(edge, (x, y), 3, 255, -1)



# Find contact angle
dl = np.polyder(lpol)
dr = np.polyder(rpol)

dl = dl(zero_y1)
dr = dr(zero_y2)

contact_right = 180 - np.degrees(np.arctan2(1,-1*dr))
contact_left = np.degrees(np.arctan2(1,-1*dl))

# Draw to output image
a = bound.start.next
for i in range(bound.get_size()-1):
    x = a.get_x()
    y = a.get_y()
    blank[y,x] = [0,255,255]
    a = a.get_next()

blank = draw(zero_x1,zero_y1,blank,dl)
blank = draw(zero_x2,zero_y2,blank,dr)

cv2.circle(blank, (zero_x1, zero_y1), 3, [0,0,255], -1)
cv2.circle(blank, (zero_x2, zero_y2), 3, [0,0,255], -1)

# Writing to txt file
files = os.listdir()

i = 1
out = "result " + str(i) + ".png"
while out in files:
    i += 1
    out = "result " + str(i) + ".png"

if "output.txt" in files:
    file = open("output.txt", "a") 
else:
    file = open("output.txt","w")
    file.write("Trial number, left pinning point contact angle, right pinning point contact angle\n")
file.write(str(i) + "," + str(contact_left) + "," + str(contact_right) +"\n")

# Output image
cv2.imwrite(out,blank)