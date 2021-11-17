# Author @xavier_barneclo
# Automated image based contact angle measuring device code
# Goal is to simply run this script while connected to the camera
# Things to improve: user interface

import cv2
import numpy as np
from list import list
import os

#
## TO-DO: add user interface section here
#

# Update list
# 1. User interface
# 2. Optimize for dye pictures (i.e. change pinning point detection, offset, and convergence code)

#
# Variable declaration section
#
img_file = "optical_filter_one.jpg"        #picture file to run script on
white = 255               #white pixel value
black = 0                 #black pixel value
order = 2                 #order of polynomial fit
d_factor = 10             #factor to calculate number of pixels to run polyfit on
offset = 5                #pixel offset for derivative calculation (often times actual pinning point corner is slightly too low)
convergence = 15          #difference in angle to determine model convergence
divergence = 20           #number of points to try until giving up
error_flag = 5            #number of pixel offset between pinning points before user is alerted to check image manually

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
def min(c,t):
    r_dist = np.inf
    l_dist = np.inf

    rhs = 0
    lhs = 0

    x_g = t[0]
    for q in range(c.shape[0]):
        i = c[q][0]
        dist = i[0] - x_g
        if dist > 0 and dist < r_dist:
            rhs = q
            r_dist = c[rhs][0][0] - x_g

        if dist < 0 and abs(dist) < l_dist:
            lhs = q
            l_dist = abs(c[lhs][0][0] - x_g)

    c = [c[rhs],c[lhs]]
    return c


# Drawing helper function
def draw(x_0,y_0,blank,m,d):
    len = int(d*1.5)
    y = y_0
    x = int(m*(y-y_0)+x_0)
    for i in range(len):
        if not ((x >= 0 and x < blank.shape[1]) and (y >= 0 and y < blank.shape[0])):
            break
        blank[y,x] = [255,0,0]
        y -= 1
        x = int(m*(y-y_0)+x_0)
    return blank

def max(c):
    max = np.array([])
    for i in c:
        if i.size > max.size:
            max = i
    return max

def find_top(img, lhs):
    row = lhs[0]
    for i in range(row,0,-1):
        cond = False
        for q in range(0,img.shape[1]):
            if img[i][q] == 255:
                cond = True
        if not cond:
            for q in range(0,img.shape[1]):
                if img[i + 1][q] == 255:
                    return [row,q]

#
# Contact angle measurement main body
#

# Greyscale image reading
image = cv2.imread(img_file,0)
row = image.shape[0]
column = image.shape[1]

# Output images to write to
blank = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
black = np.zeros((row,column,1), np.uint8)

# Otsu's method applied
otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

# Canny edge detection (using Otsu's thresholds)
edge = cv2.Canny(image_result, otsu_threshold/2,otsu_threshold)

# countour detection to guarantee corners aren't detected within droplet
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
con = max(contours)
cv2.drawContours(black, [con], -1, (255,255,255), 1)

# Right and left most edge detection
lhs = None
rhs = None
for i in range(column - 1):
    for q in range(row - 1,0,-1):
        if black[q][i] == white and not lhs:
            lhs = [q,i]
        if black[q][column - 1 - i] == white and not rhs:
            rhs = [q,column - 1 - i]
    if lhs and rhs:
        break

# Generate boundary points list, this boundary will contain both the substrate and the droplet
bound = list(swap(lhs))
a = bound.add(edge,white)
while a:
    a = bound.add(edge,white)

# The heart of the contact angle measurement
# Finds a point somewhere near the center of the droplet (may not actually be centered)
# Then the first closest corners to the left and right of this point
# From there generates points list from the boundary points list of length delta and polyfits to this point
# Using the derivative and arctan2, the contact angle is then measured at the pinning point
# 

num = 2
close = False
top = find_top(black,lhs)
delta = int(distance(rhs,lhs)/d_factor)

while not close:
    corners = cv2.goodFeaturesToTrack(black, num, 0.01, 10)
    corners = min(corners, swap(top))
    corners = np.int0(corners)

    c = []
    for i in corners:
        c.append(i.ravel())

    # Find pinning points in boundary list
    lpp = bound.find(c[1])
    rpp = bound.find(c[0])

    # get points for polyfitting
    le = np.array(bound.get_list(lpp,delta))
    ri = np.array(bound.get_list_back(rpp,delta))

    # Polyfit
    zero_y1 = le[1,0]
    zero_y2 = ri[1,0]
    zero_x1 = le[0,0]
    zero_x2 = ri[0,0]

    lpol = np.poly1d(np.polyfit(le[1,:],le[0,:],4))
    rpol = np.poly1d(np.polyfit(ri[1,:],ri[0,:],4))


    # Find contact angle
    dl = np.polyder(lpol)
    dr = np.polyder(rpol)

    suml = 0
    sumr = 0
    
    for i in range(offset + 1):
        suml += dl(zero_y1 - i)
        sumr += dr(zero_y2 - i)
    
    dl = suml/(offset + 1)
    dr = sumr/(offset + 1)

    dely = c[0][1] - c[1][1]
    delx = c[0][0] - c[1][0]
    substrate_angle = np.degrees(np.arctan2(-1 * dely,delx))

    contact_right = 180 - np.degrees(np.arctan2(1,-1*dr)) + substrate_angle
    contact_left = np.degrees(np.arctan2(1,-1*dl)) - substrate_angle

    if abs(contact_right - contact_left) < convergence:
        close = True

    if num > divergence:
        print("Solution diverges, shape is too rough for proper pinning point detection")
        break
    num += 1

flag = False
if error_flag < abs(c[1][1] - c[0][1]):
    flag = True
    print("Warning, large pinning point deviation observed")

# Draw to output image
a = bound.start.next
for i in range(bound.get_size()-1):
    x = a.get_x()
    y = a.get_y()
    blank[y,x] = [0,255,255]
    a = a.get_next()

blank = draw(zero_x1,zero_y1,blank,dl,delta)
blank = draw(zero_x2,zero_y2,blank,dr,delta)

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
    file.write("Trial number, left pinning point contact angle, right pinning point contact angle, flag for error\n")
file.write(str(i) + "," + str(contact_left) + "," + str(contact_right) + "," + str(flag) + "\n")

# Output image
cv2.imwrite(out,blank)