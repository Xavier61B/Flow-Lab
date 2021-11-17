import cv2
import numpy as np

def max(c):
    max = np.array([])
    for i in c:
        if i.size > max.size:
            max = i
    return max

img_file = "optical_filter_one.jpg"
image = cv2.imread(img_file,0)

row = image.shape[0] - 1
column = image.shape[1] - 1

#image = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
image = cv2.bilateralFilter(image, 9, 75, 75)

# Output image to write to
blank = np.zeros((row,column,1), np.uint8)

otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

edge = cv2.Canny(image_result, otsu_threshold/2,otsu_threshold)

contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
con = max(contours)
cv2.drawContours(blank, [con], -1, (255,255,255), 1)



# Detect pinning points by minimzing distance from edges to corner points
corners = cv2.goodFeaturesToTrack(blank, 2, 0.01, 10)
#corners = min(corners, swap(lhs), swap(rhs))
corners = np.int0(corners)

c = []
for i in corners:
    a = i.ravel()
    x,y = a
    cv2.circle(blank, (x, y), 2, 255, -1)

cv2.imwrite("canny.png",edge)
cv2.imwrite("otsu.png",image_result)
cv2.imwrite("cont.png",blank)