import numpy as np
import pdb
import cv2
img_path = "/home/dipesh/aws/crop_n_resize/790-10-1.jpg"
mask_path = "/home/dipesh/aws/crop_n_resize/790-10-1_2.jpg"
img = cv2.imread(img_path)
mask = cv2.imread(mask_path,0)



all_ones = np.where(mask)
x_min, x_max = np.min(all_ones[0]),np.max(all_ones[0])
y_min, y_max = np.min(all_ones[1]),np.max(all_ones[1])
cropped_mask = mask[x_min:x_max,y_min:y_max]
cv2.imwrite("cropped_mask.jpg", cropped_mask)

pdb.set_trace()
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = []


contours = cont
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))

drawing = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    cv2.drawContours(drawing, hull, i, color, 1, 8)

# cv2.imwrite("out0.jpg", out[0])
# cv2.imwrite("out1.jpg", out[1])
# cv2.imshow('mask', mask)
# cv2.imshow('img', img)
cv2.waitKey(0) 

cv2.imwrite("comb_precessed.jpg", drawing)
# cv2.imwrite("precessed.jpg", mask)
# pdb.set_trace()