import cv2
import sys

width = int(sys.argv[2])
height = int(sys.argv[3])

img = cv2.imread(sys.argv[1], 0)
img_res = cv2.resize(img, (width, height))
cv2.imwrite(sys.argv[4], img_res)

