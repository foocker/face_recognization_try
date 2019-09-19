import numpy as np
import cv2


img = np.load('img.npy')
t = np.load('target.npy')
im = img.transpose(1, 2, 0)

x1, y1, x2, y2 = int(t[0][0]), int(t[0][1]), int(t[0][2]), int(t[0][3])
m = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
coords = list(map(int, t[0][5:]))

polylines = []
for i in range(0, 10, 2):
    polylines.append([coords[i], coords[i+1]])
    im = cv2.circle(im, (coords[i], coords[i+1]), 5, (0, 255, 0), -1)
    im = cv2.polylines(im, [np.array(polylines, np.int32)], True, (0, 255, 255), 4)

cv2.imwrite('imgg.jpg', im)
