from Supports.to_palet import to_palet
import os
import numpy as np
import cv2
from scipy.signal import convolve
from collections import Counter

verbose = True

path = os.path.join('Data','Разметка','Labeling_ships_clear','video.avi')
cap = cv2.VideoCapture(path)
if (cap.isOpened()== False): 
  raise Exception("Error opening video stream or file")
ret, image = cap.read()
H,L,D = np.shape(image)

sobel_gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
if verbose:
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result',L,H)

how_much_edges = 2

counter = 0


while(cap.isOpened()):
    ret, image = cap.read()
    #image = to_palet(IMAGE = image, colors = list(range(0,255,50)), verbose = False)
#    edges = convolve(image,sobel_gy,'valid') 
    edges = cv2.Canny(image, 100, 200)
    
    ret, labels = cv2.connectedComponents(edges)
    _,counts = np.unique(labels, return_counts = True)

    best_lines = np.array(np.argpartition(counts, -how_much_edges)[-how_much_edges:])
    points = np.zeros_like(labels)
    for line in best_lines:
        points[labels == line] = line

    labels = points

    horison = np.argwhere(labels == best_lines[0])

    #lines = cv2.HoughLines(labels, 1, np.pi/180, 200)
    #uni,counts = np.unique(horison[:,1], return_counts = True)
    
    

    #_,counts = np.unique(labels, return_counts = True)
    P = np.polyfit(horison[:,1],horison[:,0],1)
    tmp = np.array(list(range(0,L)))
    tmp = [i for i in tmp if i not in horison[:,1] ]
    y_tmp = np.polyval(P,tmp)
    extention = np.array([y_tmp,tmp]).T

    horison = np.vstack((horison, extention.astype(int)))

    labels[horison[:,0],horison[:,1]] = 100
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    

    # set bg label to black
    labeled_img[label_hue==0] = 0
    if verbose:
        cv2.imshow('Result',labeled_img)
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

