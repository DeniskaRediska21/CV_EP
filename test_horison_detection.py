from Supports.to_palet import to_palet
import os
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from IPython import display
import time
#from scipy.signal import convolve
#from collections import Counter

verbose = True

path = os.path.join('Data','oleg_5120x2600_h256_24fps.mov')
#path = os.path.join('Data','out.mp4')
cap = cv2.VideoCapture(path)
if (cap.isOpened()== False): 
  raise Exception("Error opening video stream or file")
ret, image = cap.read()

resolution_scale = 0.3

H,L,D = np.shape(image)
H,L = int(H/(1/resolution_scale)),int(L/(1/resolution_scale))
image = cv2.resize(image, (L,H))

#sobel_gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
if verbose:
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result',L,H)

how_much_edges = 2

counter = 0


while(cap.isOpened()):
    ret, image = cap.read()
    image = cv2.resize(image, (L,H))
    #image = to_palet(IMAGE = image, colors = list(range(0,255,50)), verbose = False)
#    edges = convolve(image,sobel_gy,'valid') 
    edges = cv2.Canny(image, 100, 200)
    
    ret, labels = cv2.connectedComponents(edges)
    print(np.shape(np.unique(labels)))

    _,counts = np.unique(labels, return_counts = True)

    best_lines = np.array(np.argpartition(counts, -how_much_edges)[-how_much_edges:])
    points = np.zeros_like(labels)
    for line in best_lines:
        points[labels == line] = line

    labels = points

    horison = np.argwhere(labels == best_lines[0])

    #lines = cv2.HoughLines(labels, 1, np.pi/180, 200)
    uni,counts = np.unique(horison[:,1], return_counts = True)
    uni = uni[counts>1]
    for uni_ in uni:
        indeces = np.argwhere(horison[:,1]==uni_)
        min_y_index = np.argmax(horison[indeces,0])
        indeces = np.delete(indeces,min_y_index)
        
        horison = np.delete(horison,indeces,axis = 0)


#    d = np.abs(horison[:,0] - np.median(horison[:,0]))
#    mdev = np.mean(d)
#    s = d/mdev if mdev else np.zeros(len(d))
#    m = 1
#    horison = horison[s<m,:]
    
    #_,counts = np.unique(labels, return_counts = True)
    P = np.polyfit(horison[:,1],horison[:,0],1)
    tmp = np.array(list(range(0,L)))
    tmp = [i for i in tmp if i not in horison[:,1] ]
    y_tmp = np.polyval(P,tmp)
    extention = np.array([y_tmp,tmp]).T


    extention= np.delete(extention,extention[:,0]>H, axis = 0)
    extention= np.delete(extention,extention[:,1]>L, axis = 0)
    extention = np.delete(extention,(extention<0).any(axis = 1), axis = 0)
    

    horison = np.vstack((horison, extention.astype(int)))


    #horison[:,0] = scipy.signal.medfilt(horison[:,0],1001)

    

    #print(f'{np.size(np.unique(horison[:,1]))}/{np.size(horison[:,1])}')
#    labels = np.zeros_like(labels)
#    labels[horison[:,0],horison[:,1]] = 1
#    # Map component labels to hue val
#    label_hue = np.uint8(179*labels/np.max(labels))
#    blank_ch = 255*np.ones_like(label_hue)
#    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
#    # cvt to BGR for display
#    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#    
#
#    # set bg label to black
#    labeled_img[label_hue==0] = 0
#
#    _,alpha = cv2.threshold(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY)
#    
#    b,g,r = cv2.split(labeled_img)
#    labeled_img = cv2.merge([b,g,r,alpha],4)


    if verbose:
        image[horison[:,0],horison[:,1],:] = (0,0,255)
        cv2.imshow('Result',image)
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite('screenshot.jpg',image)
            break
