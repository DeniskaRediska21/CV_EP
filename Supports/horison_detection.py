import cv2
import numpy as np
import cupy as cp
from cupyx.scipy.signal import convolve

def detect_horison(image,L,L_out,H,kernel,convolve_list,slices,num,horison_angle, horison_height, horison_angle_trashold, horison_angle_delta_trashold, horison_height_trashold, horison_height_delta_trashold):

    image_r = cv2.resize(image,(L,H)) # resize for other operations
    M = []
    image_r = cp.array(image_r)
    for i in range(0,np.size(convolve_list)-1):
        M.append(np.argmax(np.abs(convolve(cp.mean(image_r[:,convolve_list[i]:convolve_list[i+1]],axis = 1),kernel,'valid'))).get())
    l = np.linspace(0,1280,slices+3)[1:-1]
    l = np.delete(l,int(np.size(l)/2))
    M = np.array(M) 
    d = np.abs(M - np.median(M))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    m = 2
    
    M = np.array(M)
    M = M[s<m] + num
    l = l[s<m]
    

# Horison line extrapolation


    l3 = 0
    l4 = L_out
    
    h3,h4 = np.polyval(np.polyfit(l,M,1),[l3,l4])
    
    horison_angle_new = np.arctan((h4-h3)/(l4-l3))*180/np.pi
    flag_horison_angle_delta = np.abs(horison_angle - horison_angle_new)>horison_angle_delta_trashold
    horison_angle = horison_angle_new
    flag_horison_angle = horison_angle > horison_angle_trashold

    horison_height_new = H/2 - np.mean((h3,h4))
    flag_horison_height_delta = np.abs(horison_height - horison_height_new)>horison_height_delta_trashold
    flag_horison_height = horison_height > horison_height_trashold
    return l3, l4, h3, h4, horison_angle, horison_height, flag_horison_angle, flag_horison_angle_delta,flag_horison_height,flag_horison_height_delta


