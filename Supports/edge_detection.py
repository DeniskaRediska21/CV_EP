import cupy as cp
import cupyx.scipy.ndimage as ndimage
import numpy as np
import cupyx


def detect_edges(image, l3, l4,h3, h4, kernel2,conv2_prev,ED_trashold,edge_detection_edge_number_trashold):
    image = cp.array(image)
    image = cupyx.scipy.ndimage.gaussian_filter(image, sigma = 0.3)
    #image = cv2.GaussianBlur(image, (3,3),0) # Bluring 
    conv2 = np.abs(ndimage.convolve1d(input = image[:int(np.max((h3,h4))),:].astype(float), weights =kernel2.astype(float), axis = 1)).get()
    #image = image.get()
    #conv2= np.abs(ndimage.convolve1d(input = image[:int(np.max((h3,h4))),:].astype(float),weights = kernel2.astype(float), axis = 1))
    if not np.shape(conv2_prev) == (0,):
        I= np.min(np.vstack((np.shape(conv2),np.shape(conv2_prev))),axis = 0)
        conv2_meaned = np.mean((conv2[:int(I[0]),:],conv2_prev[:int(I[0]),:]),axis = 0)
        conv2 = np.vstack((conv2_meaned,conv2[I[0]:,:]))
        
    conv2[conv2<ED_trashold] = 0
    points = np.argwhere(conv2 > 0)
# Zeroing points bellow the horison line through cross product
    v1 = (l4 - l3, h4 - h3)
    V2 = np.transpose(np.vstack((l4 - points[:,0], h4 - points[:,1])))
    xp = np.multiply(v1[0],V2[:,1]) - np.multiply(v1[1], V2[:,0])
    z_points = points[xp>0,:]
    conv2[z_points[:,0],z_points[:,1]] = 0



    conv2_prev = conv2 # Saving to use in averaging

    flag_edge_detection = len(points) > edge_detection_edge_number_trashold
    return image, conv2_prev, flag_edge_detection, conv2, points


