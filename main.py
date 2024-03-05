import imageio
import cv2
import numpy as np
import scipy
import scipy.signal
import os
from sklearn.cluster import MeanShift, estimate_bandwidth

cluster_treshold = 20
num = 30
num2 = 5
fps = 30
ED_trashold = 50
verbose = True
# path = 'Data/vecteezy_fishermen-going-to-the-sea-on-a-motor-boat_8051772.mov'
#path = 'Data/Video/1_2021_03_02_15_35_17_removed.mov'
path = 'Data/Video/2_2019_09_04_18_59_20_removed.mp4'
#path = 'Data/Video/3_2021_03_02_06_16_46_removed.mov'
#path = 'Data/Video/4_2021_03_02_07_10_50_removed.mp4'
#path = 'Data/Video/5_2021_03_05_10_52_37_removed.mp4'
#path = 'Data/2021_03_02_06_16_46_removed.mov'


out_directory = path[:path.rfind('.')-1]



if not os.path.exists(out_directory):
    os.mkdir(out_directory)

cap = cv2.VideoCapture(path)


line_width = 1


writer = imageio.get_writer(f'{out_directory}/video.avi', fps=fps)


# Check if camera opened successfully
if (cap.isOpened()== False): 
  raise Exception("Error opening video stream or file")
 
ret, image = cap.read()
image = np.mean(image, axis = 2).astype('uint8')
L,H = int(2*1280/320), int(720)

L_out = 1280

show_r = True

image = cv2.resize(image,(L,H))

H,L = np.shape(image)
L =int(L/2)*2
image = image[:,:L]
mid = int(L/2)
kernel_column = np.concatenate((np.linspace(0, -1, num = num), np.linspace(1,0,num = num)))
S = np.shape(image)
kernel = np.transpose(np.tile(kernel_column, (mid,1)))


kernel2 = np.concatenate((np.linspace(0, -1, num = num2), np.linspace(1,0,num = num2)))
#kernel2 = np.array([1,-1])



if verbose:
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    if show_r:
        cv2.resizeWindow('Result', 1280,int(720/2))
    else:
        cv2.resizeWindow('Result', 1280,720)


count = 0
names = []
conv2_prev = []

cluster_centers = []
cluster_centers_prev = []

# Read until video is completed
while(cap.isOpened()):
    ret, image = cap.read()
    if ret == True:

        image = np.mean(image, axis = 2).astype('uint8') # Making grayscale image from RGB
        # image = cv2.resize(image,(1280,720))
        image = cv2.resize(image,(L_out,H)) # resize for horison detection
        image_r = cv2.resize(image,(L,H)) # resize for other operations

# Horison detection
        conv_L = scipy.signal.convolve(image_r[:,:mid], kernel, 'valid') # Convolution of left half with kernel
        conv_R = scipy.signal.convolve(image_r[:,mid:], kernel, 'valid') # Convolution of right half with kernel
        
        M_L = np.argmax(np.abs(conv_L)) # Index of max of conlolved image half 
        M_R = np.argmax(np.abs(conv_R)) # Index of max of conlolved image half

# Horison line extrapolation
        # l1 = int(mid/2)
        # l2 = mid + int(mid/2)
        l1 = int(L_out/4)
        l2 = 3*int(L_out/4)
        h1 = M_L + num
        h2 = M_R + num

        k = (h2 - h1)/(l2 - l1)
        b = h1 - k*l1

        l3 = 0
        l4 = L_out

        h3 = k * l3 + b
        h4 = k * l4 + b

        image = cv2.GaussianBlur(image, (3,3),0) # Bluring 
        #image = cv2.line(image, (l3, int(h3)), (l4, int(h4)), (0,0,255),line_width)

# Edge detection
        #conv2 = cv2.Sobel(src=image[:int(np.min((h3,h4))),:], ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1) # Sobel Edge Detection on the X axis
        conv2= np.abs(scipy.ndimage.convolve1d(input = image[:int(np.max((h3,h4))),:].astype(float), weights = kernel2.astype(float), axis = 1))
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


# Clustering
        # The following bandwidth can be automatically detected using
        # bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=500)
        bandwidth = 50
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(points)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        if np.size(cluster_centers_prev):
            for i,cluster_center in enumerate(cluster_centers):
                if np.min(np.abs(np.sum(cluster_center - cluster_centers_prev,axis = 1)))<cluster_treshold:
                    np.delete(cluster_centers,i,0)
                    
        cluster_centers_prev = cluster_centers

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        #print("number of estimated clusters : %d" % n_clusters_)

        if show_r:
            l1_r = int(L/4)
            l2_r = 3*int(L/4)
            h1_r = M_L + num
            h2_r = M_R + num
    
            k_r = (h2_r - h1_r)/(l2_r - l1_r)
            b_r = h1_r - k*l1_r
            l3_r = 0
            l4_r = L
    
            h3_r = k_r * l3_r + b_r
            h4_r = k_r * l4_r + b_r
            
            image_r = cv2.line(image_r, (l3_r, int(h3_r)), (l4_r, int(h4_r)), (0,0,255),line_width)
            


        names.append(f"{out_directory}/%d.jpg"%count)
        count+=1

        if verbose:
            if show_r:
                
                conv2_disp = np.vstack(((255*conv2/np.max(conv2)).astype('uint8'),image[np.shape(conv2)[0]:,:]))
                for center in cluster_centers:
                    conv2_disp = cv2.circle(conv2_disp, (int(center[1]),int(center[0])), radius=10, color=(255, 255, 255), thickness=10)

                disp = np.hstack((image,conv2_disp))
                cv2.imshow('Result', disp)
                cv2.imwrite(names[-1], disp)
                writer.append_data(disp)
            else:
                cv2.imshow('Result',image)
                cv2.imwrite(names[-1], image)
                writer.append_data(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
                
    else:
        break

writer.close()
cap.release()
cv2.destroyAllWindows()
