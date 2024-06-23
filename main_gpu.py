# To profile use:
# python -m cProfile -s time -o p.dat -m main_gpu
# snakeviz p.dat
import imageio
import cv2
import numpy as np
import scipy
import scipy.signal

import time
import os

import six
import sys
sys.modules['sklearn.externals.six'] = six
#from utils.mean_shift_cosine_gpu import MeanShiftCosine as MeanShift

#from  meanshift.mean_shift_gpu  import  MeanShiftEuc as MeanShift
from sklearn.cluster import DBSCAN
#from sklearn.cluster import MeanShift
import csv
import cupy as cp
import cupyx

import cupyx.scipy.ndimage as ndimage
from cupyx.scipy.signal import convolve

np.seterr(invalid='ignore')

cluster_treshold = 20
num = 60
num2 = 5
fps = 30
ED_trashold = 20
bandwidth = 10

test_accuracy = False



horison_angle_trashold = 30
horison_height_trashold = 400
horison_angle_delta_trashold = 10
horison_height_delta_trashold = 100

horison_angle = 0
horison_height = 0

edge_detection_edge_number_trashold = 1000

clustering_cluster_number_trashold = 30

path_ = os.path.join('Data','Разметка','Labeling_ships_clear')
path_accuracy = os.path.join(path_,'labels')

number_hit = []
number = []

write = False
verbose = True


rect_line_width = 2

path = os.path.join(path_,'video.avi')

out_directory = path[:path.rfind('.')]

labels_ = []
if test_accuracy:
    txt_files=[]
    
    for file in os.listdir(path_accuracy):
        if file.endswith('.txt'):
            txt_files.append(file)
            with open(path_accuracy+'/'+file) as csvfile:
                reader=csv.reader(csvfile, delimiter = ' ')
                rows= []
                for row in reader:
                    rows.append([float(value) for value in row])
                labels_.append(rows)


if not os.path.exists(out_directory):
    os.mkdir(out_directory)

cap = cv2.VideoCapture(path)


line_width = 2

if write:
    writer = imageio.get_writer(f'{out_directory}/video.avi', fps=fps)


# Check if camera opened successfully
if (cap.isOpened()== False): 
  raise Exception("Error opening video stream or file")
 
ret, image = cap.read()
image = np.mean(image, axis = 2).astype('uint8')

slices = 8
L,H = int(slices*1280/320), int(720)
r_L,r_H = 86,86
L_out = 1280

show_r = True

image = cv2.resize(image,(L,H))

H,L = np.shape(image)
L =int(L/2)*2
image = image[:,:L]
mid = int(L/2)
kernel_column = np.concatenate((np.linspace(0, -1, num = num), np.linspace(1,0,num = num)))
S = np.shape(image)





kernel2 = np.concatenate((np.linspace(0, -1, num = num2), np.linspace(1,0,num = num2)))
kernel2 = cp.array(kernel2)




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

convolve_list = list(range(0,L + int(L/slices),int(L/slices)))
kernel = cp.array(kernel_column)
#kernel = np.transpose(np.tile(kernel_column, (int(L/slices),1)))

# MAIN CYCLE

while(cap.isOpened()):
    flag_horison_angle = False
    

    rect_centers = []
    if test_accuracy:
        label = np.array(labels_[count])
    cluster_centers = []
    ret, image = cap.read()
    if ret == True:

        image = np.mean(image, axis = 2).astype('uint8') # Making grayscale image from RGB
        # image = cv2.resize(image,(1280,720))
        image = cv2.resize(image,(L_out,H)) # resize for horison detection
        image_r = cv2.resize(image,(L,H)) # resize for other operations

        t0= time.time()
# Horison detection
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

        image = cp.array(image)
        image = cupyx.scipy.ndimage.gaussian_filter(image, sigma = 0.3)
        #image = cv2.GaussianBlur(image, (3,3),0) # Bluring 

# Edge detection
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


# Clustering
        if np.size(points) > 0:
            #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            #ms = MeanShift(bandwidth=bandwidth,  GPU = True)
            ms = DBSCAN(eps =50, min_samples = 1)
            ms.fit(points)
            labels = ms.labels_
            
            #cluster_centers = ms.cluster_centers_
            cluster_centers = np.array([np.mean(points[ms.labels_ == v],axis = 0) for v in np.unique(ms.labels_)])

            if np.size(cluster_centers_prev):
                for i,cluster_center in enumerate(cluster_centers):
                    if np.min(np.abs(np.sum(cluster_center - cluster_centers_prev,axis = 1)))<cluster_treshold:
                        np.delete(cluster_centers,i,0)
                        
            cluster_centers_prev = cluster_centers
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            flag_clustering = n_clusters_ > clustering_cluster_number_trashold 
# Framing
 
        center_points = cluster_centers
        TRASH = np.full(np.shape(center_points)[0], True)
        rects = []
        for j,center_point in enumerate(center_points):
            
            det = np.abs(center_points - center_point)
            step = (det < [r_L,r_H]).all(axis = 1)
            flag = True
            for i,rect in enumerate(rects):
                if (np.abs(rect - center_point) < [r_L,r_H]).all():
                    rects[i] = np.vstack((np.squeeze(rects[i]),center_point[TRASH[j]]))
                    TRASH[j] = False
                    # step = np.delete(step,j)
                    flag = False
            if flag:
                if len(rects)>0: 
                    rects.append(center_points[np.all((step,TRASH),axis = 0)]) 
                     
                else:
                    rects = list([center_points[step]])
                
                TRASH[step] = False
        
        mean = []
        for i,rect in enumerate(rects):
            mean.append(np.mean(rect,axis = 0))
        mean = np.array(mean)
        
        if np.size(mean)>0:
            rect_centers = np.flip(mean - [r_L/2,r_H/2],axis = 1)

        total = time.time() - t0
# Plotting and Saving


        names.append(f"{out_directory}/%d.jpg"%count)
        
        image = image.get()
        if verbose:
            if show_r:      
                conv2_disp = np.vstack(((255*conv2/np.max(conv2)).astype('uint8'),image[np.shape(conv2)[0]:,:]))
                for center in cluster_centers:
                    conv2_disp = cv2.circle(conv2_disp, (int(center[1]),int(center[0])), radius=10, color=(255, 255, 255), thickness=10)
                    if test_accuracy:
                        for row in label:
                            conv2_disp = cv2.circle(conv2_disp, (int(row[1] * L_out),int(row[2] * H)), radius=5, color=(255, 0, 0), thickness=10)
                # image = cv2.line(image, (l3, int(h3)), (l4, int(h4)), (0,0,255),line_width)
                if np.size(mean)>0:
                    for rect_center in rect_centers:
                        rect_center[rect_center<0] = 0
                        image = cv2.rectangle(image, rect_center.astype(int), (rect_center + [r_L,r_H]).astype(int),255,rect_line_width)
                

                disp = np.hstack((image,conv2_disp))
                cv2.imshow('Result', disp)
                
                if write:
                    cv2.imwrite(names[-1], disp)
                    writer.append_data(disp)
            else:
                cv2.imshow('Result',image)
                if write:
                    cv2.imwrite(names[-1], image)
                    writer.append_data(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        
        acc = 0
        if test_accuracy:
            if len(rect_centers) > 0:
                for row in label:
                    acc += any(np.abs(row[1] * L_out - rect_centers[:,0]) < r_L) and any(np.abs(row[2] * H - rect_centers[:,1]) < r_H)
            
            number_hit.append(acc)
            number.append(label.shape[0])
        
            print(f'{number_hit[count]} / {label.shape[0]}      {1/total} fps')
        else:
            print(f'{1/total} fps')
        
        
        
        count+=1       
    else:
        break

if write:
    writer.close()
    
cap.release()
cv2.destroyAllWindows()

# print(accuracy)

if False:
    import matplotlib.pyplot as plot
    accuracy = 100 * (np.array(number_hit)/np.array(number))
    plot.plot(accuracy)
    plot.plot(np.mean(accuracy) * np.ones(np.shape(accuracy)))
    plot.title('Точность решения задачи выделения зон интереса')
    plot.xlabel('Номер кадра')
    plot.ylabel('Точность, %')
    plot.legend(['Точность','Среднее значение'])
    plot.grid()
    plot.show()
