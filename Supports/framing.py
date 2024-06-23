import numpy as np


def frame(cluster_centers,r_H,r_L):
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
    return rect_centers, mean


