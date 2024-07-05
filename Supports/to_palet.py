from PIL import Image
import numpy as np
import cv2

def to_palet(PATH = '', colors = [0,255], verbose = True, IMAGE = None, output_type = 'cv'): 

    colors = np.array(colors)
    if IMAGE is None:
        img = np.array(Image.open(PATH))/255
    else:
        img = np.array(IMAGE)/255

    
    if len(colors.shape) == 1:
        colors = colors[None,:]
        img = np.mean(img,axis = 2)[:,:,None]
        
    coords = (img*(colors.shape[1]-1)+0.5).astype(np.uint8)
    
    for  i in range(colors.shape[0]):
        coords[:,:,i] = colors[i,coords[:,:,i]]
    
    
    if img.shape[2]==1:
        coords = np.squeeze(coords)

    if output_type == 'PIL':
        coords = Image.fromarray(coords)
        if verbose:
            coords.show()
    elif output_type == 'cv':
        coords = cv2.cvtColor(coords, cv2.COLOR_RGB2BGR)
    
    return coords


# palet = [list(range(0,255,25)),list(range(0,255,25)),list(range(0,255,25))]
# img = to_palet('Data/Penguins.jpg', colors = palet, verbose = True)
# img.save("Results/to_palet1.jpg")
# 
# 
# palet = [[54,77,249,245],[48,76,148,245],[98,125,23,245]]
# img = to_palet('Data/Penguins.jpg', colors = palet, verbose = True)
# img.save("Results/to_palet2.jpg")
