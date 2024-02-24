import imageio
import cv2
import numpy as np
import scipy
import os

num = 30
fps = 30
verbose = True
#path = 'Data/Video/2019_09_04_18_59_20_removed.mp4'
#path = 'Data/Video/1_2021_03_02_15_35_17_removed.mov'
path = 'Data/vecteezy_fishermen-going-to-the-sea-on-a-motor-boat_8051772.mov'


out_directory = path[:path.rfind('.')-1]



if not os.path.exists(out_directory):
    os.mkdir(out_directory)

cap = cv2.VideoCapture(path)


line_width = 2


writer = imageio.get_writer(f'{out_directory}/video.avi', fps=fps)

if verbose:
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', 1280,720)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
ret, image = cap.read()
image = np.mean(image, axis = 2).astype('uint8')
image = cv2.resize(image,(1280,720))
H,L = np.shape(image)
L =int(L/2)*2
image = image[:,:L]
mid = int(L/2)
kernel_column = np.concatenate((np.linspace(0, -1, num = num), np.linspace(1,0,num = num)))
S = np.shape(image)
kernel = np.transpose(np.tile(kernel_column, (mid,1)))

count = 0
names = []
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, image = cap.read()
    if ret == True:

        image = np.mean(image, axis = 2).astype('uint8') # Making grayscale image from RGB
        image = cv2.resize(image,(1280,720))

        conv_L = scipy.signal.convolve(image[:,:mid], kernel, 'valid') # Convolution of left half with kernel
        conv_R = scipy.signal.convolve(image[:,mid:], kernel, 'valid') # Convolution of right half with kernel
        
        M_L = np.argmax(np.abs(conv_L)) # Index of max of conlolved image half 
        M_R = np.argmax(np.abs(conv_R)) # Index of max of conlolved image half

        l1 = int(mid/2)
        l2 = mid + int(mid/2)
        h1 = M_L + num
        h2 = M_R + num

        k = (h2 - h1)/(l2 - l1)
        b = h1 - k*l1

        l3 = 0
        l4 = L

        h3 = k * l3 + b
        h4 = k * l4 + b
     
        image = cv2.line(image, (l3, int(h3)), (l4, int(h4)), (0,0,255),line_width)


        names.append(f"{out_directory}/%d.jpg"%count)
        cv2.imwrite(names[-1], image)
        count+=1
        writer.append_data(image)
        if verbose:
            cv2.imshow('Result',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
                
    else:
        break

writer.close()
cap.release()
cv2.destroyAllWindows()
