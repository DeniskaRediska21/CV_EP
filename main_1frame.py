import cv2
import numpy as np
import scipy


num = 30
path = 'Data/Boat.jpeg'
#path = 'Data/Boat2.jpeg'

image = cv2.imread(path)
image = np.mean(image, axis = 2).astype('uint8')

H,L = np.shape(image)

L =int(L/2)*2

image = image[:,:L]
mid = int(L/2)


image_L = image[:,:mid]
image_R = image[:,mid:]


kernel_column = np.concatenate((np.linspace(0, -1, num = num), np.linspace(1,0,num = num)))

kernel = np.transpose(np.tile(kernel_column, (mid,1)))

conv_L = scipy.signal.convolve(image_L, kernel, 'valid')
conv_R = scipy.signal.convolve(image_R, kernel, 'valid')

M_L = np.argmax(np.abs(conv_L))
M_R = np.argmax(np.abs(conv_R))

image = cv2.line(image, (0,M_L+num), (L,M_R+num), (0,0,255),1)

print(np.shape(kernel))
print(H)
print(np.shape(conv_L))

cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 1280,720)
cv2.imshow('Result',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
