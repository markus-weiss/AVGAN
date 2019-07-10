import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Load data  
log     = cv2.imread('LogMag_1.jpg')  
phase   = cv2.imread('UwrappedPhase_1.jpg')
#cv2.imshow('image', img)  

#print(log[:,:,1])

npLog = np.ndarray((224,224,6))

npLog[:,:,0] = log[:,:,0]
npLog[:,:,1] = log[:,:,1]
npLog[:,:,2] = log[:,:,2]

npLog[:,:,3] = phase[:,:,0]
npLog[:,:,4] = phase[:,:,1]
npLog[:,:,5] = phase[:,:,2]

#print(npLog)

plt.figure(figsize=(4,4))
plt.imshow(log)
plt.show()
