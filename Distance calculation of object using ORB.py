import opencv as cv2
import numpy as np
import time

imgF = cv2.imread("/home/vishav/Documents/Project/DepthCal/Data/front.png", 0)
imgR = cv2.imread("/home/vishav/Documents/Project/DepthCal/Data/rear.png", 0)

imgF = cv2.resize(imgF, (480, 720))
imgR = cv2.resize(imgR, (480, 720))

imgF = imgF[300:338, 230:279]
imgR = imgR[227:260, 289:327]

start = time.time()
dC = -50
f = 532

liF = []
for i in range(imgF.shape[0]):
    for j in range(imgF.shape[1]):
        if imgF[i][j]>=180 and imgF[i][j]<=255:
            imgF[i][j] = 255
        else:
            imgF[i][j] = 0
            liF.append(i)
            
fy0 = min(liF)
fy1 = max(liF)

liR = []
for i in range(imgR.shape[0]):
    for j in range(imgR.shape[1]):
        if imgR[i][j]>=180 and imgR[i][j]<=255:
            imgR[i][j] = 255
        else:
            imgR[i][j] = 0
            liR.append(i)
            
ry0 = min(liR)
ry1 = max(liR)

pH_f1 = ry1 - ry0
pH_f2 = fy1 - fy0
	
aH = dC / (((1/pH_f1) - (1/pH_f2))*f)
print(aH)
	
dist = (aH*f)/pH_f2
	
print(dist)

end = time.time()

print(1/(end-start))
