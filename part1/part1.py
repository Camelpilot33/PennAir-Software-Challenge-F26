import cv2
import numpy as np

#Load image
img = cv2.imread('./part1/input_p1.png')


#Filter pixels based on brightness (using value channel in HSV)
value = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
binary = cv2.threshold(value, 127, 255, cv2.THRESH_BINARY)[1]

#Morph open, then close to remove noise and fill holes
kernel = np.ones((9,9), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#Find contours of the binary image
contours= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

#Draw contours and mark centers
white=(255, 255, 255)
black=(0, 0, 0)
for contour in contours:
    #Trace the outlines
    cv2.drawContours(img, [contour], 0, black, 2)

    #Mark the center
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        cv2.circle(img, (cX, cY), 5, black, -1)
        #Draw coords
        cv2.putText(img, f"coords: [{cX}, {cY}]", (cX, cY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)


#save/display
cv2.imwrite('part1/output_p1.png', img)
cv2.imshow('output p1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
