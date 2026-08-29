import cv2  # type: ignore
import numpy as np
import time

#Time recording (for performance measuring)
start_time = time.time()

#load video, get info
cap = cv2.VideoCapture('./part2/input_p2.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#output video
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('./part2/output_p2.mp4', fourcc, fps, (frame_width, frame_height))

def process_image(input_img): #same as part1

    #Filter pixels based on brightness (using value channel in HSV)
    value = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)[:,:,2]
    binary = cv2.threshold(value, 127, 255, cv2.THRESH_BINARY)[1]

    #Morph open, then close to remove noise and fill holes
    kernel = np.ones((int(8/1278*frame_width), int(8/713*frame_height)), np.uint8) # adjusted kernel size based on video resolution (adapted from 8x8 in part 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #Find contours of the binary image
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    #Draw contours and mark centers
    white=(255, 255, 255)
    black=(0, 0, 0)
    for contour in contours:
        #Trace the outlines
        cv2.drawContours(input_img, [contour], 0, black, 2)

        #Mark the center
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            cv2.circle(input_img, (cX, cY), 5, black, -1)
            #Draw coords
            cv2.putText(input_img, f"coords: [{cX}, {cY}]", (cX, cY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    return input_img


#Process video frame by frame
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_image(frame) #run the algorithm on the frame
    out.write(processed_frame)
    
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    print(f'\rProgress: {frame_count}/{total_frames} ({progress:.1f}%)', end='', flush=True) #progress counter

    # # This will only print out the first 5s of the video, used for debug purposes:
    # if frame_count/fps >= 5:
    #     break

#Runtime measurement
end_time = time.time()
print(f"\nTotal runtime: {end_time - start_time:.2f} s")
print(f"Amount of video processed: {frame_count*(1/fps):.2f} s")


print('Done!')
cap.release()
out.release()