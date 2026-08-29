import cv2  # type: ignore
import numpy as np
import time

#Reduce the video size for processing speed
PROCESS_SCALE = 0.5


#Time recording (for performance measuring)
start_time = time.time()

#load video, get info
cap = cv2.VideoCapture('./part3/input_p3.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#output video
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('./part3/output_p3.mp4', fourcc, fps, (frame_width, frame_height))

# #testing on a still frame
# img = cv2.imread('./part3/still_frame.png')
# frame_width, frame_height = img.shape[1], img.shape[0]

#width and height adjusted for rescale
proc_width = max(1, int(frame_width * PROCESS_SCALE))
proc_height = max(1, int(frame_height * PROCESS_SCALE))

#similar algorithm to part1, but using gradient magnitude to find smooth shapes instead of hsv value
kernel = np.ones((int(13/1278*proc_width), int(13/713*proc_height)), np.uint8) # (kernel used for morphological operations) adjusted kernel size based on video resolution (adapted from 8x8 in part 1)
def process_image(input_img):

    #scale image down for speed
    scaled = cv2.resize(input_img, (proc_width, proc_height), interpolation=cv2.INTER_AREA)

    #Same hsv value filter from before
    value = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)

    #Now I use Sobel (gradient of image intensity) to find the smoothness of the image
    grad_x = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # smoothness = white-||gradient||
    smoothness = 255.0 - cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    smoothness = smoothness.astype(np.uint8)

    # threshold for low gradient magnitude (ie smooth shapes)
    binary = cv2.threshold(smoothness, 230, 255, cv2.THRESH_BINARY)[1]

    #morphological open. I removed close because the shapes are well defined, and it was burning time for very little gain in quality
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #find contours of the binary image
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    #draw the contours and mark the centers
    white = (255, 255, 255)
    min_area = 0.005 * proc_width * proc_height  # minimum area threshold for contours reduces noise from small contours
    for contour in contours:
        #skip very small contours
        if cv2.contourArea(contour) < min_area:
            continue

        #scale everything up to the original image size so the output video is good quality
        contour_scaled = (contour / PROCESS_SCALE).astype(np.int32)
        cv2.drawContours(input_img, [contour_scaled], 0, (0, 0, 0), 2)

        #mark center
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(int(M['m10'] / M['m00'])/PROCESS_SCALE)
            cY = int(int(M['m01'] / M['m00'])/PROCESS_SCALE)
            cv2.circle(input_img, (cX, cY), 5, white, -1)
            cv2.putText(input_img, f"coords: [{cX}, {cY}]", (cX, cY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    return input_img

#process video frame by frame
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_image(frame) #run the algorithm on the frame
    out.write(processed_frame)
    
    frame_count += 1
    progress = (frame_count / total_frames) * 100
    print(f'\rProgress: {frame_count}/{total_frames} ({progress:.1f}%)', end='', flush=True)

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