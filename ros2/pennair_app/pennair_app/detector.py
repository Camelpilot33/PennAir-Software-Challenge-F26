import cv2
import numpy as np

# Constants from problem statement
K = np.array([[2564.3186869, 0, 0], [0, 2569.70273111, 0], [0, 0, 1]], dtype=np.float32)
CIRCLE_RADIUS = 10.0  # circle radius in inches
FOCAL_LENGTH = 0.5 * (K[0, 0] + K[1, 1])
# Reduce the video size for processing speed
PROCESS_SCALE = 0.5


def process_image(input_img):
    frame_height, frame_width = input_img.shape[:2]
    proc_width = max(1, int(frame_width * PROCESS_SCALE))
    proc_height = max(1, int(frame_height * PROCESS_SCALE))
    kernel = np.ones(
        (int(13 / 1278 * proc_width), int(13 / 713 * proc_height)), np.uint8
    )  # (kernel used for morphological operations) adjusted kernel size based on video resolution (adapted from 8x8 in part 1)
    # scale image down for speed
    scaled = cv2.resize(
        input_img, (proc_width, proc_height), interpolation=cv2.INTER_AREA
    )

    # Same hsv value filter from before
    value = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)

    # Now I use Sobel (gradient of image intensity) to find the smoothness of the image
    grad_x = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # smoothness = white-||gradient||
    smoothness = 255.0 - cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    smoothness = smoothness.astype(np.uint8)

    # threshold for low gradient magnitude (ie smooth shapes)
    binary = cv2.threshold(smoothness, 230, 255, cv2.THRESH_BINARY)[1]

    # morphological open. I removed close because the shapes are well defined, and it was burning time for very little gain in quality
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # find contours of the binary image
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    min_area = 0.005 * proc_width * proc_height

    # DETECTIONS
    detections = []
    for contour in contours:
        # skip very small contours
        if cv2.contourArea(contour) < min_area:
            continue

        # scale everything back up to the original image size so the output video is good quality
        contour_scaled = (contour / PROCESS_SCALE).astype(np.int32)
        # removed drawing

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(int(M["m10"] / M["m00"]) / PROCESS_SCALE)
            cY = int(int(M["m01"] / M["m00"]) / PROCESS_SCALE)
            radius_px = cv2.minEnclosingCircle(contour)[1]
            radius_px = max(radius_px / PROCESS_SCALE, 1.0)  # scale up to original size

            # z = f * R / r
            # f, r in px, R,z in inches, https://ksimek.github.io/2013/08/13/intrinsic/
            depth_in = (FOCAL_LENGTH * CIRCLE_RADIUS) / radius_px
            # convert to normal format
            outline = [[int(point[0][0]), int(point[0][1])] for point in contour_scaled]
            detections.append(
                {"x": cX, "y": cY, "depth": float(depth_in), "outline": outline}
            )
    return detections
