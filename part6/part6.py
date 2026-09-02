import cv2  # type: ignore
import numpy as np
import subprocess
import time
import platform

# Constants from problem statement
K = np.array([[2564.3186869, 0, 0], [0, 2569.70273111, 0], [0, 0, 1]], dtype=np.float32)
CIRCLE_RADIUS = 10.0  # circle radius in inches
FOCAL_LENGTH = 0.5 * (K[0, 0] + K[1, 1])
# Reduce the video size for processing speed
PROCESS_SCALE = 0.5


# Time recording (for performance measuring)
start_time = time.time()

# load video, get info
cap = cv2.VideoCapture("./part6/input_p6.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# output video
if platform.system() == "Windows":
    # use the dll
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        "./part6/output_p6.mp4", fourcc, fps, (frame_width, frame_height)
    )
else:
    # for non windows systems
    output_path = "./part6/output_p6.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{frame_width}x{frame_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    ffmpeg_write = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# width and height adjusted for rescale
proc_width = max(1, int(frame_width * PROCESS_SCALE))
proc_height = max(1, int(frame_height * PROCESS_SCALE))

# similar algorithm to part1, but using gradient magnitude to find smooth shapes instead of hsv value
kernel = np.ones(
    (int(13 / 1278 * proc_width), int(13 / 713 * proc_height)), np.uint8
)  # (kernel used for morphological operations) adjusted kernel size based on video resolution

# tracking data for pre-specified shapes
tracking = [
    [
        (327.2, 7690, (255, 0, 0), "Circle"),
        (377.4, 8777.0, (0, 255, 0), "Pentagon"),
        (308.2, 4682.5, (0, 0, 255), "Triangle"),
    ],
    {"Circle": [], "Pentagon": [], "Triangle": []},
]


def classify_shape(contour, cX, cY, t):
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    # by miniming the difference in perimeter and area, we can classify the shape
    # this is pretty simplified, but it works well for this case
    score = 0.1
    name = "Untracked"
    color = (255, 255, 255)

    for t_peri, t_area, t_clr, t_name in tracking[0]:
        peri_diff = abs(peri - t_peri) / t_peri
        area_diff = abs(area - t_area) / t_area
        total_diff = peri_diff + area_diff
        if total_diff < score:
            score = total_diff
            name = t_name
            color = t_clr

    # all scores > 0.1:
    if name == "Untracked":
        return name, color

    # find speed to make sure its not a misidentified object
    prev_pos = tracking[1][name][-1] if tracking[1][name] else (cX, cY, t)
    delta_t = max(t - prev_pos[2], 1e-6)
    speed = np.sqrt((cX - prev_pos[0]) ** 2 + (cY - prev_pos[1]) ** 2) / delta_t
    if speed > 3000:
        name = "Untracked"
        color = (255, 255, 255)
    return name, color


def process_image(
    input_img, frames_passed
):  # exact same as p1, but pulled out morph kernel to adjust for video resolution

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

    # draw the contours and mark the centers
    min_area = (
        0.005 * proc_width * proc_height
    )  # minimum area threshold for contours reduces noise from small contours

    # Find depth
    depth_in = 0
    most_circle = 0
    for contour in contours:
        # skip very small contours
        if cv2.contourArea(contour) < min_area:
            continue

        # Is this a circle?
        circularity = (
            4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
        )
        if abs(1 - circularity) > most_circle:
            most_circle = abs(1 - circularity)
            radius_px = cv2.minEnclosingCircle(contour)[1]
            radius_px = max(radius_px / PROCESS_SCALE, 1.0)  # scale up to original size
            # z = f * R / r
            # f, r in px, R,z in inches, https://ksimek.github.io/2013/08/13/intrinsic/
            depth_in = (FOCAL_LENGTH * CIRCLE_RADIUS) / radius_px

    # time passed in s
    t = frames_passed / fps

    for contour in contours:
        # skip very small contours
        if cv2.contourArea(contour) < min_area:
            continue

        # scale everything back up to the original image size so the output video is good quality
        contour_scaled = (contour / PROCESS_SCALE).astype(np.int32)

        # Mark centers and calculate depth
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(int(M["m10"] / M["m00"]) / PROCESS_SCALE)
            cY = int(int(M["m01"] / M["m00"]) / PROCESS_SCALE)

            # classify shape
            name, color = classify_shape(contour, cX, cY, t)
            cv2.drawContours(
                input_img,
                [contour_scaled],
                -1,
                color,
                max(1, int(2 / PROCESS_SCALE)),
            )

            # if this shape is tracked, add its location to tracking data
            if name != "Untracked":
                tracking[1][name].append((cX, cY, t))

            # draw center circle and text
            cv2.circle(input_img, (cX, cY), 5, color, -1)
            cv2.putText(
                input_img,
                f"{name}: [{cX}, {cY}] | depth: {depth_in:.1f} in",
                (cX, cY + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    # draw tracking path for last 0.5s
    for name, path in tracking[1].items():
        # remove points with age older than 0.5s
        while len(path) > 0 and (t - path[0][2]) > 0.5:
            path.pop(0)

        if len(path) < 2:
            continue
        for i in range(1, len(path)):
            pt1 = (int(path[i - 1][0]), int(path[i - 1][1]))
            pt2 = (int(path[i][0]), int(path[i][1]))
            cv2.line(
                input_img,
                pt1,
                pt2,
                tracking[0][[t[3] for t in tracking[0]].index(name)][2],
                2,
            )
    return input_img


# process video frame by frame
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_image(
        frame, frame_count
    )  # run the algorithm on the frame
    if platform.system() == "Windows":
        out.write(processed_frame)
    else:
        ffmpeg_write.stdin.write(processed_frame.astype(np.uint8).tobytes())

    frame_count += 1
    progress = (frame_count / total_frames) * 100
    print(
        f"\rProgress: {frame_count}/{total_frames} ({progress:.1f}%)",
        end="",
        flush=True,
    )

    # # This will only print out the first 5s of the video, used for debug purposes:
    # if frame_count/fps >= 5:
    #     break

# Runtime measurement
end_time = time.time()
print(f"\nTotal runtime: {end_time - start_time:.2f} s")
print(f"Amount of video processed: {frame_count*(1/fps):.2f} s")


print("Done!")
cap.release()
if platform.system() == "Windows":
    out.release()
else:
    ffmpeg_write.stdin.close()
    ffmpeg_rc = ffmpeg_write.wait()
    if ffmpeg_rc != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {ffmpeg_rc}")
