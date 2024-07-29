import cv2

def read_video(video_path : str):
    cap = cv2.VideoCapture(video_path)
    
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
 
    frames = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()

    return frames

def write_video(out_frames, out_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    w, h = out_frames[0].shape[1], out_frames[0].shape[0]
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
    for frame in out_frames:
        out.write(frame)

    out.release()