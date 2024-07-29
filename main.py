import cv2
from helpers import read_video, write_video, segment_image, get_segmentation_coordinates
from trackers import Tracker

def main():
    # Read input video
    filename = "test1.mp4"
    vid_frames = read_video(f"input/{filename}")

    # Perform inference on each frame
    tracker = Tracker("models/yolov5n_best.pt")
    tracks = tracker.get_obj_tracks(vid_frames, read_pickle=True, pickle_path="pickles/tracks.pkl")

    # for _, player in tracks['players'][0].items():
    #     bounding_box = player['bounding_box']
    #     frame = vid_frames[0]

    #     cropped_frame = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]
    #     # cv2.imwrite("output/cropped_frame.jpg", cropped_frame)
    #     desired_contours = segment_image(cropped_frame)
    #     coordinates = get_segmentation_coordinates(desired_contours)
        
    #     cv2.drawContours(cropped_frame, desired_contours, -1, (0, 255, 0), cv2.FILLED)
    #     cv2.imshow("image", cropped_frame)
    #     cv2.waitKey(0)

    #     cv2.destroyAllWindows()
    #     break
    out_frames = tracker.draw_annotations(vid_frames, tracks)

    # Write output to new video file
    out_filename = filename.split(".")[0]+".avi"
    # write_video(out_frames, f"output/{out_filename}")


if __name__=="__main__":
    main()
    # sec()

# def sec():
#     im = cv2.imread("input/1.png")
#     cv2.ellipse(
#             im,
#             center=(400, 400), 
#             axes=(int(50), int(50/4)), 
#             angle=0, 
#             startAngle=0, 
#             endAngle=215, 
#             color=(255,0,0), 
#             thickness=2,
#             lineType=cv2.LINE_4
#         )
#     cv2.imshow("image", im)
#     cv2.waitKey(0)

#     cv2.destroyAllWindows()