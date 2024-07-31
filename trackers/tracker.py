import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pickle
import sys

from helpers.segmentation_utils import segment_image
sys.path.append('../')
from helpers import get_bounding_box_centre, get_bounding_box_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(
            model_path # replace with custom trained model later
        )
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, verbose=False) # set min confidence=0.1 (adjust if needed) . verbose set to false to prevent all the printing
            detections += detections_batch

        return detections
    
    def get_obj_tracks(self, frames, read_pickle=False, pickle_path=None):

        if read_pickle and pickle_path is not None and os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                return pickle.load(f)
                
        detections = self.detect_frames(frames)

        tracks = { 
            "players": [], #  {index:{"bbox":[x,y,x,y]}}
            "ball": [],
            "referees": [],
        }

        for frame_n, detection in enumerate(detections):
            names = detection.names
            names_inv = {v: k for k, v in names.items()}
            # print("Names:", names)
            
            # convert for sv detection format
            detection_sv = sv.Detections.from_ultralytics(detection)
            
            # overwrite goalkeeper with player label (change when dataset is larger for accurate keeper detection)
            for obj, class_id in enumerate(detection_sv.class_id):
                if names[class_id] == "goalkeeper":
                    detection_sv.class_id[obj] = names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_sv) 
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            for frame_detection in detection_with_tracks:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == names_inv["player"]:
                    tracks["players"][frame_n][track_id] = {"bounding_box":bounding_box}

                elif class_id == names_inv["ball"]:
                    tracks["ball"][frame_n][track_id] = {"bounding_box":bounding_box}

                elif class_id == names_inv["referee"]:
                    tracks["referees"][frame_n][track_id] = {"bounding_box":bounding_box}

            for frame_detection in detection_sv:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == names_inv["ball"]:
                    tracks["ball"][frame_n][1] = {"bounding_box":bounding_box}

        if pickle_path is not None:
            with open(pickle_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bounding_box, colour, track_id=None):
        y2 = int(bounding_box[3])
        x_centre, _ = get_bounding_box_centre(bounding_box)
        width = get_bounding_box_width(bounding_box)
        
        cv2.ellipse(
            frame,
            center=(x_centre, y2), 
            axes=(int(width), int(width*0.4)), 
            angle=0, 
            startAngle=-55, 
            endAngle=215, 
            color=colour, 
            thickness=2,
            lineType=cv2.LINE_4
        )
        rect_w, rect_h = 50, 20
        rect_x1 = x_centre - rect_w//2
        rect_x2 = x_centre + rect_w//2
        rect_y1 = (y2 - rect_h//2) + 20 # 10 is some integer offset
        rect_y2 = (y2 + rect_h//2) + 20 # 10 is some integer offset

        if track_id is not None:
            cv2.rectangle(
                frame,
                pt1=(int(rect_x1), int(rect_y1)),
                pt2=(int(rect_x2), int(rect_y2)), 
                color=colour, 
                thickness=cv2.FILLED)
            text_x = rect_x1 + 20 # 15px padding
            text_y = rect_y1 + 15 # 10px padding

            if track_id > 9: # account for bigger num (visual effect)
                text_x -= 5
            if track_id > 99: # account for bigger num (visual effect)
                text_x -= 7
            
            c = get_bounding_box_centre((rect_x1, rect_y1, rect_x2, rect_y2))
        
            cv2.putText(frame, f"{track_id}", (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    def draw_annotations(self, frames, tracks):
        out_frames = []
        for frame_n, frame in enumerate(frames):
            frame = frame.copy()

            players_dict = tracks["players"][frame_n]
            ball_dict = tracks["ball"][frame_n]
            referees_dict = tracks["referees"][frame_n]

            for track_id, player in players_dict.items():
                player_colour = player.get("team_colour", (255, 234, 48))
                frame = self.draw_ellipse(frame, player["bounding_box"], player_colour, track_id)

                # bounding_box = player["bounding_box"]
                # cropped_frame = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]
                # desired_contours = segment_image(cropped_frame)

                # cv2.drawContours(cropped_frame, desired_contours, -1, (0, 255, 0), cv2.FILLED)
                # cv2.imshow("image", cropped_frame)
                # cv2.waitKey(0)

                # cv2.destroyAllWindows()
                # break

            for _, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee["bounding_box"], (69, 75, 255)) # we dont care much abt referee info so dont provide track_id

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bounding_box"], (87, 255, 95))

            out_frames.append(frame)
        return out_frames
    
    def draw_triangle(self, frame, bounding_box, colour):
        x, _ = get_bounding_box_centre(bounding_box)
        y = int(bounding_box[1])

        triangle_pts = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_pts], 0, colour, cv2.FILLED)
        # cv2.drawContours(frame, [triangle_pts], 0, (0,0,0), 2)
        return frame