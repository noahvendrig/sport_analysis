import cv2
from helpers import read_video, write_video
from helpers.bounding_box_utils import get_centre, get_closest_foot_position, get_distance
from trackers import Tracker
from team_management import TeamManager
from ball_management import BallManager

def main():
    # Read input video
    # filename = "test1.mp4"
    filename="barca1.mov"
    # filename = "edited_che_mun_1.mov"
    vid_frames = read_video(f"input/{filename}")

    # Perform inference on each frame
    # tracker = Tracker("models/yolov8s_best.pt")
    modelName= "yolov8s_best_4_augmented.pt"
    tracker = Tracker(f"models/{modelName}")
    tracks = tracker.get_obj_tracks(vid_frames, read_pickle=True, pickle_path="pickles/tracks3_v8_barca.pkl")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    # assingn player to teams
    team_manager = TeamManager()
    team_manager.assign_team_colour(vid_frames[0], tracks["players"][0])

    ball_manager = BallManager(tracks["ball"])
    team_possession = []
    possible_passes = []
    
    for frame_n, player_tracks in enumerate(tracks["players"]):
        for player_id, track in player_tracks.items():
            team_id = team_manager.assign_team(vid_frames[frame_n], track["bounding_box"], player_id)
            # track["team_id"] = team_id
            tracks["players"][frame_n][player_id]["team"] = team_id
            tracks["players"][frame_n][player_id]["team_colour"] = team_manager.team_colours[team_id]

        ball_bounding_box = tracks["ball"][frame_n][1]["bounding_box"]
        player_in_possession = ball_manager.get_player_in_possession(player_tracks, ball_bounding_box)
        
        if player_in_possession != -1:
            tracks["players"][frame_n][player_in_possession]["in_possession"] = True
            team_in_possession = tracks["players"][frame_n][player_in_possession]["team"]
            team_possession.append(team_in_possession)
            
            possession_bounding_box = tracks["players"][frame_n][player_in_possession]["bounding_box"]
            ball_centre = get_centre(ball_bounding_box)
            if (get_distance(get_closest_foot_position(possession_bounding_box, ball_centre), ball_centre) < ball_manager.intercept_distance):
                possible_passes.append(ball_manager.get_possible_passes(player_tracks, player_in_possession, tracks["players"][frame_n][player_in_possession], ball_bounding_box))
            else:
                possible_passes.append([])
        else:
            team_possession.append(team_possession[-1] if len(team_possession) > 0 else -1)
            possible_passes.append([-1])
    
    out_frames = tracker.draw_annotations(vid_frames, tracks, team_possession, possible_passes)

    # Write output to new video file
    out_filename = filename.split(".")[0]+"_"+modelName.split(".")[0]+".avi"
    write_video(out_frames, f"output/{out_filename}")


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