import cv2
from helpers import read_video, write_video
from trackers import Tracker
from team_management import TeamManager

def main():
    # Read input video
    # filename = "test1.mp4"
    filename = "edited_che_mun_1.mov"
    vid_frames = read_video(f"input/{filename}")

    # Perform inference on each frame
    tracker = Tracker("models/yolov5n_best.pt")
    tracks = tracker.get_obj_tracks(vid_frames, read_pickle=True, pickle_path="pickles/tracks1.pkl")

    # assingn player to teams
    team_manager = TeamManager()
    team_manager.assign_team_colour(vid_frames[0], tracks["players"][0])

    for frame_n, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team_id = team_manager.assign_team(vid_frames[frame_n], track["bounding_box"], player_id)
            # track["team_id"] = team_id
            tracks["players"][frame_n][player_id]["team"] = team_id
            tracks["players"][frame_n][player_id]["team_colour"] = team_manager.team_colours[team_id]
    
    out_frames = tracker.draw_annotations(vid_frames, tracks)

    # Write output to new video file
    out_filename = filename.split(".")[0]+".avi"
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