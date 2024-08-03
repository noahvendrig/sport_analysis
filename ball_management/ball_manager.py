
from helpers.bounding_box_utils import get_distance


class BallManager:
    def __init__(self, ball_dict):
        self.ball_tracks = ball_dict
        self.ball_possession = {} # {frame_n: player_id}
    
    def get_proximity(self, player_bounding_box, ball_bounding_box):
        player_pos = ((player_bounding_box[0]+player_bounding_box[2])/2, (player_bounding_box[3])) # get dist from player foot (bottom of image)
        ball_pos = ((ball_bounding_box[0]+ball_bounding_box[2])/2, (ball_bounding_box[1]+ball_bounding_box[3])/2) # get dist from ball bottom

        return get_distance(player_pos, ball_pos)
    
    
    
    # Calculates possession of DETECTED ball
    def get_possession(self, player_detection, player_id):    
        for frame_n, balls_dict in enumerate(self.ball_tracks): # ideally we should have only 1 ball track per frame
            for ball_id, ball_detection in balls_dict.items():
                # for player_id, player_detection in players_dict[frame_n].items():
                ball_bounding_box = ball_detection["bounding_box"]
                player_bounding_box = player_detection["bounding_box"]
                
                proximity = self.get_proximity(player_bounding_box, ball_bounding_box)
                if proximity < 30: # some arbitrary value
                    # print(proximity)
                    # self.ball_tracks[frame_n][ball_id]["nearby_players"].append(player_id)
                    self.ball_tracks[frame_n][ball_id].setdefault("nearby_players", []).append(player_id)

        
        # return self.ball_tracks