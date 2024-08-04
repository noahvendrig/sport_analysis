
from helpers.bounding_box_utils import distance_point_to_line, get_centre, get_closest_foot_position, get_distance


class BallManager:
    def __init__(self, ball_dict):
        self.ball_tracks = ball_dict
        self.ball_possession = {} # {frame_n: player_id}
        self.intercept_distance = 25 # some arbitrary value
        self.possession_distance = 50
    
    def get_proximity(self, player_bounding_box, ball_bounding_box):
        # player_pos = ((player_bounding_box[0]+player_bounding_box[2])/2, (player_bounding_box[3])) # get dist from player foot (bottom of image)
        player_pos = ((player_bounding_box[0]+player_bounding_box[2])/2, (player_bounding_box[1]+player_bounding_box[3])/2) # get dist from player centre
        ball_pos = ((ball_bounding_box[0]+ball_bounding_box[2])/2, (ball_bounding_box[1]+ball_bounding_box[3])/2) # get dist from ball bottom

        return get_distance(player_pos, ball_pos)
    
    
    def get_player_in_possession(self, players, ball_bounding_box):
        ball_position = get_centre(ball_bounding_box)
        min_distance_detected = 999999
        player_in_possession = -1
        
        for player_id, player in players.items():
            player_bounding_box = player["bounding_box"]
            
            distance_left_foot = get_distance((player_bounding_box[0], player_bounding_box[3]), ball_position)
            distance_right_foot = get_distance((player_bounding_box[2], player_bounding_box[3]), ball_position)
            curr_min_distance = min(distance_left_foot, distance_right_foot)
            
            if curr_min_distance < min_distance_detected:
                min_distance_detected = curr_min_distance
                player_in_possession = player_id
                # self.ball_tracks.setdefault("nearby_players", []).append(player_id) 
        
        return player_in_possession  
    
    def get_possible_passes(self, players, player_in_possession_id, player_in_possession, ball_bounding_box):
        ball_position = get_centre(ball_bounding_box)
        team_in_possession = player_in_possession["team"]
        possible_passes = []
        
        
        # player_in_possession_bounding_box = player_in_possession["bounding_box"]
        # if (get_distance(player_in_possession_bounding_box, ball_position) > self.possession_distance): # distance to be considered as in control of the ball
        #     return []
        
        for player_id, player in players.items():
            player_bounding_box = player["bounding_box"]
            
            if team_in_possession != player["team"] or player_in_possession_id == player_id: # same player or player is a member of current team (we want opponents only)
                continue
            
            # distance_left_foot = get_distance((player_bounding_box[0], player_bounding_box[3]), ball_position)
            # distance_right_foot = get_distance((player_bounding_box[2], player_bounding_box[3]), ball_position)
            
            # player_position = None
            # if distance_left_foot < distance_right_foot:
            #     player_position = (player_bounding_box[0], player_bounding_box[3])
            # else:
            #     player_position = (player_bounding_box[2], player_bounding_box[3])
                
            player_position = get_closest_foot_position(player_bounding_box, ball_position)
            
            
            obstructed = False
            for other_player_id, other_player in players.items():
                if other_player_id == player_id or team_in_possession == other_player["team"]: # same player or player is a member of current team (we want opponents only)
                    continue

                other_player_bounding_box = other_player["bounding_box"]
                other_player_left_position = ( # left foot
                    (other_player_bounding_box[0]),
                    (other_player_bounding_box[3])
                )
                
                other_player_right_position = ( # right foot
                    (other_player_bounding_box[2]),
                    (other_player_bounding_box[3])
                )
            

                left_distance_to_line = distance_point_to_line(
                    other_player_left_position[0], other_player_left_position[1],
                    ball_position[0], ball_position[1],
                    player_position[0], player_position[1]
                )
                
                right_distance_to_line = distance_point_to_line(
                    other_player_right_position[0], other_player_right_position[1],
                    ball_position[0], ball_position[1],
                    player_position[0], player_position[1]
                )
                
                distance_to_line = min(left_distance_to_line, right_distance_to_line)

                if distance_to_line < self.intercept_distance:
                    obstructed = True
                    break # a player can intercept, so we dont allow this as a possible pass

            if not obstructed:
                possible_passes.append(player_position) # could append id? or dict with id: position
        
        return possible_passes