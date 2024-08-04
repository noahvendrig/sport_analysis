def get_bounding_box_centre(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bounding_box_width(bounding_box):
    x1, _, x2, _ = bounding_box
    return x2 - x1

def get_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 # pythagoras

def get_closest_foot_position(player_bounding_box, ball_position):
    left_foot = (player_bounding_box[0], player_bounding_box[3])
    right_foot = (player_bounding_box[2], player_bounding_box[3])
    
    distance_left_foot = get_distance(left_foot, ball_position)
    distance_right_foot = get_distance(right_foot, ball_position)
    
    return left_foot if distance_left_foot < distance_right_foot else right_foot

def distance_point_to_line(px, py, x1, y1, x2, y2):
    # Calculate the distance from point (px, py) to the line segment (x1, y1) -> (x2, y2)
    line_mag = get_distance((x1, y1), (x2, y2))

    if line_mag < 1e-10:
        return get_distance((px, py), (x1, y1))

    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    if u < 0:
        closest_point = (x1, y1)
    elif u > 1:
        closest_point = (x2, y2)
    else:
        closest_point = (x1 + u * (x2 - x1), y1 + u * (y2 - y1))
    
    return get_distance((px, py), closest_point)

def get_centre(bounding_box):
    return ((bounding_box[0]+bounding_box[2])/2, (bounding_box[1]+bounding_box[3])/2)