import cv2
from helpers.segmentation_utils import k_cluster, get_img_weighted_avg, crop_frame


class TeamManager:
    def __init__(self):
        self.team_colours = {}
        self.teams = {} # player id, and what team theyre in

    def bin_round(self, num):
        if num < 0.5:
            return 0
        else:
            return 1
        
    def get_clustered_img(self, img, labels):
        return labels.reshape(img.shape[0], img.shape[1])

    def get_player_colour(self, frame, bounding_box):
        img = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

        top_half = img[0:int(img.shape[0]/2), :]

        k_means = k_cluster(top_half, n_clusters=2)
        k_means_img = self.get_clustered_img(top_half, k_means.labels_)

        corner_clusters = [k_means_img[0, 0], k_means_img[0, -1], k_means_img[-1, 0], k_means_img[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count) # get the most common cluster in the corners
        
        player_cluster_idx = 1 - non_player_cluster
        player_colour = k_means.cluster_centers_[player_cluster_idx] # returns colour from segmentation, however this is not the true colour. it will be some shade of the real colour (which we cluster later)
        return player_colour

        # weighted_avg = get_img_weighted_avg(k_means_img) # value between 0 to 1 inclusive.
        # player_cluster_idx = 1 - self.bin_round(weighted_avg) # the val is 1 or 0 so subtract from 1 get opposite
        # player_colour = k_means.cluster_centers_[player_cluster_idx] # returns colour from segmentation, however this is not the true colour. it will be some shade of the real colour (which we cluster later)

        return player_colour
    
    def assign_team_colour(self, frame, player_detections):
        rough_player_colours = []
        for _, player_detection in player_detections.items():
            bounding_box = player_detection['bounding_box']
            player_colour = self.get_player_colour(frame, bounding_box) # send cropped img to the segmentation
            
            rough_player_colours.append(player_colour)

        k_means_teams = k_cluster(rough_player_colours, n_clusters=2)

        self.k_means_teams = k_means_teams

        self.team_colours[0] = k_means_teams.cluster_centers_[0]
        self.team_colours[1] = k_means_teams.cluster_centers_[1]

        # now cluster the colours into two distinct groups

    def assign_team(self, frame, bounding_box, player_id):
        if player_id in self.teams:
            return self.teams[player_id]
        
        player_colour = self.get_player_colour(frame, bounding_box) # send cropped img to the segmentation
        team_id = self.k_means_teams.predict(player_colour.reshape(1, -1))[0] # outputs 0 or 1

        self.teams[player_id] = team_id # save to dict
        return team_id


