"""The optional full-person display cannot change tracker/control inputs."""
import unittest
import numpy as np
from . import pose_full_box


class FullBoxTests(unittest.TestCase):
    def test_occluded_legs_use_a_clipped_geometric_prior(self):
        pose=np.zeros((17,3))
        box, evidence=pose_full_box.project_box((100,100,140,200),pose,640,480)
        self.assertEqual(evidence,'standing_prior')
        self.assertEqual(box,(100.,100.,140.,277.))
        box,_=pose_full_box.project_box((100,400,140,470),pose,640,480)
        self.assertEqual(box[3],480)

    def test_current_unique_pose_changes_only_the_display_copy(self):
        rows=[[100.,100.,140.,200.,7,.9,0,0]]
        detections=np.array([[100.,100.,140.,200.,.9,0]])
        keypoints=dict(boxes=detections[:,:4].copy(),kpts=np.zeros((1,17,3)))
        result=pose_full_box.project_tracks(rows,detections,keypoints,640,480,0)
        self.assertEqual(rows[0][3],200.)
        self.assertEqual(result[0][3],277.)
        self.assertEqual(result[0][4:],rows[0][4:])
        keypoints=dict(boxes=np.repeat(detections[:,:4],2,axis=0),kpts=np.zeros((2,17,3)))
        self.assertEqual(pose_full_box.project_tracks(rows,detections,keypoints,640,480,0),rows)
