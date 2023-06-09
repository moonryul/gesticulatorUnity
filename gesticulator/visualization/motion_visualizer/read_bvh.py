""" This script can read a BVH file frame by frame.
    Each frame will be converted into 3d Coordinates
    """

# @author: Taras Kucherenko


import motion_visualizer.bvh_helper as BVH

import numpy as np


def append(current_node, current_coords, main_joints):

    # check if we want this coordinate
    if current_node.name in main_joints:
        # append it to the coordinates of the current node
        curr_point = current_node.coordinates.reshape(3)
        current_coords.append(curr_point)

    for kids in current_node.children:
        append(kids, current_coords, main_joints)


def obtain_coords(root, frames, duration, main_joints): #  root, frames, frame_time = BVH.load(bvh_file);    duration = len(frames)

    total_coords = []

    for fr in range(duration): # for each frame fr

        current_coords = []

        root.load_frame(frames[fr])
        root.apply_transformation()

        # Visualize the frame
        append(root, current_coords, main_joints)

        total_coords.append(current_coords)

    return total_coords


def read_bvh_to_array(bvh_file):

    root, frames, frame_time = BVH.load(bvh_file)
    duration = len(frames)

    # We removed lower-body data, retaining 15 upper-body joints out of the original 69.
        # Fingers were not modelled due to poor data quality
        
    main_joints = [
        "Hips",  #1
        "Spine",  #2
        "Spine1", #3
        "Spine2", #4
        "Spine3", #5
        "Neck",  #6
        "Neck1",  #7
        "Head",  # Head and spine  #8
        "RightShoulder", #9
        "RightArm",  #10
        "RightForeArm",  #11
        "RightHand",  #12
        
        "RightHandThumb1", #13
        "RightHandThumb2",  #14
        "RightHandThumb3", #15     
        "RightHandIndex1",  #16
        "RightHandIndex2",  #17
        "RightHandIndex3",  #18
        "RightHandMiddle1", #19
        "RightHandMiddle2", #20
        "RightHandMiddle3", #21
        "RightHandRing1",  #22
        "RightHandRing2",  #23
        "RightHandRing3",  #24
        "RightHandPinky1", #25
        "RightHandPinky2", #26
        "RightHandPinky3",  # Right hand #27
        
        "LeftShoulder",  #28 => 13
        "LeftArm",        #29  => 14
        "LeftForeArm",  #30  => 15
        "LeftHand",      #31  => 16
        "LeftHandThumb1", #32
        "LeftHandThumb2", #33
        "LeftHandThumb3", #34
        "LeftHandIndex1", #35
        "LeftHandIndex2", #36
        "LeftHandIndex3", #37
        "LeftHandMiddle1", #38
        "LeftHandMiddle2", #39
        "LeftHandMiddle3", #40
        "LeftHandRing1", #41
        "LeftHandRing2", #42
        "LeftHandRing3", #43
        "LeftHandPinky1", #44
        "LeftHandPinky2", #45
        "LeftHandPinky3",  # left hand #46
    ]

    coord = obtain_coords(root, frames, duration, main_joints)

    coords_np = np.array(coord)

    # Center to hips
    hips = coords_np[:, 0, :]
    coords_np = coords_np - hips[:, np.newaxis, :]

    return coords_np


if __name__ == "__main__":

    file_path = "/home/taras/Documents/Datasets/SpeechToMotion/Irish/raw/TestMotions/NaturalTalking_001.bvh"

    result = read_bvh_to_array(file_path)
