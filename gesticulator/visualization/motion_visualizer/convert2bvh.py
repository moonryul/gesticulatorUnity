import numpy as np
import subprocess
import os
import sys

from pymo.writers import BVHWriter
import joblib

#data_pipe_path = '../gesticulator/utils/data_pipe.sav'
#write_bvh(data_pipe_path, input_array, bvh_file.name, fps)

# # Gesticulator only supports these 15 joints
#         joint_names = [
#             'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
#             'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
#             'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']
        
        
def write_bvh(datapipe_file, anim_clip, filename, fps): #anim_clip: nparray, quaternions
    data_pipeline = joblib.load(datapipe_file[0]) #This function can load data_pipeline object of PipeLine class saved separately during the  dump
   
   #MJ:  joint_angles = data_pipeline.inverse_transform( predicted_motion.detach())[0].values in my_convert_to_euler_angles
    inv_data = data_pipeline.inverse_transform(anim_clip) #MJ: data_pipeline is an object of class PipeLine; data_pipeline.inverse_transform is a getter property of Pipeline
                             # anim_clip: paramter_type ="expmap":.inverse_transform  applies expmap2euler()
                             # inverse_transform() creates 174 channels from 45 channels because it restores all of the nonused joints to their fixed original values
    writer = BVHWriter()
    for i in range(0, anim_clip.shape[0]): #MJ: anim_clip.shape = (1,520,45)
        with open(filename, "w") as f:
            
            writer.write(inv_data[i], f, framerate=fps)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: python convert2bvh.py <path to data_pipe file> <path to prediction file>"
        )
        sys.exit(0)

    datapipe_file = sys.argv[1]
    pred_file = sys.argv[2]
    print("data pipline: " + datapipe_file)
    print("prediction file: " + pred_file)

    jt_data = np.load(pred_file)
    if jt_data.ndim == 2:
        jt_data = np.expand_dims(jt_data, axis=0)
    out_filename = os.path.splitext(os.path.basename(pred_file))[0] + ".bvh"
    print("writing:" + out_filename)
    write_bvh(datapipe_file, jt_data[:, :, :], out_filename, fps=20)

