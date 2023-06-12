from argparse import ArgumentParser
import os
import subprocess

import sys

import torch
import librosa
import numpy as np
import joblib
from gesticulator.visualization.pymo.writers import BVHWriter

# Python 3.3 and above, any folder (even without a __init__.py file) is considered a package'

from gesticulator.model.model import GesticulatorModel
from gesticulator.interface.gesture_predictor import GesturePredictor
from gesticulator.visualization.motion_visualizer.generate_videos import visualize

from gesticulator.visualization.motion_visualizer.convert2bvh import write_bvh

#sys.path = ['D:\\dropbox\\metaverse\\gesticulator\\demo', 'C:\\Users\\moon\\anaconda3\\envs\\gest_env\\python36.zip', 
# 'C:\\Users\\moon\\anaconda3\\envs\\gest_env\\DLLs', 'C:\\Users\\moon\\anaconda3\\envs\\gest_env\\lib', 
# 'C:\\Users\\moon\\anaconda3\\envs\\gest_env', 'C:\\Users\\moon\\anaconda3\\envs\\gest_env\\lib\\site-packages',
#  'd:\\dropbox\\metaverse\\gesticulator', 'd:\\dropbox\\metaverse\\gesticulator\\gesticulator\\visualization']


# https://github.com/Svito-zar/gesticulator/blob/master/install_script.py
# import sys
# import subprocess

# commands = ["-m pip install -r gesticulator/requirements.txt",
#             "-m pip install -e .",
#             "-m pip install -e gesticulator/visualization"] 
#                ==> gesticulator/visualization is added to sys.path, visualization foler has setup,py
#                ==> "motion_visualizer" and "pymo"  packages are  installed

# for cmd in commands:
#     subprocess.check_call([sys.executable] + cmd.split())''


##def  main(audio_file, audio_text, audio_array=None,  sample_rate=-1):
def  main1(audio_text, audio_file):
    
    audio_array, sample_rate = librosa.load(audio_file)
    
    joint_angles = main(audio_text, audio_array,  sample_rate)
    
    bvh_file ="gen_motion.bvh"
    
    resultMat = joint_angles
    print("type of resultMat in python=\n", type(resultMat))
    #print ( resultMat )
    length_of_motion = resultMat.shape[0] # e.g. ==528
    # print(f"length_of_motion={length_of_motion}\n"); # 

    hipPosAndRot = np.ndarray( (length_of_motion,6), dtype=np.float32)
    # set the position and rotation of the Hips joint.
    for i in range( length_of_motion ):
    #   print(str(i) + ":")
    #   for  j in range(45):
    #      print (  resultMat[i][j], end=' ')
        hipPosAndRot[i,:] = [0,0,0,0,0,0]
    
    
    resultMat =  np.concatenate([hipPosAndRot, resultMat], dim=1)
    #  write_bvh((data_pipe_dir,), # write_bvh expects a tuple
    #           motion_clip,
    #           bvh_file,
    #           20)
    
    
    # writer = BVHWriter()
    # for i in range(0, resultMat.shape[0]):
    #     with open(bvh_file, "w") as f:
    #         writer.write( resultMat[i], f, framerate=20) 
  


# changing the current working directory temporarily:
 
# # Get the current working directory
# current_directory = os.getcwd()

# # Set the path to the submodule's demo.py file
# submodule_directory = os.path.join(current_directory, "gesticulatorUnity")

# # Execute the main function with a modified current directory
# try:
#     os.chdir(submodule_directory)
#     main()
# finally:
#     os.chdir(current_directory)
    
    
def  main2(audio_text, audio_array,  sample_rate):

    
    joint_angles = main(audio_text, audio_array,  sample_rate)
    
    bvh_file ="gen_motion.bvh"
    
    resultMat = joint_angles
    print("type of resultMat in python=\n", type(resultMat))
    #print ( resultMat )
    length_of_motion = resultMat.shape[0] # e.g. ==528
    # print(f"length_of_motion={length_of_motion}\n"); # 

    hipPosAndRot = np.ndarray( (length_of_motion,6), dtype=np.float32)
    # set the position and rotation of the Hips joint.
    for i in range( length_of_motion ):
    #   print(str(i) + ":")
    #   for  j in range(45):
    #      print (  resultMat[i][j], end=' ')
        hipPosAndRot[i,:] = [0,0,0,0,0,0]
    
    
    resultMat =  np.concatenate([hipPosAndRot, resultMat], dim=1)
    
    # writer = BVHWriter()
    # for i in range(0, resultMat.shape[0]):
    #     with open(bvh_file, "w") as f:
    #         writer.write( resultMat[i], f, framerate=20)

    
      
    return resultMat

def main(audio_text, audio_array,  sample_rate):
    
    
    current_directory = os.getcwd()
  # Set the path to the submodule's demo.py file
    submodule_directory = os.path.join(current_directory, "gesticulatorUnity\\demo")
    
    try:
        os.chdir(submodule_directory)
        args = parse_args()
        
        #audio_array, sample_rate = librosa.load(audio_file)
        
        
        feature_type, audio_dim = check_feature_type(args.model_file) #MJ: supported_features = ("MFCC", "Pros", "MFCC+Pros", "Spectro", "Spectro+Pros")
        #MJ => we use "Spectro" because audio_dim  = 64

        # 1. Load the model
        model = GesticulatorModel.load_from_checkpoint(  #MJ: model should be obtained in c# script once for all, not for every utterance
            args.model_file, inference_mode=True)
        # This interface is a wrapper around the model for predicting new gestures conveniently
        gp = GesturePredictor(model, feature_type)

        # 2. Predict the gestures with the loaded model
        #motion = gp.predict_gestures(args.audio, args.text) # motion is a tensor: args.text is either a file path or a **string itself**
        #audio_type ="array"
        ##def predict_gestures(self, audio_file, audio_text, audio_array, sample_rate):
    
        predicted_motion = gp.predict_gestures(audio_text, audio_array, sample_rate )#
        #motion =  "input_array must be a numpy array"
        # 3. Visualize the results
        motion_length_sec = int(predicted_motion.shape[1] / 20)
        
        
        predicted_motion = gp.predict_gestures(audio_text, audio_array, sample_rate) #MJ: rotation =vec*theta
        #MJ: =>   predicted_motion = self.model.forward(audio, text, use_conditioning=True, motion=None)
        #         return predicted_motion
        
        joint_angles = my_convert_to_euler_angles(predicted_motion) 
    
    finally:        
    # restore the original cwd
       os.chdir(current_directory)    
       
    return joint_angles
        
def mainpy(audio_text, audio_file):
    
    args = parse_args()
    
    audio_array, sample_rate = librosa.load(audio_file)
    
    
    feature_type, audio_dim = check_feature_type(args.model_file) #MJ: supported_features = ("MFCC", "Pros", "MFCC+Pros", "Spectro", "Spectro+Pros")
    #MJ => we use "Spectro" because audio_dim  = 64

    # 1. Load the model
    model = GesticulatorModel.load_from_checkpoint(  #MJ: model should be obtained in c# script once for all, not for every utterance
        args.model_file, inference_mode=True)
    # This interface is a wrapper around the model for predicting new gestures conveniently
    gp = GesturePredictor(model, feature_type)

    # 2. Predict the gestures with the loaded model
    #motion = gp.predict_gestures(args.audio, args.text) # motion is a tensor: args.text is either a file path or a **string itself**
    #audio_type ="array"
    ##def predict_gestures(self, audio_file, audio_text, audio_array, sample_rate):

    predicted_motion = gp.predict_gestures(audio_text, audio_array, sample_rate )#
    # => (520,45), 45 of exponential maps (x,y,z) = theta*axis_vec:The exponential map uses three parameters to parameterize SO(3), which means
    #  3. Visualize the results
    motion_length_sec = int(predicted_motion.shape[1] / 20)
    
    
    #predicted_motion = gp.predict_gestures(audio_text, audio_array, sample_rate)
    #MJ: =>   predicted_motion = self.model.forward(audio, text, use_conditioning=True, motion=None)
    #         return predicted_motion: https://arxiv.org/pdf/1103.5263.pdf
    #https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
    
    #every rotation is the exponential of an antisymmetric matrix A. R = exp(theta*K_u); K_u= the cross product matrix of u
    # exp(theta*K_{u}) = the rotation by theta radian about u;
    # # The exponential map effects a transformation from the axis-angle representation of rotations to rotation matrices,
    
        
    bvh_file ="gen_motion_python_test.bvh"
    data_pipe_dir='../gesticulator/utils/data_pipe.sav'
    write_bvh((data_pipe_dir,), # write_bvh expects a tuple
            predicted_motion.detach(),
            bvh_file,
            20)
    
    joint_angles = my_convert_to_euler_angles(predicted_motion) #joint_angles:  a numpy array of joint rotations with a shape of (n_frames, 15, 3)
    
    bvh_file ="gen_motion.bvh"
    
    resultMat = joint_angles
    print("type of resultMat in python=\n", type(resultMat))
    #print ( resultMat )
    length_of_motion = resultMat.shape[0] # e.g. ==528
    # print(f"length_of_motion={length_of_motion}\n"); # 

    hipPosAndRot = np.ndarray( (length_of_motion,6), dtype=np.float32) #(520,6)
    # set the position and rotation of the Hips joint.
    for i in range( length_of_motion ):
    #   print(str(i) + ":")
    #   for  j in range(45):
    #      print (  resultMat[i][j], end=' ')
        hipPosAndRot[i,:] = [0,0,0,0,0,0]
    
    print(f'hipPosAndRot={hipPosAndRot.shape}; resultMat={resultMat.shape}')
    resultMat =  np.concatenate([hipPosAndRot, resultMat], axis=1)
    
    # writer = BVHWriter()
    # for i in range(0, resultMat.shape[0]): #(520,51), 45+6 = 51
    #     with open(bvh_file, "w") as f:
    #         writer.write( resultMat[i], f, framerate=20)

    
      
    return resultMat

            
def check_feature_type(model_file):
    """
    Return the audio feature type and the corresponding dimensionality
    after inferring it from the given model file.
    """
    params = torch.load(model_file, map_location=torch.device('cpu'))

    # audio feature dim. + text feature dim.
    audio_plus_text_dim = params['state_dict']['encode_speech.0.weight'].shape[1]

    # This is a bit hacky, but we can rely on the fact that 
    # BERT has 768-dimensional vectors
    # We add 5 extra features on top of that in both cases.
    text_dim = 768 + 5  # 773

    audio_dim = audio_plus_text_dim - text_dim

    if audio_dim == 4:
        feature_type = "Pros"
    elif audio_dim == 64:  # audio-dim = 64
        feature_type = "Spectro"
    elif audio_dim == 68:
        feature_type = "Spectro+Pros"
    elif audio_dim == 26:
        feature_type = "MFCC"
    elif audio_dim == 30:
        feature_type = "MFCC+Pros"
    else:
        print("Error: Unknown audio feature type of dimension", audio_dim)
        exit(-1)

    return feature_type, audio_dim

def my_convert_to_euler_angles(predicted_motion):
        """
        Convert the motion returned by Gesticulator to joint
        rotations around the X, Y and Z axes (relative to the T-pose).

        Args:
            predicted_motion:  the output of the Gesticulator model
        
        Returns:
            rotations:  a numpy array of joint rotations with a shape of (n_frames, 15, 3) =>MJ: I changed it to shape (1,45)
                        where 15 is the number of joints supported by Gesticulator
                        and 3 is the number of axes (X,Y,Z)    
        """
        # The pipeline contains the transformations from data preprocessing
        # It allows us to convert from exponential maps to euler angles
        data_pipe_dir='../gesticulator/utils/data_pipe.sav'
        #data_pipeline = joblib.load("../utils/data_pipe.sav")
        data_pipeline = joblib.load(data_pipe_dir)
        
        #motion_array = predicted_motion.detach().numpy() #MJ: (1,520,45)
        # NOTE: 'inverse_transform' returns a list with one MoCapData object
        joint_angles = data_pipeline.inverse_transform( predicted_motion.detach())[0].values #MJ: (520,174), where 174 is the number of total joints,including the fingers

        
        # Gesticulator only supports these 15 joints
        joint_names = [
            'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
            'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
            'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

        n_joints = len(joint_names)
        n_frames = joint_angles.shape[0]

        # The output joint angles will be stored in 3 separate arrays
        rotations = np.empty((n_frames, n_joints*3)) 

        for joint_idx, joint_name in enumerate(joint_names):
            
            x = joint_angles[joint_name + '_Xrotation']
            y = joint_angles[joint_name + '_Yrotation']
            z = joint_angles[joint_name + '_Zrotation']

            # for frame_idx in range(n_frames):
            #     #rotations[frame_idx, joint_idx, :] = [x[frame_idx], y[frame_idx], z[frame_idx]]
            #     #MJ: The euler angles order is 'ZXY' = roll, pitch, yaw in our case:
            #     rotations[frame_idx, joint_idx, :] = [ z[frame_idx], x[frame_idx], y[frame_idx]]
            
            for frame_idx in range(n_frames):
                #rotations[frame_idx, joint_idx, :] = [x[frame_idx], y[frame_idx], z[frame_idx]]
                #MJ: The euler angles order is 'ZXY' = roll, pitch, yaw in our case:
                rotations[frame_idx, joint_idx:joint_idx+3] = [ z[frame_idx], x[frame_idx], y[frame_idx]]

        return rotations
    
def _convert_to_euler_angles(predicted_motion):
        """
        Convert the motion returned by Gesticulator to joint
        rotations around the X, Y and Z axes (relative to the T-pose).

        Args:
            predicted_motion:  the output of the Gesticulator model
        
        Returns:
            rotations:  a numpy array of joint rotations with a shape of (n_frames, 15, 3) =>MJ: I changed it to shape (1,45)
                        where 15 is the number of joints supported by Gesticulator
                        and 3 is the number of axes (X,Y,Z)    
        """
        # The pipeline contains the transformations from data preprocessing
        # It allows us to convert from exponential maps to euler angles
        data_pipe_dir='../gesticulator/utils/data_pipe.sav'
        #data_pipeline = joblib.load("../utils/data_pipe.sav")
        data_pipeline = joblib.load(data_pipe_dir)
        
        motion_array = predicted_motion.detach().numpy() #MJ: (1,520,45)
        # NOTE: 'inverse_transform' returns a list with one MoCapData object
        joint_angles = data_pipeline.inverse_transform(motion_array)[0].values #MJ: (520,174)

        # Gesticulator only supports these 15 joints
        joint_names = [
            'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
            'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
            'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

        n_joints = len(joint_names)
        n_frames = joint_angles.shape[0]

        # The output joint angles will be stored in 3 separate arrays
        rotations = np.empty((n_frames, n_joints, 3)) 

        for joint_idx, joint_name in enumerate(joint_names):
            
            x = joint_angles[joint_name + '_Xrotation']
            y = joint_angles[joint_name + '_Yrotation']
            z = joint_angles[joint_name + '_Zrotation']

            # for frame_idx in range(n_frames):
            #     #rotations[frame_idx, joint_idx, :] = [x[frame_idx], y[frame_idx], z[frame_idx]]
            #     #MJ: The euler angles order is 'ZXY' = roll, pitch, yaw in our case:
            #     rotations[frame_idx, joint_idx, :] = [ z[frame_idx], x[frame_idx], y[frame_idx]]
            
            for frame_idx in range(n_frames):
                #rotations[frame_idx, joint_idx, :] = [x[frame_idx], y[frame_idx], z[frame_idx]]
                #MJ: The euler angles order is 'ZXY' = roll, pitch, yaw in our case:
                rotations[frame_idx, joint_idx, :] = [ z[frame_idx], x[frame_idx], y[frame_idx]]

        return rotations


def truncate_audio(input_path, target_duration_sec):
    """
    Load the given audio file and truncate it to 'target_duration_sec' seconds.
    The truncated file is saved in the same folder as the input.
    """
    audio, sr = librosa.load(input_path, duration = int(target_duration_sec))
    output_path = input_path.replace('.wav', f'_{target_duration_sec}s.wav')

    librosa.output.write_wav(output_path, audio, sr)

    return output_path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--audio', type=str, default="input/jeremy_howard.wav", help="path to the input speech recording")
    parser.add_argument('--text', type=str, default="input/jeremy_howard.json",
                        help="one of the following: "
                             "1) path to a time-annotated JSON transcription (this is what the model was trained with) "
                             "2) path to a plaintext transcription, or " 
                             "3) the text transcription itself (as a string)")
    parser.add_argument('--video_out', '-video', type=str, default="output/generated_motion.mp4",
                        help="the path where the generated video will be saved.")
    parser.add_argument('--model_file', '-model', type=str, default="models/default.ckpt",
                        help="path to a pretrained model checkpoint")
    parser.add_argument('--mean_pose_file', '-mean_pose', type=str, default="../gesticulator/utils/mean_pose.npy",
                        help="path to the mean pose in the dataset (saved as a .npy file)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    mainpy(args.text, args.audio)
