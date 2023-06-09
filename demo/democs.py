from argparse import ArgumentParser
import os
import subprocess

import sys

import torch
import librosa

# Python 3.3 and above, any folder (even without a __init__.py file) is considered a package'

from gesticulator.model.model import GesticulatorModel
from gesticulator.interface.gesture_predictor import GesturePredictor
from gesticulator.visualization.motion_visualizer.generate_videos import visualize

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
    savedcwd = os.getcwd()
    cwdForPython = "D:/Dropbox/metaverse/gesticulator/demo"
    os.chdir(cwdForPython)

    print( "I am HERE: os.getcwd:" + os.getcwd())
    
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
   
    motion = gp.predict_gestures(audio_text, audio_array, sample_rate )#
    #motion =  "input_array must be a numpy array"
    # 3. Visualize the results
    motion_length_sec = int(motion.shape[1] / 20)

   

    visualize(motion.detach(), "temp.bvh", "temp.npy", "temp.mp4", 
              start_t = 0, end_t = motion_length_sec, 
              data_pipe_dir = '../gesticulator/utils/data_pipe.sav')

    # Add the audio to the video
    command = f"ffmpeg -y -i {args.audio} -i temp.mp4 -c:v libx264 -c:a libvorbis -loglevel quiet -shortest {args.video_out}"
    subprocess.call(command.split())

    print("\nGenerated video:", args.video_out)
    
    # Remove temporary files
    for ext in ["npy", "mp4"]:
        os.remove("temp." + ext)
    
    # List of list <==> 2D numpy array:
    # https://stackoverflow.com/questions/64791850/converting-a-list-of-lists-into-a-2d-numpy-array
    # https://stackoverflow.com/questions/9721884/convert-2d-numpy-array-into-list-of-lists

    #resultMat = motion.detach().numpy()[0].tolist() # This will convert matrix to a list of lists
    resultMat = motion.detach().numpy()[0]

    print("type of resultMat in python=\n", type(resultMat))
    #print ( resultMat )
    #length_of_motion = resultMat.shape[0] #==520
    #print(f"length_of_motion={length_of_motion}\n"); # length_of_motion=528 ?? not 520?
    
    # for i in range( length_of_motion ):   
    #   print(str(i) + ":")
    #   for  j in range(45):
    #      print (  resultMat[i][j], end=' ')    

    #   print("\n")      
    
    #restore the original cwd
    os.chdir(savedcwd)
    return resultMat

    
    
def  main2(audio_text, audio_array,  sample_rate):

    #save the current working directory
    savedcwd = os.getcwd()
    cwdForPython = "D:/Dropbox/metaverse/gesticulator/demo"
    os.chdir(cwdForPython)

    print( "I am HERE: os.getcwd:" + os.getcwd())
         
 
    args = parse_args()
    
    feature_type, audio_dim = check_feature_type(args.model_file) #MJ: supported_features = ("MFCC", "Pros", "MFCC+Pros", "Spectro", "Spectro+Pros")
    #MJ => we use "Spectro" because audio_dim  = 64

    # 1. Load the model
    model = GesticulatorModel.load_from_checkpoint(  #MJ: model should be obtained in c# script once for all, not for every utterance
        args.model_file, inference_mode=True)
    # This interface is a wrapper around the model for predicting new gestures conveniently
    gp = GesturePredictor(model, feature_type)

    # 2. Predict the gestures with the loaded model
      
    motion = gp.predict_gestures(audio_text, audio_array, sample_rate )#
    # 3. Visualize the results
    motion_length_sec = int(motion.shape[1] / 20)

    resultMat = motion.detach().numpy()[0]

    print("type of resultMat in python=\n", type(resultMat))
    
    #restore the original cwd
    os.chdir(savedcwd)
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
    
    main1(args.text, args.audio)
