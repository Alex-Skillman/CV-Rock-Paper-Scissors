import numpy as np
from shared.Data.fingerNames import joints
import mediapipe as mp # type: ignore

def extract_joint_coordinates(hand_landmarks, width, height):
    joint_coordinates = []
    
    for joint in joints:
        x, y, z = finger_locator(joint, hand_landmarks, width, height)
        joint_coordinates.append(x)
        joint_coordinates.append(y)
        joint_coordinates.append(z)
    
    return np.array(joint_coordinates, dtype = np.float32).reshape(-1, 3)

def finger_locator (joint, hand_landmarks, width, height):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


    marker = mp_hands.HandLandmark[joint]
    landmark = hand_landmarks.landmark[marker]
   
    x = landmark.x *  width
    y = landmark.y *  height
    z = landmark.z
    
    return x, y, z

def find_game_start_phase (frames, fps, width, height, game_start_phase):
    
    
    return game_start_phase
    
