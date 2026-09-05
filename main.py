import webcam # Assumes your custom wrapper works perfectly
import cv2
import mediapipe as mp # type: ignore
import torch # type: ignore
import numpy as np
import time
from shared.Data.fingerNames import joints
from shared.functions.mediapipe_funcs import draw_annotations
from shared.functions.RPS_ai_funcs import load_model
from shared.functions.joint_pos_funcs import extract_joint_coordinates, finger_locator
from shared.dataclasses import coordinates

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Defines what each tensor means outputted from MLP
labels = ["Rock", "Paper", "Scissors"]

# Initializes the webcam and returns camera object
camera, height, width = webcam.webcam_init()

# Define text properties
org = (10, 30)           # (x, y) coordinates of the bottom-left corner of the text
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1.0           # Font size multiplier
color = (255, 255, 255)       # Text color in BGR (Green)
thickness = 2             # Line thickness
lineType = cv2.LINE_AA    # Anti-aliased line for smoother text rendering

# Define the Ai Architecture set when training
input_dim = 63
hidden_dim = 32
output_dim = 3

# Calls a function to load the model
model = load_model(input_dim, hidden_dim, output_dim)

# Initalizes variables needed to run game start function
fps = camera.get(cv2.CAP_PROP_FPS)
game_start_phase = 0
past_wrist_coords = []
current_wrist_coords = coordinates(0,0)
data_for_sec = 2  #seconds that wrist coords are stored for
min_amplitude = .15  #Wrist must move 15% of the screen for a game to be able to start

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.65,
        max_num_hands = 1
    ) as hands:
        
        while camera.isOpened():
            success, frame = camera.read() 
            
            if not success:
                continue
            
            # 1. Convert to RGB for MediaPipe processing
            # We treat 'frame' directly to keep memory references clean
            frame.flags.writeable = False
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            
            # Init the predicted class variable so we can use it anywhere inside the program
            predicted_class = None
            
            # 2. Allow drawing back on the original frame
            frame.flags.writeable = True
            
            # Loop for drawing onto hand if a hand that needs landmarks is present
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    WRIST_INDEX = 0 # index of wrist inside of the joints list
                    SCALE_JOINT_INDEX = 8 # index of Middle Finger MCP inside of the joints list

                    coords = extract_joint_coordinates(hand_landmarks, width, height)
                    
                    wrist = coords[WRIST_INDEX].copy()
                    coords-=wrist
                    
                    scale = float(np.linalg.norm(coords[SCALE_JOINT_INDEX]))
                    if scale < 1e-6:
                        scale = 1e-6
                    coords /= scale
                    
                    joint_list = coords.flatten().tolist()
                    
                    # Convert the normalizes joint list
                    input_tensor = torch.from_numpy(np.array(joint_list, dtype=np.float32)).unsqueeze(0)
                    
                    # Feed the input_tensor to the model
                    model.eval()
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    predicted_class = int(torch.argmax(output, dim=1).item())
                    
                    x, y = finger_locator([WRIST_INDEX], hand_landmarks, width, height)
                    current_wrist_coords = coordinates(x, y)
                    
                    draw_annotations(frame, hand_landmarks)
            
            # Print predicted class onscreen is hand is present
            if predicted_class is not None:
                cv2.putText(frame, f"Current hand prediction: {labels[predicted_class]}", org, fontFace, fontScale, color, thickness, lineType)
            else:
                cv2.putText(frame, "No hand detected", org, fontFace, fontScale, color, thickness, lineType)
                
            # Saves the past positions into an list
            past_wrist_coords.append(current_wrist_coords)
            # If past wrist cords exceeds 2 seconds worth of data then remove last index at
            if len(past_wrist_coords) < 2/fps:
                past_wrist_coords.pop(0)
            
            # Runs all this code once and allows for me to break
            flag = True
            while flag:
                
                if len(past_wrist_coords) == (data_for_sec*fps):
                    # Check total amplitude
                    y_amplitude = abs(past_wrist_coords[0].y - past_wrist_coords[-1].y)
                    x_amplitude = abs(past_wrist_coords[0].x - past_wrist_coords[-1].x)
                    if y_amplitude < (min_amplitude*height):
                        if x_amplitude < (min_amplitude*width):
                            game_start_phase += 1
                # Change the flag to false once the loop is over 
                flag = False
                
                flag = False
            
            # Display the frame AFTER drawing annotations
            cv2.imshow("Webcam Feed", frame)
            
            # Checks for a keypress of the 'q' key to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
camera.release()
cv2.destroyAllWindows() # Clean up window assets on exit
