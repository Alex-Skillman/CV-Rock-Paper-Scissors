import webcam # Assumes your custom wrapper works perfectly
import cv2
import mediapipe as mp # type: ignore
import torch # type: ignore
import numpy as np
import time
from shared.Data.fingerNames import joints
from shared.functions.mediapipe_funcs import draw_annotations
from shared.functions.RPS_ai_funcs import load_model
from shared.functions.joint_pos_funcs import extract_joint_coordinates

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Defines what each tensor means outputted from MLP
labels = ["Rock", "Paper", "Scissors"]
winning_moves = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}

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
game_start_phase = 0
beat_times = []
last_wrist_y = None
last_wrist_x = None
last_direction = 0
last_turn_y = None
last_turn_x = None
smoothed_wrist_y = None
smoothed_wrist_x = None
min_start_beat_distance = 0.04
direction_dead_zone = 0.002
third_beat_tolerance = 0.65
vertical_movement_ratio = 1.5
required_start_beats = 4
# This is set after beat two and used to validate beat three.
average_time_between_first_two = None
tracking_lost_at = None
tracking_grace_period = 0.75
# The computer-play overlay uses the move that beats the confirmed player move.
computer_play = None
show_computer_play = False
computer_play_timeout = 2.0
computer_play_shown_at = None
required_positive_frames = 3
positive_move_label = None
positive_move_frames = 0
waiting_for_move_prediction = False

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.1,
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
                    
                    draw_annotations(frame, hand_landmarks)
            
            # Print predicted class onscreen is hand is present
            if predicted_class is not None:
                cv2.putText(frame, f"Current hand prediction: {labels[predicted_class]}", org, fontFace, fontScale, color, thickness, lineType)
            else:
                cv2.putText(frame, "No hand detected", org, fontFace, fontScale, color, thickness, lineType)

            if results.multi_hand_landmarks:
                now = time.monotonic()
                if tracking_lost_at is not None:
                    if now - tracking_lost_at > tracking_grace_period:
                        beat_times = []
                        average_time_between_first_two = None
                    tracking_lost_at = None

                raw_wrist_y = results.multi_hand_landmarks[0].landmark[0].y
                raw_wrist_x = results.multi_hand_landmarks[0].landmark[0].x
                if smoothed_wrist_y is None:
                    smoothed_wrist_y = raw_wrist_y
                    smoothed_wrist_x = raw_wrist_x
                else:
                    smoothed_wrist_y = 0.65 * smoothed_wrist_y + 0.35 * raw_wrist_y
                    smoothed_wrist_x = 0.65 * smoothed_wrist_x + 0.35 * raw_wrist_x
                wrist_y = smoothed_wrist_y
                wrist_x = smoothed_wrist_x

                if last_wrist_y is not None and last_wrist_x is not None:
                    wrist_delta = wrist_y - last_wrist_y

                    # Ignore tiny landmark jitter, then count a real down-to-up
                    # wrist movement as one RPS start beat.
                    if abs(wrist_delta) >= direction_dead_zone:
                        direction = 1 if wrist_delta > 0 else -1
                        if last_turn_y is not None and last_turn_x is not None:
                            vertical_distance = abs(wrist_y - last_turn_y)
                            horizontal_distance = abs(wrist_x - last_turn_x)
                            is_vertical_beat = vertical_distance >= horizontal_distance * vertical_movement_ratio
                            if last_direction == 1 and direction == -1 and vertical_distance >= min_start_beat_distance and is_vertical_beat:
                                beat_times.append(now)

                                if len(beat_times) == 2:
                                    average_time_between_first_two = beat_times[1] - beat_times[0]
                                elif len(beat_times) >= 3:
                                    current_beat_time = beat_times[-1] - beat_times[-2]
                                    if abs(current_beat_time - average_time_between_first_two) <= third_beat_tolerance:
                                        if len(beat_times) == required_start_beats:
                                            game_start_phase += 1
                                            print("Game started")
                                            waiting_for_move_prediction = True
                                            positive_move_label = None
                                            positive_move_frames = 0
                                            computer_play = None
                                            show_computer_play = False
                                            computer_play_shown_at = None
                                            beat_times = []
                                            average_time_between_first_two = None
                                    else:
                                        # A beat outside the established rhythm starts a new sequence.
                                        beat_times = []
                                        average_time_between_first_two = None

                        if direction != last_direction:
                            last_turn_y = wrist_y
                            last_turn_x = wrist_x
                        last_direction = direction

                last_wrist_y = wrist_y
                last_wrist_x = wrist_x
            else:
                if tracking_lost_at is None:
                    tracking_lost_at = time.monotonic()
                    # Keep the completed beats, but do not compare coordinates
                    # from before and after a tracking gap.
                    last_wrist_y = None
                    last_wrist_x = None
                    last_direction = 0
                    last_turn_y = None
                    last_turn_x = None
                    smoothed_wrist_y = None
                    smoothed_wrist_x = None
                elif time.monotonic() - tracking_lost_at > tracking_grace_period:
                    beat_times = []
                    average_time_between_first_two = None

            # Draw the game-start phase on screen.
            start_phase = len(beat_times)
            phase_label = f"Game start: move down {required_start_beats} times" if start_phase == 0 else f"Game start: beat {start_phase}/{required_start_beats}"
            cv2.putText(frame, phase_label, (10, 65), fontFace, 0.7, color, thickness, lineType)
            for beat_index in range(required_start_beats):
                beat_color = (0, 255, 0) if beat_index < start_phase else (100, 100, 100)
                cv2.circle(frame, (25 + beat_index * 30, 92), 9, beat_color, -1)

            # After game start, require the same network prediction for several
            # consecutive frames before displaying it.
            if waiting_for_move_prediction:
                if predicted_class is None:
                    positive_move_label = None
                    positive_move_frames = 0
                else:
                    predicted_move = labels[predicted_class]
                    if predicted_move == positive_move_label:
                        positive_move_frames += 1
                    else:
                        positive_move_label = predicted_move
                        positive_move_frames = 1

                    if positive_move_frames >= required_positive_frames:
                        computer_play = winning_moves[predicted_move]
                        show_computer_play = True
                        computer_play_shown_at = None
                        waiting_for_move_prediction = False

            # Show the confirmed neural-network prediction across the screen,
            # then hide it after the configured timeout.
            if show_computer_play and computer_play is not None:
                if computer_play_shown_at is None:
                    computer_play_shown_at = time.monotonic()

                if time.monotonic() - computer_play_shown_at < computer_play_timeout:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                    display_text = computer_play.upper()
                    text_size, _ = cv2.getTextSize(display_text, fontFace, 4.0, 8)
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    cv2.putText(frame, display_text, (text_x, text_y), fontFace, 4.0, (255, 255, 255), 8, lineType)
                else:
                    show_computer_play = False
            
            # Display the frame AFTER drawing annotations
            cv2.imshow("Webcam Feed", frame)
            
            # Checks for a keypress of the 'q' key to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
camera.release()
cv2.destroyAllWindows() # Clean up window assets on exit
