import cv2
import mediapipe as mp
import time
import math
import pygame
import os

"""
This program allows the user to control the music player using hand gestures.

Authors: Dariusz Karasiewicz, Mikołaj Kusiński
"""

class MusicPlayer:

    def __init__(self):
        """
        Initializes a music player object using Pygame library.

        This constructor initializes the Pygame library, sets up the music mixer,
        loads the music playlist, and sets the initial state of the music player.
        """
        pygame.init()
        pygame.mixer.init()
        self.playlist = self.load_music_playlist()
        self.current_track_index = 0
        self.playing = False

    def load_music_playlist(self):
        """
        Loads a music playlist from the specified folder.

        This method scans the "music" folder for MP3 files and creates a playlist
        containing the full paths to each music track.

        Returns:
        - playlist: A list of strings, each representing the full path to a music track.
        """
        music_folder = "music"
        music_files = [file for file in os.listdir(music_folder) if file.endswith(".mp3")]
        return [os.path.join(music_folder, file) for file in music_files]

    def play(self):
        """
        Plays the current music track.

        This method checks if the music mixer is not currently playing any music,
        loads the next track from the playlist, plays it, and updates the player's state.
        """
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(self.playlist[self.current_track_index])
            pygame.mixer.music.play()
            self.playing = True

    def stop(self):
        """
        Stops the currently playing music track.

        This method stops the playback of the current music track and updates
        the player's state to indicate that no music is currently playing.
        """
        pygame.mixer.music.stop()
        self.playing = False

    def next_track(self):
        """
        Plays the next music track in the playlist.

        This method stops the currently playing track, updates the index to the next track,
        and then plays the newly selected track.
        """
        self.stop()
        self.current_track_index = (self.current_track_index + 1) % len(self.playlist)
        self.play()

    def previous_track(self):
        """
        Plays the previous music track in the playlist.

        This method stops the currently playing track, updates the index to the next track,
        and then plays the newly selected track.
        """
        self.stop()
        self.current_track_index = (self.current_track_index - 1) % len(self.playlist)
        self.play()

def finger_up(finger_bottom, finger_tip) -> bool:
    """
    Determines if a finger is in an 'up' position.

    This function compares the positions of the bottom and tip of a finger and
    returns True if the finger is in an upward position, and False otherwise.

    Parameters:
    - finger_bottom: The y-coordinate of the bottom of the finger.
    - finger_tip: The y-coordinate of the tip of the finger.

    Returns:
    - bool: True if the finger is in an upward position, False otherwise.
    """
    if finger_bottom > finger_tip:
        return True
    else:
        return False

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

counter = 0
reset_interval = 50
exit_couter=0

last_move=None
player = MusicPlayer()
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    gesture_detected = False  # Flag to track if any gesture is detected
    if counter >= reset_interval:
        last_move = None
        counter = 0 

    counter += 1 
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4:  # Thumb tip
                    thumb_tip =  cy
                if id == 0:  # Thumb tip
                    thumb_bottom =  cy

                if id == 8:  # Index tip
                    index_tip =  cy
                if id == 5:  # Index tip
                    index_bottom =  cy

                if id == 9:  # Środek dłoni
                    middle_center = cy
                if id == 12:
                    middle_tip = cy
                
                if id == 13:
                    ring_bottom = cy
                if id == 16:
                    ring_tip = cy

                if id == 17:
                    small_bottom = cy
                if id == 20:
                    small_tip = cy
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw a line between the thumb and index finger

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            gesture_detected = True
            if finger_up(thumb_bottom,thumb_tip) and finger_up(small_bottom,small_tip) and finger_up(index_bottom, index_tip) and finger_up(middle_center, middle_tip):
                cv2.putText(img, "Hello", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if last_move != "🖐":
                    exit_couter=0
                    print("🖐")
                    last_move="🖐"
                    player.stop()
            elif finger_up(index_bottom,index_tip) and finger_up(small_bottom,small_tip):
                cv2.putText(img, "Satan", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if last_move != "🤘":
                    exit_couter=0
                    print("🤘")
                    last_move = "🤘"
                    player.next_track()
            elif finger_up(index_bottom,index_tip) and not finger_up(middle_center,middle_tip) and not finger_up(ring_bottom,ring_tip) and not finger_up(small_bottom, small_tip):
                cv2.putText(img, "Wskazujacy w gore", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if last_move != "☝️":
                    exit_couter=0
                    print("☝️")
                    last_move="☝️"
                    player.previous_track()
            elif thumb_tip < middle_center:
                cv2.putText(img, "Kciuk w gore", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if last_move != "👍":
                    exit_couter+=1
                    last_move="👍"
                    print("👍")
                elif exit_couter==2:
                    exit()                    

    if not gesture_detected:
        cv2.putText(img, "No gesture detected", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
