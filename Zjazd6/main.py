import cv2
import mediapipe as mp
import time
import math
import pygame
import os

class MusicPlayer:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.playlist = self.load_music_playlist()
        self.current_track_index = 0
        self.playing = False

    def load_music_playlist(self):
        music_folder = "music"
        music_files = [file for file in os.listdir(music_folder) if file.endswith(".mp3")]
        return [os.path.join(music_folder, file) for file in music_files]

    def play(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(self.playlist[self.current_track_index])
            pygame.mixer.music.play()
            self.playing = True

    def stop(self):
        pygame.mixer.music.stop()
        self.playing = False

    def next_track(self):
        self.stop()
        self.current_track_index = (self.current_track_index + 1) % len(self.playlist)
        self.play()

    def previous_track(self):
        self.stop()
        self.current_track_index = (self.current_track_index - 1) % len(self.playlist)
        self.play()

# Inicjalizacja odtwarzacza

# while True:
#     print("1. Play/Pause")
#     print("2. Stop")
#     print("3. Next Track")
#     print("4. Previous Track")
#     print("5. Quit")

#     choice = input("Wybierz opcję: ")

#     if choice == "1":
#         if player.playing:
#             pygame.mixer.music.pause()
#             player.playing = False
#         else:
#             pygame.mixer.music.unpause()
#             player.playing = True
#     elif choice == "2":
#         player.stop()
#     elif choice == "3":
#         player.next_track()
#     elif choice == "4":
#         player.previous_track()
#     elif choice == "5":
#         player.stop()
#         break
#     else:
#         print("Niepoprawny wybór. Wybierz liczbę od 1 do 5.")


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

last_move=None
player = MusicPlayer()
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    gesture_detected = False  # Flag to track if any gesture is detected
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4:  # Thumb tip
                    thumb_tip = (cx, cy)

                if id == 8:  # Index tip
                    index_tip = (cx, cy)

                if id == 9:  # Środek dłoni
                    palm_center = cy
                # print(id, cx, cy)
                if id == 12:
                    middle_tip = cy
                
                if id == 16:
                    ring_tip = cy

                if id == 20:
                    small_tip = cy
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw a line between the thumb and index finger

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            gesture_detected = True

            if thumb_tip[0] < palm_center and small_tip < palm_center and ring_tip < palm_center and index_tip[0] < palm_center and middle_tip < palm_center:
                cv2.putText(img, "Hello", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if last_move != "🖐":
                    print("🖐")
                    last_move="🖐"
                    player.stop()
            elif index_tip[0] < palm_center and middle_tip >= palm_center:
                cv2.putText(img, "Wskazujacy w gore", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if last_move != "☝️":
                    print("☝️")
                    last_move="☝️"
                    player.previous_track()
            elif middle_tip < palm_center and index_tip[0] < palm_center:
                cv2.putText(img, "Znak Pokoju", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if last_move != "✌":
                    last_move="✌️"
                    print("✌️")
                    player.next_track()
            elif thumb_tip[0] < palm_center:
                cv2.putText(img, "Kciuk w gore", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if last_move != "👍":
                    last_move="👍"
                    print("👍")
                    break
    if not gesture_detected:
        cv2.putText(img, "No gesture detected", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)