import cv2
import openai
#from openai import OpenAI
import asyncio
import json
import base64
import os
import subprocess
import shutil
import time
import ollama
from typing import Optional, List, Any

#
# Video Understanding via Llama
# Llamacon hackathon 2025 project
# Split frames out of the video based on scene changes
# Send each frame to the Scout model for Summarization
# Concatenate all frame descriptions and send to the 70b model for video summarization
#
openai.api_key = os.getenv("LLAMA_API_KEY")
openai.base_url="https://api.llama.com/compat/v1/"
openai.api_type="openai"

SYS_PROMPT="You are a world class analyzer of images and can accurate detect objects in a scene and infer what may be going on in the scene 2 seconds before and 2 seconds afterwards"
GEN_PROMPT="Tell me what is going on in this scene"

SYS_PROMPT_SUMM="You are a world class summarizer of image descriptions that consistent different frames of a video.  You can accurate infer and summarize what is going on given a series of descriptions of different frames taken from a video.  Each description is delimited by the text 'Frame X response is:' where X is the frame number and tells you the chronological order of the frame."
GEN_PROMPT_SUMM=""

accum_vision = 0
attempts_vision = 0
# Function to encode the image
def encode_image(image_path: str) -> str:
    """Encode the image at the given path to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def call_vision_model(base64_image) -> str:
    """Call the vision model with the provided image file."""
    global accum_vision, attempts_vision
    #base64_image = encode_image(image_file)
    start_time = time.time()
    
    response = openai.chat.completions.create(
        model="Llama-4-Scout-17B-16E-Instruct-FP8",
        messages=[
            {"role": "system", "content": [
                {
                    "type": "text",
                    "text": f"{SYS_PROMPT}"
                }
            ]},
            {
                "role": "user", "content": [
                    {
                        "type": "text",
                        "text": f"{GEN_PROMPT}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]
    )

    duration_time = time.time() - start_time
    accum_vision += duration_time
    attempts_vision += 1
    print(f"call_vision execution time: {duration_time:.2f} seconds")
    print(f"call_vision average: {accum_vision / attempts_vision:.2f} seconds")
    
    return response.choices[0].message.content

def process_frame(frame):
    success, buffer = cv2.imencode('.jpg', frame)
    if success:
        jpg_bytes = buffer.tobytes()
        base64_str = base64.b64encode(jpg_bytes).decode('utf-8')
        # Now base64_str contains the base64-encoded JPEG image
    else:
        print("Failed to encode frame")

    return base64_str

frame_descriptions = ""

def split_by_scene(video_path, threshold=1):
    global frame_descriptions
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_hist = cv2.calcHist([prev_frame], [0,1,2], None, [8,8,8],[0, 256, 0, 256, 0, 256])
    prev_hist=cv2.normalize(prev_hist, prev_hist).flatten()

    scene_num = 1
    while True:
        ret, frame = cap.read()
        if (scene_num == 1):
            ret_response = call_vision_model(process_frame(frame))
            frame_response = f"Frame {scene_num} response is: {ret_response}"
            frame_descriptions += frame_response
            print(f"\n =============== Frame {scene_num} ============ \n{frame_response}")
        if not ret:
            print(f"{scene_num} total scenes analyzed")
            break # we're done?

        curr_hist = cv2.calcHist([frame], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        curr_hist=cv2.normalize(curr_hist, curr_hist).flatten()

        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        if diff > threshold:
            # Scene change detected
            print(f"Scene change detected at frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")
            ret_response = call_vision_model(process_frame(frame))
            frame_response = f"Frame {scene_num} response is: {ret_response}"
            frame_descriptions += frame_response
            print(f"\n =============== Frame {scene_num} ============ \n{frame_response}")

        prev_frame = frame.copy()
        prev_hist = curr_hist.copy()
        scene_num += 1

    cap.release()

def summarize(frame_descrptions: str):
    start_time = time.time()
    response = openai.chat.completions.create(
        model="Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": [
                {
                    "type": "text",
                    "text": f"{SYS_PROMPT_SUMM}"
                }
            ]},
            {
                "role": "user", "content": [
                    {
                        "type": "text",
                        "text": f"{frame_descriptions}",
                    },
                ],
            }
        ]
    )

    duration_time = time.time() - start_time
    print(f"call_vision execution time: {duration_time:.2f} seconds")
    
    return response.choices[0].message.content    

videos = {
    "desktop":"/Users/dpang/tmp/monitor_4_2024-11-29_07-08-53.mp4",
    "diving":"/Users/dpang/Downloads/Diving.M4V",
    "tennis":"/Users/dpang/Downloads/tennis.MP4",
    "flame":"/Users/dpang/Downloads/BMFlameTruck.MOV",
    "bourne":"/Users/dpang/Downloads/Bourne.MOV"
}
split_by_scene(videos["bourne"], 0.5)
summary = summarize(frame_descriptions)

print(f"\n ==========  SUMMARY ===========\n{summary}")

            
