import cv2
import os
import numpy as np
import json
import argparse
from tqdm import trange


def create_video(images_folder, output_video, fps, font_scale, text_color, text_position):
    images = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front')) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    frame = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i in trange(1, len(images)):
        image = images[i]
        f = open(os.path.join(images_folder, f'meta/{i:04}.json'), 'r')
        meta = json.load(f)
        steer = float(meta['steer'])
        throttle = float(meta['throttle'])
        brake = float(meta['brake'])
        # command = float(meta['command'])
        # command_list = ["VOID", "LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT",  "CHANGE LANE RIGHT",]
        speed = float(meta['speed'])
        text = f'speed: {round(speed,2)}, steer: {round(steer,2)}, throttle: {round(throttle,2)}, brake: {round(brake,2)}'#, command: {command_list[int(command)]}'
        img = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), image))
        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
        video.write(img)
    video.release()

parser = argparse.ArgumentParser(description='Generate video from closed-loop evaluation results')
parser.add_argument('-f', '--folder', required=True,
                    help='Path to the scenario folder (parent of rgb_front/ and meta/)')
parser.add_argument('-o', '--output', default=None,
                    help='Output video path (default: <folder>/output.mp4)')
parser.add_argument('--fps', type=int, default=15, help='Frames per second (default: 15)')
args = parser.parse_args()

images_folder = args.folder
output_video = args.output if args.output else os.path.join(images_folder, 'output.mp4')
fps = args.fps
font_scale = 1
text_color = (255, 255, 255)
text_position = (50, 50)

create_video(images_folder, output_video, fps, font_scale, text_color, text_position)
