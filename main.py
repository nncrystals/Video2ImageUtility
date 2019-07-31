import cv2
import argparse
import logging
import pathlib
import numpy as np
import os

video_name = 'data.avi'
metadata_name = 'metadata_v7.3.mat'

parser = argparse.ArgumentParser()

parser.add_argument('--use_metadata', action='store_true', help='Use the metadata to put the images in different folders (search in the same directory)')
parser.add_argument('--fps', type=int, help='Specify a different fps than the video''s ', default=10)
parser.add_argument('video_or_dir', help='Path to video or the directory where video.avi lies in')
parser.add_argument('--output_dir', help='Output path. Default to ./exported_images', default='./exported_images')
args = parser.parse_args()

if os.path.isdir(args.video_or_dir):
    video_path = os.path.join(args.video_or_dir, video_name)
else:
    video_path = args.video_or_dir

video_dir_path = os.path.dirname(video_path)

metadata_time = None
metadata_state = None

if args.use_metadata:
    import h5py

    metadata_path = os.path.join(video_dir_path, metadata_name)
    matfile = h5py.File(metadata_path, 'r')
    metadata_time = matfile.get('ControllerState/time')[0]
    metadata_state = matfile.get('ControllerState/signals/values')[0]
    logging.info('Successfully loaded metadata: ' + metadata_path)


# read the video file
vc = cv2.VideoCapture(video_path)

if not vc.isOpened():
    logging.error('Error opening video file:' + video_path)
    exit(-1)


idx = 0
while True:
    ret, frame = vc.read()

    if not ret:
        logging.info('No more frames')
        break
    else:
        out_dir = None
        if args.use_metadata:
            secs = idx / args.fps
            time_idx = np.argmin(abs(secs - metadata_time))
            state = int(metadata_state[time_idx])
            out_dir = os.path.join(args.output_dir, str(state))
        pathlib.Path(out_dir).mkdir(exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f'{idx}.png'), frame)
        idx += 1
vc.release()