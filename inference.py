from utils import get_frames, get_batches
from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
import numpy as np
import os
import torch
import pdb
from config import device
import time
import cv2

video_path = "20240415_161511.mxf"

####### Load model
supernet_best_f1 = TransNetV2Supernet().eval()
# model path
pretrained_path = os.path.join('./ckpt_0_200_0.pth')
model_dict = supernet_best_f1.state_dict()
pretrained_dict = torch.load(pretrained_path, map_location=device)
pretrained_dict = {k:v for k,v in pretrained_dict['net'].items() if k in model_dict} 
model_dict.update(pretrained_dict)
supernet_best_f1.load_state_dict(model_dict)
supernet_best_f1 = supernet_best_f1.cuda(int(device[-1]))
supernet_best_f1.eval()
####### set threshold
threshold = 0.296
frames = get_frames(video_path)
predictions = [] 

def predict(batch):
    batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
    batch = batch.to(device)
    one_hot = supernet_best_f1(batch)
    if isinstance(one_hot, tuple):
        one_hot = one_hot[0]
    return torch.sigmoid(one_hot[0])
start_time = time.time()
for batch in get_batches(frames):
    one_hot = predict(batch)
    one_hot = one_hot.detach().cpu().numpy()
    predictions.append(one_hot[25:75])
end_time = time.time()
print(end_time-start_time)
predictions = np.concatenate(predictions,0)[:len(frames)]
# the index of shot boundary
indices = np.where(predictions>threshold)[0]

indices = np.append(indices, len(frames)-1)
indices = np.append(0, indices)
middle_frames_idx = [(indices[i]+indices[i+1])//2 for i in range(indices.shape[0]-1)]
####### read frame of the middle frame of a shot
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f'Error: Could not open video file{video_path}')
    
for idx in middle_frames_idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    while not ret and idx+1 < len(frames):
        print(f'can\'t read idx:{idx} frame from video_path:{video_path}')
        idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
    if not ret:
        print('has reached the end of video')
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pillow_image = Image.fromarray(rgb_frame)
