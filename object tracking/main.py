import gc
import os
import sys
import numpy as np
import torch
import cv2

sys.path.append("./sam2")

from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

def get_bbox_prompt(frames_dir, det_ckpt) :
    first_frame = frames_dir + "/1.jpg"

    detection_model = YOLO(detection_ckpt)
    det_result = detection_model(first_frame)[0]

    # Detected bounding boxes
    bboxes = det_result.boxes.xyxy
    bboxes_cpu = bboxes.to('cpu').numpy()

    prompts = {}

    for idx, bbox in enumerate(bboxes_cpu):
        prompts[idx] = ((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), 0)
    
    print(prompts)

    return prompts

if __name__ =="__main__" :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frames_dir = 'data/frames/'
    output_path = 'result.mp4'

    cwd = os.getcwd()
    print(cwd)
    detection_ckpt = "ckpts/yolo11s.pt"
    samurai_cfg = os.path.join(cwd + "/sam2/sam2/configs/sam2/sam2_hiera_s.yaml")
    samurai_ckpt = "sam2/checkpoints/sam2_hiera_small.pt"

    samurai_predictor = build_sam2_video_predictor(samurai_cfg, samurai_ckpt, device=device)

    prompts = get_bbox_prompt(frames_dir, detection_ckpt)
    
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)])
    loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
    h, w, _ = loaded_frames[0].shape
    frame_rate = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, frame_rate, (w, h))

    # Inference
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        initial_state = samurai_predictor.init_state(frames_dir, offload_video_to_cpu=True)
        bbox, tracking_label = prompts[1]

        _, _, masks = samurai_predictor.add_new_points_or_box(initial_state, box=bbox, frame_idx=0, obj_id=16)

        for frame_idx, object_ids, masks in samurai_predictor.propagate_in_video(initial_state):
            mask_visualize = {}
            bbox_visualize = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)

                if len(non_zero_indices) == 0:
                    bbox = [0,0,0,0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max, y_max]
                
                bbox_visualize[obj_id] = bbox
                mask_visualize[obj_id] = mask

            # Visualize
            frame_img = loaded_frames[frame_idx]
            color = [(255,0,0)]

            # 1. Visualize mask
            for obj_id, mask in mask_visualize.items():
                mask_img = np.zeros((h, w, 3), np.uint8)
                # Put color only on mask
                mask_img[mask] = color[(obj_id + 1) % len(color)]       
                frame_img = cv2.addWeighted(frame_img, 1, mask_img, beta=0.7, gamma=0)   
            
            # 2. Visualize bbox
            for obj_id, mask in bbox_visualize.items():
                cv2.rectangle(frame_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color[obj_id % len(color)], thickness=2)

            output_video.write(frame_img)

        output_video.release()

    del samurai_predictor, initial_state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()