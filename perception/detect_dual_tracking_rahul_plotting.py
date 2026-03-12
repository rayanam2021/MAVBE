import argparse
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np
from collections import deque
import cv2

# ---------- PATH SETUP ----------
# Path to this script
FILE = Path(__file__).resolve()
SCRIPT_ROOT = FILE.parents[0]  # folder containing this script

# Repo root where deep_sort and yolov9 live
REPO_ROOT = SCRIPT_ROOT  # adjust if your repo root is elsewhere

# YOLO repo path
YOLO_ROOT = REPO_ROOT / "yolov9"  # path to your YOLO repo

# deep_sort repo path
DEEP_SORT_ROOT = REPO_ROOT / "deep_sort"  # path to your deep_sort repo

# Add to sys.path in order: deep_sort first, then YOLO
for path in [DEEP_SORT_ROOT, YOLO_ROOT, REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ---------- IMPORTS ----------
# Now Python can find deep_sort and yolov9 modules
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Directory containing this script (repo root)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# YOLO utils
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, increment_path,
                           non_max_suppression, print_args, scale_boxes, strip_optimizer)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

import torchvision.transforms as T   # <-- Add this import here
import torchreid


# Global buffer for trails (last 64 points) and full trajectories (all points for plot)
data_deque = {}
full_trajectories = {}

# Class names (COCO)
def classNames():
    cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    return cocoClassNames
className = classNames()

def colorLabels(classid):
    if classid == 0: #person
        color = (85, 45, 255)
    elif classid == 2: #car
        color = (222, 82, 175)
    elif classid == 3: #Motorbike
        color = (0, 204, 255)
    elif classid == 5: #Bus
        color = (0,149,255)
    else:
        color = (200, 100,0)
    return tuple(color)

import matplotlib.pyplot as plt


def load_gt_trajectories(gt_path):
    """
    Load MOT-format GT file (frame,id,left,top,width,height) and return
    gt_trajectories[id] = [(cx, cy), ...] in frame order (center of bbox).
    """
    gt_trajectories = {}
    path = Path(gt_path)
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            tid = int(parts[1])
            tid = 3 - tid if tid in [1, 2] else tid #HARDOCDED NOTE TAKE CARE BAD
            left, top, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cx = left + w / 2.0
            cy = top + h / 2.0
            if tid not in gt_trajectories:
                gt_trajectories[tid] = []
            gt_trajectories[tid].append((frame_idx, cx, cy))
    for tid in gt_trajectories:
        gt_trajectories[tid].sort(key=lambda x: x[0])
        gt_trajectories[tid] = [(x, y) for _, x, y in gt_trajectories[tid]]
    return gt_trajectories


def save_trajectories(full_trajectories, save_path='trajectories.png', gt_trajectories=None, width=1280, height=720):
    plt.figure(figsize=(12, 8))
    for track_id, points in full_trajectories.items():
        if len(points) == 0:
            continue
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], marker='o', markersize=2, label='Pred ID %d' % track_id)
    if gt_trajectories:
        for gt_id, points in gt_trajectories.items():
            if len(points) == 0:
                continue
            points = np.array(points)
            plt.plot(
                points[:, 0], points[:, 1],
                linestyle='--', marker='s', markersize=2, alpha=0.8,
                label='GT ID %d' % gt_id,
            )
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    plt.xlabel('X pixels')
    plt.ylabel('Y pixels')
    plt.title('Object Trajectories (predicted and ground truth)')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    print("[INFO] Trajectory plot saved to %s" % save_path)
    plt.close()

def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0,0)):
    height, width, _ = frame.shape
    for key in list(data_deque):
        if identities is None or key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]; y1 += offset[1]; x2 += offset[0]; y2 += offset[1]
        center = int((x1+x2)/2), int((y1+y2)/2)
        cat = int(categories[i]) if categories is not None else 0
        color = colorLabels(cat)
        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        data_deque[id].appendleft(center)
        if id not in full_trajectories:
            full_trajectories[id] = []
        full_trajectories[id].append(center)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{id}:{className[cat]}"
        text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
        c2 = x1 + text_size[0], y1 - text_size[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, color, -1)
        cv2.putText(frame, label, (x1, y1-2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, 2, (0,255,0), cv2.FILLED)

        if draw_trails:
            for j in range(1, len(data_deque[id])):
                if data_deque[id][j-1] is None or data_deque[id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j+j)) * 1.5)
                cv2.line(frame, data_deque[id][j-1], data_deque[id][j], color, thickness)
    return frame

@smart_inference_mode()
def run(weights=ROOT / 'yolo.pt', save_plot_name="yash", source=ROOT / 'data/images', data=ROOT / 'data/coco.yaml',
        imgsz=(640,640), conf_thres=0.75, iou_thres=0.85, max_det=1000,
        device='', view_img=False, nosave=False, draw_trails=False,
        project=ROOT / 'runs/detect', name='exp', exist_ok=False,
        half=False, dnn=False, vid_stride=1, gt=''):

    source = str(source)
    save_img = not nosave
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://','rtmp://','http://','https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    # Directories
    save_dir = increment_path(Path(project)/name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    full_trajectories.clear()
    plot_width, plot_height = 1280, 720

    # ---------------- VIDEO WRITER SETUP ----------------
    save_video_path = ROOT/f'runs/detect/trajectories_{save_plot_name}.mp4'
    vid_writer = None
    # ----------------------------------------------------

    # Load YOLO model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)



    # Initialize ReID model
    reid_model = torchreid.models.build_model(
        name='osnet_x1_0',   
        num_classes=1000,
        pretrained=True
    )

    reid_model.eval()
    reid_model.to(device)

    reid_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),   # standard ReID size
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # vid_path, vid_writer = [None]*bs, [None]*bs

    # Initialize in-repo tracker (Behavioral EKF)
    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance=0.2, nn_budget=100)
    metric = nn_matching.NearestNeighborDistanceMetric(
        metric="cosine",
        matching_threshold=0.5,  # this was max_cosine_distance
        budget=100               # this was nn_budget
    )
    tracker = Tracker(metric)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        # print("here")
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # Inference
        with dt[1]:
            pred = model(im, augment=False)[0][1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            ims = im0.copy()
            plot_height, plot_width = ims.shape[:2]

            # Initialize video writer once (for first frame)
            if vid_writer is None:
                if vid_cap:  # if input is a video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                else:
                    fps = 30  # default fps if webcam or images

                h, w = ims.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_writer = cv2.VideoWriter(str(save_video_path), fourcc, fps, (w, h))




            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                xywh_bboxs, confs, oids = [], [], []

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    w, h = abs(x2-x1), abs(y2-y1)
                    xywh_bboxs.append([cx, cy, w, h])
                    confs.append(float(conf))
                    oids.append(int(cls))

                # Convert YOLO detections → Deep SORT Detection objects
                # detections = []
                # for j in range(len(xywh_bboxs)):
                #     cx, cy, w, h = xywh_bboxs[j]
                #     conf = confs[j]
                #     x_tl = cx - w/2
                #     y_tl = cy - h/2
                #     tlwh = [x_tl, y_tl, w, h]
                #     # feature = np.zeros(128)
                #     # feature = np.ones(128) * 1e-6
                #     # detections.append(Detection(tlwh, conf, feature))

                detections = []

                for j in range(len(xywh_bboxs)):
                    cx, cy, w, h = xywh_bboxs[j]
                    conf = confs[j]
                    x_tl = int(cx - w/2)
                    y_tl = int(cy - h/2)
                    w = int(w)
                    h = int(h)

                    # Crop detection from original frame
                    crop = im0[y_tl:y_tl+h, x_tl:x_tl+w]

                    if crop.size == 0:
                        continue

                    # Convert BGR → RGB
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                    # Apply transform
                    input_tensor = reid_transform(crop).unsqueeze(0).to(device)

                    with torch.no_grad():
                        features = reid_model(input_tensor)

                    feature = features.cpu().numpy().flatten()
                    feature = feature / np.linalg.norm(feature)

                    tlwh = [x_tl, y_tl, w, h]
                    detections.append(Detection(tlwh, conf, feature))



                print(len(detections))
                # Update tracker
                tracker.predict()
                tracker.update(detections)

                # Collect outputs
                outputs = []
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    x1, y1, w, h = track.to_tlwh()
                    x2, y2 = x1 + w, y1 + h
                    outputs.append([x1, y1, x2, y2, track.track_id, 0])
                outputs = np.array(outputs)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    object_id = outputs[:, 5]
                    draw_boxes(ims, bbox_xyxy, draw_trails, identities, object_id)
                    if vid_writer is not None:
                        vid_writer.write(ims)

            # Show image
            if view_img and False:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), ims.shape[1], ims.shape[0])
                cv2.imshow(str(p), ims)
                cv2.waitKey(1)

    if vid_writer is not None:
        vid_writer.release()
        print(f"[INFO] Video saved to {save_video_path}")

    gt_trajectories = None
    if gt:
        gt_trajectories = load_gt_trajectories(gt)
        if gt_trajectories:
            print("[INFO] Loaded GT trajectories for %d IDs" % len(gt_trajectories))
    save_trajectories(
        full_trajectories,
        save_path=ROOT / ('runs/detect/trajectories_%s.png' % save_plot_name),
        gt_trajectories=gt_trajectories,
        width=plot_width,
        height=plot_height,
    )

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov9/weights/yolov9-c.pt', help='model path')
    parser.add_argument('--save_plot_name', type=str, required=True)
    parser.add_argument('--gt', type=str, default='', help='Optional: MOT-format GT file to plot ground-truth trajectories with predictions')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'yolov9/data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.75)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--draw-trails', action='store_true')
    parser.add_argument('--project', default=ROOT / 'runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)