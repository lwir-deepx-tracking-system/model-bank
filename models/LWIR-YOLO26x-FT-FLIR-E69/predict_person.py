"""Run the detector on an 8-bit thermal image and return source-image boxes."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='8-bit thermal image')
    parser.add_argument('--weights', default=str(Path(__file__).with_name('yolo26x_flir_ft_e69.pt')))
    parser.add_argument('--conf', type=float, default=0.25, help='candidate threshold; tune on a labeled development set')
    parser.add_argument('--device', default='cpu', help='cpu or CUDA device, e.g. 0')
    args = parser.parse_args()
    with Image.open(args.image) as im:
        if im.mode not in ('L', 'RGB', 'RGBA'):
            raise ValueError('Use a verified 8-bit thermal image; 16-bit conversion is dataset/camera dependent')
        gray = np.asarray(im.convert('L'))
    h, w = gray.shape
    ratio = min(480 / h, 640 / w)
    nw, nh = round(w * ratio), round(h * ratio)
    left = round((640 - nw) / 2 - 0.1)
    top = round((480 - nh) / 2 - 0.1)
    canvas = np.full((480, 640), 114, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    image = np.repeat(canvas[:, :, None], 3, axis=2)
    model = YOLO(args.weights, task='detect')
    result = model.predict(image, imgsz=(480, 640), rect=False, conf=args.conf,
                           max_det=300, device=args.device, verbose=False)[0]
    detections = []
    for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        if int(cls) != 0:
            continue
        x1 = max(0., min(float(w), (float(box[0]) - left) / ratio))
        y1 = max(0., min(float(h), (float(box[1]) - top) / ratio))
        x2 = max(0., min(float(w), (float(box[2]) - left) / ratio))
        y2 = max(0., min(float(h), (float(box[3]) - top) / ratio))
        if x2 > x1 and y2 > y1:
            detections.append({'class_id': 0, 'class_name': 'person', 'confidence': float(score),
                               'xywh': [x1, y1, x2-x1, y2-y1]})
    print(json.dumps({'image':args.image, 'image_size_wh':[w,h], 'detections':detections}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
