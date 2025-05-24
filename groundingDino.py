import os
import gc
import cv2
import torch
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
from groundingdino.util.inference import load_model, load_image, predict, annotate


class groundingDino:
    def __init__(self):
        self.CONFIG_PATH = "/home/m_m58330/Moh1996_Python_Projects/VQA/CVPR/LAVIS/projects/img2prompt-vqa/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.WEIGHTS_PATH = "/home/m_m58330/Moh1996_Python_Projects/VQA/CVPR/LAVIS/projects/img2prompt-vqa/Weights/groundingdino_swint_ogc.pth"
        self.model_dino = load_model(self.CONFIG_PATH, self.WEIGHTS_PATH)


    def bb_intersection_ratio(self, boxA, boxB, A, B, remover_largest):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = (abs(max((xB - xA, 0)) * max((yB - yA), 0)))
        """if xA > xB and yA > yB:
            interArea = (xB - xA) * (yB - yA)
        else:
            interArea = 0"""
        if interArea == 0:
            interArea = 0
        #interArea = (abs(xB - xA) * abs(yB - yA)).numpy()
        boxA_area = (abs(boxA[0] - boxA[2]) * abs(boxA[1] - boxA[3])).numpy()
        boxB_area = (abs(boxB[0] - boxB[2]) * abs(boxB[1] - boxB[3])).numpy()

        ratio = interArea / min(boxA_area, boxB_area)

        if not remover_largest:
            if min(boxA_area, boxB_area) == boxA_area:
                return ratio, A
            else:
                return ratio, B
        else:
            if min(boxA_area, boxB_area) == boxA_area:
                return ratio, B
            else:
                return ratio, A

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)


    def box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)


    def removing_overlapping_bb(self, image, bbox, logits, phrases, remover_largest = False, remove_th = 0.9):
        H, W, _ = image.shape
        bbox = self.box_cxcywh_to_xyxy(bbox) * torch.Tensor([W, H, W, H])

        row_lists_to_be_removed = []
        for i in range(bbox.shape[0]):
            for j in range(i+1, bbox.shape[0]):
                IoU, row = self.bb_intersection_ratio(list(bbox[i,:]), list(bbox[j,:]), i , j, remover_largest)
                if (IoU > remove_th):
                    row_lists_to_be_removed.append(row)

        bbox = self.box_xyxy_to_cxcywh(bbox)/torch.Tensor([W, H, W, H])

        return np.delete(bbox, row_lists_to_be_removed, axis=0), np.delete(logits, row_lists_to_be_removed), [item for i, item in enumerate(phrases) if i not in row_lists_to_be_removed]


    def coeff(self, x):
        x *= 100
        return ((-100/29)*x + 3100/29)/100
    

    def areaOfInterest(self, image_source, boxes):

        img = image_source

        H, W, _ = img.shape
        boxes_xyxy = self.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])


        x1 = min(boxes_xyxy[:,0])
        y1 = min(boxes_xyxy[:,1])
        x2 = max(boxes_xyxy[:,2])
        y2 = max(boxes_xyxy[:,3])


        image_area = W * H
        bbox_area = np.abs(x1-x2) * np.abs(y1-y2)
        ratio = bbox_area / (image_area)

        if (ratio.numpy() > 0.3):
            c = 0
        else:
            c = self.coeff(ratio.numpy())

        x1 -= c*0.30*np.abs(x1 - x2)
        y1 -= c*0.30*np.abs(y1 - y2)
        x2 += c*0.30*np.abs(x1 - x2)
        y2 += c*0.30*np.abs(y1 - y2)
        if (x1 < 0): x1 = torch.tensor(0)
        if (y1 < 0): y1 = torch.tensor(0)
        if (x2 > W-1): x2 = torch.tensor(W-1)
        if (y2 > H-1): y2 = torch.tensor(H-1)


        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color=(255,255,255), thickness=2)
        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]


        return cropped_img



    def bboxPredictor(self, img_adr, prompt):

        IMAGE_PATH = img_adr
        TEXT_PROMPT = prompt
        BOX_TRESHOLD = 0.25
        TEXT_TRESHOLD = 0.95

        image_source, image = load_image(IMAGE_PATH)


        boxes, logits, phrases = predict(
            model=self.model_dino,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )



        boxes, logits, phrases = self.removing_overlapping_bb(image_source, boxes, logits, phrases, False)

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        
        if boxes.numel():
            return self.areaOfInterest(image_source, boxes)
        else:
            return image_source
        #sv.plot_image(annotated_frame, (12, 12))

