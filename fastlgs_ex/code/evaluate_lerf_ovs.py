import os
import cv2 as cv
import numpy as np
import json
import torch
import open_clip
from shapely.geometry import Point, Polygon
from query_target import get_tar

label_json_path = "./lerf_ovs/frame_00140.json" # path to ground truth label
rendered_feat_path = "./lerf_ovs/00139.png" # path to rendered feature

feat = cv.imread(rendered_feat_path)
feat_width , feat_height = feat.shape[1], feat.shape[0]
label_dict = json.load(open(label_json_path))
print(label_dict["info"])
intersect = 0; union = 0
tol_IoU = 0.0; tol_obj = 0
for object in label_dict["objects"]:
    tarR, tarG, tarB = get_tar(object["category"])
    segmentation_points = object["segmentation"]
    polygon = Polygon(segmentation_points)
    for w in range(feat_width):
        for h in range(feat_height):
            B,G,R = feat[h,w,:]
            point = Point(w, h)
            if abs(R-tarR) < 5 and abs(G-tarG) < 5 and abs(B-tarB) < 5 and polygon.contains(point):
                intersect += 1
            if (abs(R-tarR) < 5 and abs(G-tarG) < 5 and abs(B-tarB) < 5) or polygon.contains(point):
                union += 1
    print("Category: ",object["category"],"Intersect: ", intersect, " Union: ", union, " IoU: ", intersect/union)
    tol_IoU += intersect/union
    tol_obj += 1
print("Total IoU: ", tol_IoU, " Total Obj: ", tol_obj, " MIoU: ", tol_IoU/tol_obj)