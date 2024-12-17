import os
import cv2 as cv
import numpy as np
from numpy.linalg import norm
import json
import torch
import open_clip
from PIL import Image

np_path = './data/TEATIME_CLIP_COLOR.npy'
st_path = './data/TEATIME_CLIP_EXTEND.npy'
gt_path = './data/gt_images'
img_path = './notebooks/videos/teatime'
npdata = np.load(np_path, allow_pickle=True)
npdata = npdata.tolist()
feat_dict = {}
new_clips = []; new_coords = []; new_img_nums = []
for info in npdata:
    clip_feat = np.array(info[0]); coord = tuple(info[1]); img_num = np.array(info[2])
    new_clips.append(clip_feat); new_coords.append(coord); new_img_nums.append(img_num)
    feat_dict[coord] = clip_feat
#print(feat_dict[(37,235,140)])
gt_files = sorted(os.listdir(gt_path))
img_files = sorted(os.listdir(img_path))
print("Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion2b_s34b_b88k",
    precision="fp16"
)
clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(clip_device)
output_path = './output_tiles'
os.makedirs(output_path, exist_ok=True)
idx = -1
for gt_file, img_file in zip(gt_files, img_files):
    idx+=1
    print(f"Processing frame {idx}...")
    gt_feat = cv.imread(os.path.join(gt_path, gt_file))#BGR
    gt_feat = cv.cvtColor(gt_feat, cv.COLOR_BGR2RGB)
    img = cv.imread(os.path.join(img_path, img_file))
    all_feats = np.unique(gt_feat.reshape(-1, 3), axis=0)
    all_feats = [tuple(feat) for feat in all_feats if tuple(feat) != (0, 0, 0)]
    for feat in all_feats:
        if feat not in feat_dict:
            print(f"{feat} not found in Dict")
            continue
        mask = cv.inRange(gt_feat, np.array(feat), np.array(feat))
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            tile = img[y:y+h, x:x+w]
            #output_filename = f"{gt_file.split('.')[0]}_feat_{feat}.png"
            #output_filepath = os.path.join(output_path, output_filename)
            #cv.imwrite(output_filepath, tile)
            tile_rgb = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
            tile_pil = Image.fromarray(tile_rgb)
            block = preprocess(tile_pil).unsqueeze(0).to(clip_device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                clip_feats = clip_model.encode_image(block)
                clip_feats /= clip_feats.norm(dim=-1, keepdim=True)
            new_clip = clip_feats.cpu().numpy()
            last_clip = feat_dict[feat]
            cosine_similarity = np.dot(new_clip[0], last_clip[0].T) / (norm(new_clip[0]) * norm(last_clip[0]))
            #print(f"Cosine similarity of {feat} in {gt_file}: {cosine_similarity}")
            if cosine_similarity < 0.8:
                feat_dict[feat] = new_clip
                new_clips.append(new_clip); new_coords.append(feat); new_img_nums.append(idx)
                print(f"Updated {feat} in {gt_file}")
                output_filename = f"{gt_file.split('.')[0]}_feat_{feat}.png"
                output_filepath = os.path.join(output_path, output_filename)
                cv.imwrite(output_filepath, tile)
np.save(st_path, zip(new_clips, new_coords, new_img_nums))