import os
import cv2 as cv
import numpy as np
from numpy.linalg import norm
import json
import torch
import open_clip
from PIL import Image

#query = 'paper napkin'
st_path = './data/TEATIME_CLIP_COLOR.npy' # path to the saved mappings
npdata = np.load(st_path, allow_pickle=True)
npdata = npdata.tolist()
clip_list = []; coord_list = []; img_num_list = []
for mapping in npdata:
    clip_list.append(np.array(mapping[0])); coord_list.append(np.array(mapping[1])); img_num_list.append(np.array(mapping[2]))
print("Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion2b_s34b_b88k",
    precision="fp16"
)
clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(clip_device)
tokenizer = open_clip.get_tokenizer("ViT-B-16")

def get_tar(query):
    print(">>> Query:",query)
    clip_text = tokenizer(["object", "stuff", "texture", str(query)])
    clip_text = clip_text.to(clip_device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        text_features = clip_model.encode_text(clip_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    relev_thre=0.40 # threshold for similarity
    coord_text_probs = []
    for clips,coords,img_nums in zip(clip_list,coord_list,img_num_list):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features=torch.from_numpy(clips[0]).unsqueeze(0).half().to(clip_device)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        if text_probs[0,3]>relev_thre:
            coord_text_probs.append((img_nums, text_probs[0,3], coords))
    coord_text_probs.sort(key=lambda x: x[1], reverse=True)
    print("total qualified:",len(coord_text_probs))
    if len(coord_text_probs)>0:
        img_num, text_prob, coord = coord_text_probs[0]
        #print("Result:",img_num, text_prob, coord)
        return coord
    else:
        return 255,255,255
    
def get_mask(image, target_rgb, tolerance):
    tarR, tarG, tarB = target_rgb
    tolR, tolG, tolB = tolerance
    B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    mask_R = (R >= tarR - tolR) & (R <= tarR + tolR)
    mask_G = (G >= tarG - tolG) & (G <= tarG + tolG)
    mask_B = (B >= tarB - tolB) & (B <= tarB + tolB)
    mask = mask_R & mask_G & mask_B
    mask = mask.astype(np.uint8) * 255
    return mask

if __name__=='__main__':
    rendered_feat_path = "./lerf_ovs/00001.png" # path to rendered feature
    query = 'bag of cookies'
    tarR, tarG, tarB = get_tar(query)
    feat = cv.imread(rendered_feat_path)
    tolerance = (5, 5, 5)
    mask = get_mask(feat, (tarR, tarG, tarB), tolerance)
    cv.imwrite(f"./query_{query}.png", mask)