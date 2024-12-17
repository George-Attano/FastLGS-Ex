import os
import torch
from tqdm import tqdm
from os import makedirs
import torchvision
import numpy as np
import open_clip
from PIL import Image
import cv2
import time
import gradio as gr
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from scipy.ndimage import label, center_of_mass

# select the device for computation (from SAM2)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def get_color_from_id(obj_id):
    np.random.seed(obj_id)  # random mapping to RGB, might encounter duplicates
    return tuple(np.random.randint(0, 256, size=3).tolist()) 

def add_mask(ori_img,mask,obj_id):
    mask_color = get_color_from_id(obj_id)
    overlay = ori_img.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask, mask_color[c], ori_img[:, :, c])
    alpha = 0.5
    img_with_mask = cv2.addWeighted(overlay, alpha, ori_img, 1 - alpha, 0)
    return img_with_mask

def add_points(ori_img,all_points,all_labels, all_ids):
    for i,point in enumerate(all_points):
        if all_labels[i]==1:
            cv2.circle(ori_img, (point[0], point[1]), 5, (0, 255, 0), -1)
        else:
            cv2.circle(ori_img, (point[0], point[1]), 5, (255, 0, 0), -1)
        if point[0] > 100:
            cv2.putText(ori_img, str(all_ids[i]), (point[0]-40, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        else:
            cv2.putText(ori_img, str(all_ids[i]), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return ori_img

def gen_masks():
    global init_points, cur_status, chosen_points, mask_generator, sam2
    init_points = []; chosen_points = []
    cur_status = "generating initial points"
    masks = mask_generator.generate(Original_imgs[ann_frame_idx])
    #masks = mask_generator.generate(Original_imgs[21])
    masks = sorted(masks, key=lambda x: x['area'])
    img = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 3), dtype=np.uint8)
    centers = []
    for mask in masks:
        m = mask['segmentation']
        labeled_array, num_features = label(m)
        if num_features != 1:
            continue
        (cy, cx) = center_of_mass(m)
        cx = int(cx); cy = int(cy)
        centers.append([cx,cy])
        color_mask = (np.random.random(3) * 255).astype(np.uint8)
        img[m] = color_mask
    dup_points = []; init_choices = []; idx = 0
    for center in centers:
        cx = center[0]; cy = center[1]
        sig = str(img[cy,cx,0])+str(img[cy,cx,1])+str(img[cy,cx,2])
        if sig in dup_points:
            continue
        dup_points.append(sig)
        cv2.circle(img, center, radius=5, color=(255, 255, 255), thickness=-1)
        if cx > 100:
            cv2.putText(img, str(255-idx), (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(img, str(255-idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        init_points.append(center); init_choices.append(255-idx)
        idx += 1
    cur_status = "initial points generated"
    #cv2.imwrite(f"initial_mask.png", img)
    del mask_generator
    del sam2
    torch.cuda.empty_cache()
    return gr.CheckboxGroup(choices=init_choices, value=init_choices, visible=True)
    
def update_anno(choices):
    global init_points, chosen_points
    chosen_points = choices
    anno_img = Original_imgs[ann_frame_idx].copy()
    for choice in choices:
        center = init_points[255-choice]
        cv2.circle(anno_img, center, radius=5, color=(255, 255, 255), thickness=-1)
        if center[0] > 100:
            cv2.putText(anno_img, str(choice), (center[0]-40, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(anno_img, str(choice), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return anno_img

def apply_points():
    global init_points, chosen_points, points_dict, labels_dict, idx_dict, Display_imgs, id_mask_dict, ann_img_num
    for poin_idx in chosen_points:
        points = np.array([init_points[255-poin_idx]], dtype=np.float32)
        labels = np.array([1], np.int32)
        points_dict[ann_frame_idx].append(init_points[255-poin_idx])
        labels_dict[ann_frame_idx].append(1)
        idx_dict[ann_frame_idx].append(poin_idx)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=poin_idx,
        points=points,
        labels=labels,
        )
    cur_img = Original_imgs[ann_frame_idx].copy()
    for i, out_obj_id in enumerate(out_obj_ids):
        curr_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        cur_img = add_mask(cur_img, curr_mask, out_obj_id)
        id_mask_dict[out_obj_id] = curr_mask
        ann_img_num[out_obj_id] = ann_frame_idx
    # show points
    cur_img = add_points(cur_img, points_dict[ann_frame_idx], labels_dict[ann_frame_idx], idx_dict[ann_frame_idx])
    Display_imgs[ann_frame_idx] = cur_img
    return cur_img

def get_pixel(evt: gr.SelectData):
    global points_dict, labels_dict, idx_dict, Display_imgs, id_mask_dict, ann_img_num
    sel_y, sel_x = evt.index[1], evt.index[0]
    points = np.array([[int(sel_x),int(sel_y)]], dtype=np.float32)
    labels = np.array([curr_label], np.int32)
    #prompts[ann_obj_id] = points, labels
    points_dict[ann_frame_idx].append([int(sel_x),int(sel_y)])
    labels_dict[ann_frame_idx].append(curr_label)
    idx_dict[ann_frame_idx].append(ann_obj_id)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    )
    cur_img = Original_imgs[ann_frame_idx].copy()
    for i, out_obj_id in enumerate(out_obj_ids):
        curr_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        cur_img = add_mask(cur_img, curr_mask, out_obj_id)
        if out_obj_id == ann_obj_id:
            id_mask_dict[out_obj_id] = curr_mask
            ann_img_num[out_obj_id] = ann_frame_idx
    # show selected points
    cur_img = add_points(cur_img, points_dict[ann_frame_idx], labels_dict[ann_frame_idx], idx_dict[ann_frame_idx])
    Display_imgs[ann_frame_idx] = cur_img
    return cur_img

def store_mapping(ori_img, mask, obj_id):
    mask_color = get_color_from_id(obj_id)
    for c in range(3):
        ori_img[:, :, c] = np.where(mask, mask_color[c], ori_img[:, :, c])
    return ori_img

def update_info():
    global cur_status
    return cur_status

def change_pattern(choice):
    global curr_label
    if choice == "Add":
        curr_label = 1
    else:
        curr_label = 0

def change_anno_id(choice):
    global ann_obj_id
    ann_obj_id = choice
    return gr.Number(value=choice)

def change_user_anno(num):
    global ann_obj_id
    ann_obj_id = int(num)
    return gr.Radio(value=None)

def update_img_slider(img_idx):
    global ann_frame_idx
    ann_frame_idx = img_idx
    return Display_imgs[img_idx]

def propagate_in_video():
    global cur_status, Display_imgs, id_mask_dict, ann_img_num, clips, coords, img_nums, first_prop, clip_model
    if first_prop:
        first_prop = False
        cur_status = "computing embeddings"
        for id, mask in id_mask_dict.items():
            mask = np.squeeze(mask)
            rows, cols = np.where(mask)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            cropped_img = Original_imgs[ann_img_num[id]][min_row:max_row, min_col:max_col]
            pil_img = Image.fromarray(cropped_img)
            pil_img.save(f"./data/crop/{id}.png") #debug use
            block = preprocess(pil_img).unsqueeze(0).to(clip_device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                clip_feats = clip_model.encode_image(block)
                clip_feats /= clip_feats.norm(dim=-1, keepdim=True)
            clips.append(clip_feats.cpu().numpy())
            coords.append(get_color_from_id(id))
            img_nums.append(ann_img_num[id])
        np.save(out_mapping_dir, zip(clips, coords, img_nums)) # img_nums is for debug use
        clips, coords, img_nums = [], [], []
        del clip_model
        torch.cuda.empty_cache()
    cur_status = "start propagating"
    video_segments = {}  # video_segments contains the per-frame segmentation results
    predictor.propagate_in_video_preflight
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        cur_status = f"forward propagating at frame {out_frame_idx}"
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        cur_status = f"backward propagating at frame {out_frame_idx}"
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    cur_status = "creating preview images"
    for out_frame_idx in range(len(frame_names)):
        out_ori_img = Original_imgs[out_frame_idx].copy()
        #img_name = f"IMG_{2286+out_frame_idx}.png"
        img_name = 'frame_'+'{0:05d}'.format(out_frame_idx+1) + ".png"
        img_path = os.path.join(out_gt_dir, img_name)
        '''
        if os.path.exists(img_path):
            map_img = cv2.imread(img_path)
            map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        else:
            map_img = np.zeros((out_ori_img.shape[0], out_ori_img.shape[1], 3), dtype=np.uint8)
        '''
        map_img = np.zeros((out_ori_img.shape[0], out_ori_img.shape[1], 3), dtype=np.uint8)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_ori_img = add_mask(out_ori_img, out_mask, out_obj_id)
            map_img = store_mapping(map_img, out_mask, out_obj_id)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, map_img)
        Display_imgs[out_frame_idx] = out_ori_img
    cur_status = "prop done"
    return Display_imgs[ann_frame_idx]


if __name__ == "__main__":
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt" # path to sam2 checkpoint
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, stability_score_thresh=0.90)

    out_gt_dir = "./data/gt_images" # path to output gt feats
    out_mapping_dir = "./data/TEATIME_CLIP_FEAT.npy" # path to save mapped feats
    if not os.path.exists(out_gt_dir):
        os.makedirs(out_gt_dir)
    video_dir = "./notebooks/videos/teatime" # path to SAM2 standard frames
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    Original_imgs = []; Display_imgs = []
    for frame_name in frame_names:
        img_path = os.path.join(video_dir, frame_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Original_imgs.append(img)
        Display_imgs.append(img.copy())

    init_points = []; chosen_points = []
    #gen_masks()

    ann_frame_idx = 0
    ann_obj_id = 1  #ANNO ID
    points_dict = {}; labels_dict = {}; idx_dict = {}; id_mask_dict = {}; ann_img_num = {}
    clips = []; coords = []; img_nums = []; first_prop = True #NPY DATA
    for i in range(len(Original_imgs)):
        points_dict[i] = []
        labels_dict[i] = []
        idx_dict[i] = []
    curr_label = 1
    inference_state = predictor.init_state(video_path=video_dir,async_loading_frames=True,offload_video_to_cpu=True)

    print("Loading CLIP...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        precision="fp16"
    )
    clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(clip_device)

    cur_status = "Awaiting Orders..."
    with gr.Blocks() as demo:
        gr.Markdown("# <center>Data Initializer Interface</center>")
        with gr.Row():
            with gr.Column():
                chat_title = gr.Markdown("## <center>Config Area</center>")
                with gr.Row():
                    anno_id_radio = gr.Radio(choices=[1, 2, 3, 4, 5], value=1, label="Select Annotation ID")
                    anno_id_input = gr.Number(value=1, label="Input Annotation ID")
                pattern_radio = gr.Radio(choices=["Add", "Del"], value="Add", label="Select Pattern")
                Img_Slider = gr.Slider(minimum=0,maximum=len(frame_names)-1,step=1,label="Image Selector",interactive=True)
                with gr.Row():
                    init_btn = gr.Button(value="Initialize")
                    init_apply_btn = gr.Button(value="Apply Initialization")
                init_check = gr.CheckboxGroup(choices=[], value=[], visible=False)
                propagate_btn = gr.Button(value="Full Propagate")
                status_info = gr.Textbox(label="Status Info",value=update_info,every=1.0,interactive=False)

            with gr.Column():
                monitor_title = gr.Markdown("## <center>Monitor</center>")
                img_window = gr.Image(value=Original_imgs[0],interactive=False)
                img_window.select(fn=get_pixel,outputs=img_window)

            anno_id_radio.input(fn=change_anno_id,inputs=anno_id_radio,outputs=anno_id_input) 
            anno_id_input.input(fn=change_user_anno,inputs=anno_id_input,outputs=anno_id_radio)
            pattern_radio.change(fn=change_pattern,inputs=pattern_radio)
            Img_Slider.change(fn=update_img_slider,inputs=Img_Slider,outputs=img_window,show_progress='hidden')
            init_btn.click(fn=gen_masks,outputs=init_check)
            init_check.change(fn=update_anno,inputs=[init_check],outputs=img_window)
            init_apply_btn.click(fn=apply_points,outputs=img_window)
            propagate_btn.click(fn=propagate_in_video,outputs=[img_window])
        
    gr.close_all()
    demo.launch(share=False)