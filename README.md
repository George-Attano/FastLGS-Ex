# FastLGS-Ex
This is simplified implementation of extending FastLGS with [SAM2](https://github.com/facebookresearch/sam2).

SAM2 shows astonishing consistency and efficiency in video segmentation, thus we use it to replace the original matching part.

The original matching preprocess is time-consuming that costs more than 25 minutes, but with SAM2 it can be done within 2 minutes.

We further provide a GUI for manual refinement.

MIoU Results on the [LERF_OVS](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view) dataset are as follows:
