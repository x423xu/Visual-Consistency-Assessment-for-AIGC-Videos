# Visual-Consistency-Assessment-for-AIGC-Videos
This is a repo for assessing the visual consistency of aigc videos.

# Where to start?
In this project, we want to address one typical problem in AIGC video, i.e. temporal consistency.
The idea is to firstly estimate the optical flow of the current frame and then transform the pixels to the next frame. Then, we want to find out how much do the transformed frame match the target frame.

Okay, now let's start from the simplest idea: pixel space transformation.
1. Take current frame as input,
2. Estimate flow
3. Transform
4. Find out how much do they match
    1. some pixels are totally not matched, just remove them
    2. some pixels are weakly matched, we need to quantify the matchness.
        - Semantic matchness: for a matched region, check if they are the same semantics (using a DINO/CLIP model), or the disimilarity
        - Low-level matchness: using a VGG/resnet to extract shallow-layer activations as low-level features. Check their transformablness.

Good! now let's manage the project module by module:
1. Optical Flow Estimation.
2. High level matchness.
3. Low-level matchness.  

# Optical flow
![flow model](https://cdn-uploads.huggingface.co/production/uploads/661f4653702ad39754d94ac0/S47EG-3TZRYgpzcjwfvzF.png)

## Flowformer
## RAFT
## Memflow

# How to start
1. Create an environment. *Comment out the packages that have conflicted versions from the environment.yml*

`conda env create -f environment.yml`

2. Activate your environment

`conda activate tcvqa`

3. Train the baseline model

`python main.py --batch_size=8 --num_epochs=200 --lr=1e-4 --weight_decay=1e-3`

# TODO
- [ ] try different backbones, swin3D/vivit
- [ ] add text-visual alignment module
- [ ] add naturalness assessment module



