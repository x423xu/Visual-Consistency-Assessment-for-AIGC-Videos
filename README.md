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
