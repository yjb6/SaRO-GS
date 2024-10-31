render:  python test.py --configpath configs/n3d_lite/flame_s.json --model_path log/flame_steak/notschedule_scalereg8e-6_prealxyzf5k_d5k_nosintf5k_hexdim32/ 

# SaRO-GS:4D Gaussian Splatting with Scale-aware Residual Field and Adaptive Optimization for Real-time Rendering of Temporally Complex Dynamic Scenes
Jinbo Yan, Rui Peng, Luyang Tang, Ronggang Wang<br>
| [Webpage](https://yjb6.github.io/SaRO-GS.github.io/)]

This repository contains the official authors implementation associated with the paper "4D Gaussian Splatting with Scale-aware Residual Field and Adaptive Optimization for Real-time Rendering of Temporally Complex Dynamic Scenes"

The patent is pending and the code will be released around December 2024. Please stay tuned.

## Installation
Coming Soon
## Data Preparation
To prepare the data for this project, please follow the data preparation method outlined in [Spacetime-GS](https://github.com/oppo-us-research/SpacetimeGaussians?tab=readme-ov-file#processing-datasets). This guide provides detailed instructions on how to format and structure your dataset to be compatible with our model.

## Testing
Testing Code
To test the model, use the following command:

    python test.py --configpath configs/n3d_lite/flame_s.json --model_path flame_steak/ --checkpoint flame_steak/ckpt_best.ply --require_segment
### Pretrained Weights
We will be releasing pretrained weights for the model in stages. Please check back regularly for updates.

## Training
Due to pending patents, the training code and related materials will be released as soon as possible. We appreciate your patience and understanding.