# SaRO-GS:4D Gaussian Splatting with Scale-aware Residual Field and Adaptive Optimization for Real-time Rendering of Temporally Complex Dynamic Scenes

>Jinbo Yan, Rui Peng, Luyang Tang, Ronggang Wang<br>
>[Arxiv](https://arxiv.org/pdf/2412.06299)|[Webpage](https://yjb6.github.io/SaRO-GS.github.io/)|[Weights](https://drive.google.com/drive/folders/1d-gjkWyYEMzUtTHGVITMhuH7jN6TFlwD?usp=drive_link)<br>
> *ACM MM 2024 __Best__ __Paper__ __Candidate__* 

This repository contains the official authors implementation associated with the paper __4D Gaussian Splatting with Scale-aware Residual Field and Adaptive Optimization for Real-time Rendering of Temporally Complex Dynamic Scenes__

## Bibtex
```
@inproceedings{yan20244d,
      title={4D Gaussian Splatting with Scale-aware Residual Field and Adaptive Optimization for Real-time rendering of temporally complex dynamic scenes},
      author={Yan, Jinbo and Peng, Rui and Tang, Luyang and Wang, Ronggang},
      booktitle={ACM Multimedia 2024}
    }
```

## Installation
- Python >= 3.9
- Install `PyTorch >= 2.2.0`. We have tested on `torch2.2.0+cu118`, but other versions should also work fine.
```sh
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```
- Install gaussian_rasterization. We use gaussian_rasterization_ch3 from SpacetimeGS to render depth map.
```sh
pip install submodels/gaussian_rasterization_ch3
```

- Install nvdiffrast based on the [documentation](https://nvlabs.github.io/nvdiffrast/#linux)
## Data Preparation
### Neural3D Dataset
To prepare the data for this project, please follow the data preparation method outlined in [Spacetime-GS](https://github.com/oppo-us-research/SpacetimeGaussians?tab=readme-ov-file#processing-datasets). This guide provides detailed instructions on how to format and structure your dataset to be compatible with our model. 
```
<location>
|---cook_spinach
|   |---colmap_<0>
|   |---colmap_<...>
|   |---colmap_<299>
|---flame_salmon1
```
### Monocluar data
 You can download the datasets from [drive](https://drive.google.com/file/d/19Na95wk0uikquivC7uKWVqllmTx-mBHt/view?usp=sharing) or [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0). Unzip the downloaded data to the project root dir in order to train. See the following directory structure for an example:
```
<location>
│   |--- mutant
│   |--- standup 
│   |---...
```

## Testing
To test the model, use the following command:

    python test.py  -m <path to trained model>   --require_segment 
- use the --require_segment flag to get the dynamic-static segmentation results
- use the --skip_test flag to skip the test view rendering
- use the --skip_val flag to skip the free view rendering

### Pretrained Weights
You can download the pre-trained weights of Neural3D dataset from [here](https://drive.google.com/drive/folders/1d-gjkWyYEMzUtTHGVITMhuH7jN6TFlwD?usp=drive_link), and then place them under the project directory.
```
<logs>
│   |--- flame_steak
│   |--- cut_roasted_beef 
│   |---...
```
When utilizing pre-trained weights, ensure that the source_path in the cfg file is updated to the actual path.

## Training

To train the model by your own, use the following command:

    python train.py -s <path to COLMAP or NeRF Synthetic dataset> --config <corresponding config file> --exp_name <the name of this training>

- use the --no_wandb flag to disable wandb.

