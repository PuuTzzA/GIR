<div align="center">

# [T-PAMI🔥] Gir: 3d gaussian inverse rendering for relightable scene factorization  

[![Paper](https://img.shields.io/badge/Paper-<Arxiv>-<COLOR>.svg)](https://arxiv.org/abs/2312.05133)
[![Project Page](https://img.shields.io/badge/Project_Page-<Website>-blue.svg)](https://3dgir.github.io/)
    
[Yahao Shi](https://scholar.google.com/citations?user=-VJZrUkAAAAJ&hl=en)<sup>1</sup>
[Yanmin Wu](https://yanmin-wu.github.io/)<sup>2</sup>
[Chenming Wu](https://chenming-wu.github.io/)<sup>3</sup>
[Xing Liu](https://scholar.google.com/citations?user=bdVU63IAAAAJ&hl=en)<sup>3</sup>
[Chen Zhao](https://scholar.google.com/citations?hl=en&user=kWzyOa8AAAAJ)<sup>3</sup>
[Haocheng Feng](https://scholar.google.com.hk/citations?user=pnuQ5UsAAAAJ&hl=zh-CN&oi=ao)<sup>3</sup>
[Jian Zhang](https://jianzhang.tech/)<sup>2</sup> 
<br>
[Bin Zhou](http://scholar.google.com/citations?user=tG4RnyYAAAAJ&hl=en&oi=ao)<sup>1</sup>
[Errui Ding](https://scholar.google.com/citations?user=1wzEtxcAAAAJ&hl=zh-CN)<sup>3</sup>
[Jingdong Wang](https://jingdongwang2017.github.io/)<sup>3</sup>
    
<sup>1</sup> Beihang University, <sup>2</sup> Peking University, <sup>3</sup> Baidu VIS
    
</div>
    
---
    
[Prerelease] Official implementation of "GIR: 3D Gaussian Inverse Rendering for Relightable Scene Factorization".
    
## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>
    
---
    
    

## 0. Installation

The installation of GIR is similar to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
```
# Clone the Repository
git clone https://github.com/guduxiaolang/GIR.git

# Create the environment
conda create -n gir python=3.7
conda activate gir
 
# Install the dependencies
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/envlight
pip install tqdm plyfile 
    
# Load HDR images correctly 
pip install imageio[full]
```
---
    
## 1. Data preparation
The files are as follows:

Blender Dataset
```
[DATA_ROOT]
|---test
|   |---<image 0>
|   |---<image 1>
|   |---...
|---train
|   |---<image 0>
|   |---<image 1>
|   |---...
|---transforms_test.json
|---transforms_train.json
```  
COLMAP Dataset 
```
[DATA_ROOT] 
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

---

## 2. Training and Evaluation
The training and evaluation commands for each dataset are provided in the shell scripts located in the `scripts` folder.
    
The basic training and testing commands are shown below.
    
```
# training
python train.py -s $data_dir --eval --port $port_num --random_background --hdr_rotation
    
# rendering
python render.py -m $model_dir --skip_train --save_name "render" -w --hdr_rotation
    
# relighting
python render.py -m $model_dir --skip_train --save_name ${hdr_list_name%.*} -w --hdr_rotation --environment_texture $hdr_dir --render_relight
```

### Evaluation Loop

During training (`train.py`), the model periodically runs an evaluation loop (every `--eval_interval` iterations, defaulting to 2000, and at the end of training) to monitor optimization quality.

1. **Standard Metrics Evaluation**:
   - Computes standard reconstruction quality metrics: **PSNR**, **SSIM**, **LPIPS**, **L1**, and **MSE** on the `test` cameras and a sample of `train` cameras.
   - If material/normal ground truths (data priors) are available, it computes additional priors metrics: **Albedo PSNR**, **Albedo SSIM**, **Albedo L1**, and **Normal Angular Error**.
   - Periodically saves side-by-side comparison grids (renders next to ground truths) for standard views to `<model_path>/eval_visuals/{test|train}/`.

2. **Relighting Performance Evaluation**:
   - **Alternate-Step Evaluation**: Executed on every second evaluation step (starting from the second evaluation iteration).
   - Under custom environment maps, it temporarily swaps out the environment light (`gaussians.envlight`) using the `.hdr` files in the dataset's `hdris` directory, without performing disk/CLI script calls.
   - Renders the scene under the target environment maps and compares it with the corresponding ground truth relighted views (from the `rgba_{hdri_name}` folders).
   - Computes **PSNR** and **SSIM** for each target HDRI.
   - Saves individual render and ground truth images under `<model_path>/eval_visuals/relight_{hdri_name}/` using original view names for better traceability.
   - **Target HDRIs**: Customizable using `--eval_relight_hdris` (defaults to `snowy_forest`, `moonless_night`, and `fireplace`). If these are not available in the dataset, the loop automatically falls back to other available HDRIs in the dataset.

3. **PDF Report Generation**:
   - Once training completes, a PDF report (`training_report.pdf`) is automatically generated. It includes:
     - Metrics tables for the final iteration.
     - Plots of the evolution of rendering performance (PSNR/SSIM/LPIPS) and loss component curves.
     - Visual comparison pages containing side-by-side renderings and ground truths for both standard and relighted views.

---

### Outputs Folder Structure

A training run saves its results to the model path directory (configured via `-m` / `--model_path` or auto-generated under `./output/<uuid>/`). The structure of the output folder is as follows:

```
[model_path]
|--- cfg_args                       # Config arguments namespace string
|--- chkpnt<iteration>.pth          # PyTorch checkpoint files saved at checkpoint iterations
|--- metrics_log.json               # JSON log of evaluation metrics per iteration
|--- training_report.pdf            # PDF training report summarizing metrics, plots, and visuals
|--- point_cloud/
|    |--- iteration_<iteration>/
|         |--- point_cloud.ply      # Exported 3D Gaussian Splatting point cloud file
|--- train_process/
|    |--- loss_components.json      # JSON log of loss component values per iteration
|    |--- renders/                  # Rendered training frames
|    |--- gt/                       # Ground truth training frames
|    |--- normal/                   # Predicted normal maps (if priors/stage active)
|    |--- albedo/                   # Predicted albedo maps (if priors/stage active)
|    |--- metallic/                 # Predicted metallic maps (if priors/stage active)
|    |--- roughness/                # Predicted roughness maps (if priors/stage active)
|    |--- ...                       # Other decomposed material / depth maps
|--- eval_visuals/
|    |--- test/                     # Comparison grids of standard renders vs test ground truth
|    |--- train/                    # Comparison grids of standard renders vs train ground truth
|    |--- relight_<hdri_name>/      # Separate renders and ground truths under custom HDRIs:
|         |--- iter<iteration>_<view_name>_render.png
|         |--- iter<iteration>_<view_name>_gt.png
```
    

---

## 3. Acknowledgements
We are quite grateful for [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [NeRO](https://github.com/liuyuan-pal/NeRO), and [Filament](https://google.github.io/filament/Filament.html)

---