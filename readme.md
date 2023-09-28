# PLIKS: A Pseudo-Linear Inverse Kinematic Solver for 3D Human Body Estimation

> [PLIKS: A Pseudo-Linear Inverse Kinematic Solver for 3D Human Body Estimation](https://arxiv.org/abs/2211.11734)    
> Karthik Shetty, Annette Birkhold, Srikrishna Jaganathan, Norbert Strobel, Markus Kowarschik, Andreas Maier, Bernhard Egger  
> CVPR 2023  


## Installation  
```
conda create -n pliks python=3.8
conda activate pliks

#Install PyTorch
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

#Install PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0

#Install other dependencies for visualization
pip install vtk==9.1.0 vedo=2021.0.5 opencv=3.4.2

#Install torch_geometric to run the model from the paper
pip install torch-geometric==1.7.2 torch-scatter==2.0.9 torch-sparse==0.6.12
```
For visualization `pytorch3d==0.4.0` is required. The sparse model does not require `torch_geometric`.  

## Data
### SMPL Files and Pretrained Model  
- Download smpl files from the official [website](https://smpl.is.tue.mpg.de/). Unzip and place `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in `model_files/`. 
- Download the pretrained model [sparse](https://drive.google.com/file/d/1sVt71jxEF16VO2-247-9prbBW8B3iR2O/view?usp=sharing) or [full](https://drive.google.com/file/d/1FZbH6HccY78zfKvwfoyOhvaGAsN28Opd/view?usp=sharing).


## Quick Demo
```
python demo.py --img_dir demo/input/* --out_dir demo/output/ --checkpoint checkpoint_sparse.pt --model MeshRegSparse
```

## Citation  
```
@InProceedings{Shetty_2023_CVPR,
    author    = {Shetty, Karthik and Birkhold, Annette and Jaganathan, Srikrishna and Strobel, Norbert and Kowarschik, Markus and Maier, Andreas and Egger, Bernhard},
    title     = {PLIKS: A Pseudo-Linear Inverse Kinematic Solver for 3D Human Body Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {574-584}
}
```

## Acknowledgement  
Code is adapted from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE), [SPIN](https://github.com/nkolot/SPIN), [DecoMR](https://github.com/zengwang430521/DecoMR), [PARE](https://github.com/mkocabas/PARE/tree/master/pare).