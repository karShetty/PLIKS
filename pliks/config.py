"""
Parts of the codes are adapt from https://github.com/nkolot/GraphCMR

This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""

PW3D_ROOT = "/cluster/dataset/3dpw"
HM36_ROOT = "/cluster/dataset/human3.6"
MUCO_ROOT = "/cluster/dataset/muco"



SMPL_FILE = 'model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
MALE_SMPL_FILE = 'model_files/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
FEMALE_SMPL_FILE = 'model_files/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

JOINT_REGRESSOR_H36M = 'model_files/J_regressor_h36m.npy'

# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

INPUT_RES = 224

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

