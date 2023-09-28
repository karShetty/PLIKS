"""
This file contains the definition of the SMPL model
Codes are adapted from https://github.com/nkolot/GraphCMR
"""
from __future__ import division

import torch
import torch.nn as nn
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pliks.utils import *
import pliks.config as cfg


class SMPL(nn.Module):
    def __init__(self, model_file=cfg.SMPL_FILE):
        super(SMPL, self).__init__()
        with open(model_file, 'rb') as f:
            smpl_model = pickle.load(f, encoding='iso-8859-1')
        J_regressor = smpl_model['J_regressor'].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i, v, J_regressor_shape).to_dense())
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso',
                                'L_Knee', 'R_Knee', 'Spine', 'L_Ankle',
                                'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
                                'Neck', 'L_Thorax', 'R_Thorax', 'Head_top',
                                'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
                                'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.register_buffer('weights', torch.FloatTensor(smpl_model['weights']))
        self.register_buffer('posedirs', torch.FloatTensor(smpl_model['posedirs']))
        self.register_buffer('v_template', torch.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs', torch.FloatTensor(np.array(smpl_model['shapedirs'])))
        self.register_buffer('faces', torch.from_numpy(smpl_model['f'].astype(np.int64)))
        self.register_buffer('kintree_table', torch.from_numpy(smpl_model['kintree_table'].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None


    def forward(self, pose, beta, simple_model = True):
        device = pose.device
        batch_size = pose.shape[0]
        if batch_size == 0:
            return pose.new_zeros([0, 6890, 3])

        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, 10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)

        if simple_model:
            v_posed = v_shaped
        else:
            # I_cube = torch.eye(3)[None, None, :].to(device)
            I_cube = torch.eye(3)[None, None, :].to(device).type(pose.dtype)

            # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
            lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
            posedirs = self.posedirs.view(-1, 207)[None, :].expand(batch_size, -1, -1)
            v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        # pad_row = torch.FloatTensor([0,0,0,1]).to(device).view(1,1,1,4).expand(batch_size, 24, -1, -1)
        pad_row = torch.tensor([0, 0, 0, 1], dtype=pose.dtype, device=device).view(1, 1, 1, 4).expand(batch_size, 24,
                                                                                                      -1, -1)

        G_ = torch.cat([G_, pad_row], dim=2)
        # G = G_.clone()
        G = [G_[:,0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
            # G[:, i, :, :] = torch.matmul(G[:, self.parent[i - 1], :, :], G_[:, i, :, :])
        G = torch.stack(G).permute(1,0,2,3)
        self.G =G
        # rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        rest = torch.cat([J, torch.zeros(batch_size, 24, 1, device=device, dtype=pose.dtype)], dim=2).view(batch_size,
                                                                                                           24, 4, 1)
        # zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        zeros = torch.zeros(batch_size, 24, 4, 3, device=device, dtype=pose.dtype)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890, batch_size, 4,
                                                                                             4).transpose(0, 1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    # The function used in Graph-CMR, outputting the 24 training joints.
    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        joints_extra = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_extra])
        joints = torch.cat((joints, joints_extra), dim=1)
        joints = joints[:, cfg.JOINTS_IDX]
        return joints

    def get_joints_i2l(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_i2l])
        return joints

    def get_joints_i2l_hm36(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_i2l_hm36])
        return joints[:,self.h36m_idx]

    def get_joints_i2lb(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_i2l_hm36])
        return joints


    def get_joints_h36m(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_i2l_hm36])
        return joints[:,self.h36m_idx]


    def get_joints_pw3d(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_i2l])
        return joints[:,self.h36m_idx][:,h36m_eval_joint]


    # The function used in Graph-CMR, get 38 joints.
    def get_full_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        joints_extra = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_extra])
        joints = torch.cat((joints, joints_extra), dim=1)
        return joints

    # Get 14 lsp joints use the joint regressor provided by CMR.
    def get_lsp_joints(self, vertices):
        joints = torch.matmul(self.lsp_regressor_cmr[None, :], vertices)
        return joints

    # Get the joints defined by SMPL model.
    def get_smpl_joints(self, vertices):
        """
        This method is used to get the SMPL model joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        return joints.contiguous()

    # Get 24 training joints using the evaluation LSP joint regressor.
    def get_train_joints(self, vertices):
        """
        This method is used to get the training 24 joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.matmul(self.train_regressor[None, :], vertices)
        return joints

    # Get 14 lsp joints for the evaluation.
    def get_eval_joints(self, vertices):
        """
        This method is used to get the 14 eval joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 14, 3)
        """
        joints = torch.matmul(self.lsp_regressor_eval[None, :], vertices)
        return joints
