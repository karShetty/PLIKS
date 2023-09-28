from pliks.model.hrnet import get_pose_net
import torch
import torch.nn as nn
from torch.nn import functional as F


class Mesh_Regressor_Sparse(nn.Module):
    def __init__(self, sparse_len=934, img_res=224,
                 **kwargs):
        super(Mesh_Regressor_Sparse, self).__init__()
        self.backbone = get_pose_net(**kwargs)

        self.final_layer = nn.Conv2d(
            in_channels=720,
            out_channels=sparse_len,
            kernel_size=1,
            stride=1,
            padding=1 if 1 == 3 else 0
        )

        self.fix_scale = img_res * 2
        self.encoder_scale = img_res // 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fcx = nn.Linear(self.encoder_scale * self.encoder_scale, self.fix_scale)

        self.fcy = nn.Linear(self.encoder_scale * self.encoder_scale, self.fix_scale)
        self.fcz = nn.Linear(self.encoder_scale * self.encoder_scale, self.fix_scale)
        self.fcw_ = nn.Linear(sparse_len, sparse_len)
        self.fcx_ = nn.Linear(self.fix_scale, 1)
        self.fcy_ = nn.Linear(self.fix_scale, 1)
        self.fcz_ = nn.Linear(self.fix_scale, 1)

        self.dec_beta = nn.Linear(sparse_len, 10)
        self.dec_trans_xy = nn.Linear(sparse_len, 2)
        self.dec_depth = nn.Linear(sparse_len, 1)

    def forward(self, img):
        x, _  = self.backbone(img)
        x = self.final_layer(x)
        x_ = x.reshape([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])
        z = self.avg_pool(x).squeeze(-1).squeeze(-1)
        fc_xscale = self.fcx(x_)
        fc_yscale = self.fcy(x_)
        fc_zscale = self.fcz(x_)
        hw = F.sigmoid(self.fcw_(z))
        pred_smpl_uvd = torch.cat([self.fcx_(F.leaky_relu(fc_xscale)),
                                   self.fcy_(F.leaky_relu(fc_yscale)),
                                   self.fcz_(F.leaky_relu(fc_zscale)),
                                   hw[..., None]], -1)
        pred_beta = self.dec_beta(z)
        pred_depth = self.dec_depth(z)
        pred_trans = self.dec_trans_xy(z)
        return pred_smpl_uvd, pred_beta, pred_depth, pred_trans
