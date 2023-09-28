from pliks.model.hrnet import get_pose_net
from pliks.smpl import SMPL
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.transforms.face_to_edge import FaceToEdge
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class MeshNet(nn.Module):
    def __init__(self, vertex_num, **kwargs):
        super(MeshNet, self).__init__()
        smpl = SMPL()
        v = smpl.v_template.cpu().data.numpy()
        f = smpl.faces.cpu().data.numpy()
        self.edge = FaceToEdge()(Data(face=torch.LongTensor(f.swapaxes(0, 1)), num_nodes=v.shape[0])).edge_index.cuda()

        self.vertex_num = vertex_num
        self.conv1 = GCNConv(168, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 3)

        self.conv_x = make_conv1d_layers([720, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([720, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([720, 56 * 224], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_2 = make_conv1d_layers([224, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_w_1 = make_conv1d_layers([720, 56 * 224], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_w_2 = make_conv1d_layers([224, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        img_feat_xy = img_feat
        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)

        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)

        # z axis
        img_feat_z = img_feat.mean((2, 3))[:, :, None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1, 224, 56)
        heatmap_z = self.conv_z_2(img_feat_z)

        # w axis
        img_feat_w = img_feat.mean((2, 3))[:, :, None]
        img_feat_w = self.conv_w_1(img_feat_w)
        img_feat_w = img_feat_w.view(-1, 224, 56)
        heatmap_w = self.conv_w_2(img_feat_w)
        coord_w = torch.sigmoid(heatmap_w.mean(-1, keepdim=True))

        xyz = torch.cat([heatmap_x, heatmap_y, heatmap_z], -1)
        xyz = F.leaky_relu(self.conv1(xyz, self.edge, ))
        xyz = F.leaky_relu(self.conv2(xyz, self.edge, ))
        xyz = self.conv3(xyz, self.edge, )

        mesh_coord = torch.cat((xyz, coord_w), 2)
        return mesh_coord


class Mesh_Regressor(nn.Module):
    def __init__(self,
                 **kwargs):
        super(Mesh_Regressor, self).__init__()
        self.backbone = get_pose_net(**kwargs)
        self.meshnet = MeshNet(6890, **kwargs)
        self.meshnet.apply(init_weights)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(720, 512)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 512)
        self.dec_beta = nn.Linear(512, 10)
        self.dec_trans_xy = nn.Linear(512, 2)
        self.dec_depth = nn.Linear(512, 1)

    def forward(self, img):
        x, _ = self.backbone(img)
        z = self.avg_pool(x).squeeze(-1).squeeze(-1)
        z = self.drop1(z)
        z = self.fc1(z)
        z = self.drop2(z)
        z = self.fc2(z)
        pred_beta = self.dec_beta(z)
        pred_depth = self.dec_depth(z)
        pred_trans = self.dec_trans_xy(z)
        pres_smpl_uvdw = self.meshnet(x)
        return pres_smpl_uvdw, pred_beta, pred_depth, pred_trans
