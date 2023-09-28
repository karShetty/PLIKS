import numpy as np
import torch
import torch.nn.functional as F
import vtk, os
from vtk.util import numpy_support

def rodrigues(theta):
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)

def quat2mat(quat):
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def rodrigues_smpl(r):
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R


#####From pytorch 3D#####
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # clipping is not important here; if q_abs is small, the candidate won't be picked
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clip(0.1))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
           F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
           ].reshape(*batch_dim, 4)


def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix):
    matrix_ = matrix
    if len(matrix.shape) != 3:
        matrix_ = matrix.unsqueeze(0)
        return quaternion_to_axis_angle(matrix_to_quaternion(matrix_))[0]
    else:
        return quaternion_to_axis_angle(matrix_to_quaternion(matrix_))[0]


def vtk_write(path, polydata, type='ply'):
    if type == 'ply':
        polydata_writer = vtk.vtkPLYWriter()
    else:
        polydata_writer = vtk.vtkOBJWriter()
    polydata_writer.SetFileName(path)
    polydata_writer.SetInputData(polydata)
    polydata_writer.Write()


def create_polydata(v_, f_, n_=None):
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(v_, deep=True))
    triangles = vtk.vtkCellArray()
    t_ = np.hstack((np.full((f_.shape[0], 1), 3), f_))
    triangles.SetCells(t_.shape[0], numpy_support.numpy_to_vtkIdTypeArray(t_.ravel(), deep=True))
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetPolys(triangles)
    if not n_ is None:
        pointNormalsArray = vtk.vtkDoubleArray()
        pointNormalsArray.SetNumberOfComponents(3)
        pointNormalsArray.SetNumberOfTuples(n_.shape[0])
        for i, n in enumerate(n_):
            pointNormalsArray.SetTuple(i, n)
        polyData.GetPointData().SetNormals(pointNormalsArray)
    return polyData


def overlay_image(background,
                   rgb_img,
                   seg_img,
                   opacity_factor=0.95):
    background_px_loc = seg_img[:, None, :, :] == 0
    output = background * background_px_loc + opacity_factor * rgb_img * (
        torch.logical_not(background_px_loc)) + (1 - opacity_factor) * (
                              torch.logical_not(background_px_loc)) * background
    return output
