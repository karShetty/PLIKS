import torch
import torch.nn as nn
from pliks.utils import rodrigues_smpl, matrix_to_axis_angle


def solve_lstq(uv_curr, xm_, b_s_, blend_curr, W, batch_size: int, this_device: torch.device, m00, m11, m02, m12,
               max_iterations: int, iteration_count: int, key_size: int, A_size: int, valid_R_list_stack,
               rot_reg_idx, bet_reg_idx, trans_reg_idx, t_pred_dl, pred_beta, lambda_factor=1, is_train=False):
    A = torch.zeros((batch_size, A_size * 2 + 10 + key_size * 3 + 3, key_size * (3 + 3) + 10), dtype=torch.float32, device=this_device)
    B = torch.zeros((batch_size, A_size * 2 + 10 + key_size * 3 + 3,), dtype=torch.float32, device=this_device)
    x1 = (m02[..., None] - uv_curr[..., 0])[:, None] * (xm_[..., 1])
    x2 = (m00[..., None])[:, None] * (xm_[..., 2]) + (m02[..., None] - uv_curr[..., 0])[:, None] * (-xm_[..., 0])
    x3 = (m00[..., None])[:, None] * (-xm_[..., 1])
    x4a = (m00[..., None])[:, None, ..., None] * b_s_[:, :, :, 0] + (m02[..., None] - uv_curr[..., 0])[:, None, ...,
                                                                    None] * b_s_[:, :, :, 2]
    x8 = blend_curr.T[None,] * (m00[..., None])[:, None]
    x9 = torch.zeros(batch_size, blend_curr.shape[1], blend_curr.shape[0], dtype=blend_curr.dtype, device=this_device)
    x10 = blend_curr.T[None,] * (m02[..., None] - uv_curr[..., 0])[:, None]
    y1 = (m11[..., None])[:, None] * (-xm_[..., 2]) + (m12[..., None] - uv_curr[..., 1])[:, None] * (xm_[..., 1])
    y2 = (m12[..., None] - uv_curr[..., 1])[:, None] * (-xm_[..., 0])
    y3 = (m11[..., None])[:, None] * (xm_[..., 0])
    y4a = (m11[..., None])[:, None, ..., None] * b_s_[:, :, :, 1] + (m12[..., None] - uv_curr[..., 1])[:, None, ...,
                                                                    None] * b_s_[:, :, :, 2]
    y8 = torch.zeros(batch_size, blend_curr.shape[1], blend_curr.shape[0], dtype=blend_curr.dtype, device=this_device)
    y9 = blend_curr.T[None,] * (m11[..., None])[:, None]
    y10 = blend_curr.T[None,] * (m12[..., None] - uv_curr[..., 1])[:, None]

    b1 = -(m00[..., None])[:, None] * (xm_[..., 0]) - (m02[..., None] - uv_curr[..., 0])[:, None] * (xm_[..., 2])
    b2 = -(m11[..., None])[:, None] * (xm_[..., 1]) - (m12[..., None] - uv_curr[..., 1])[:, None] * (xm_[..., 2])

    n = xm_.shape[2]
    A1 = torch.cat([x1[..., None], x2[..., None], x3[..., None], x8[..., None], x9[..., None], x10[..., None], ],
                   -1).permute(0, 2, 1, 3).reshape(batch_size, n, -1)
    A2 = torch.cat([y1[..., None], y2[..., None], y3[..., None], y8[..., None], y9[..., None], y10[..., None], ],
                   -1).permute(0, 2, 1, 3).reshape(batch_size, n, -1)
    A1_ = torch.cat((W[..., None] * A1, W[..., None] * A2), 1)
    A2_ = torch.cat((W[..., None] * x4a.sum(1), W[..., None] * y4a.sum(1)), 1)
    b_ = torch.cat((W * (b1.sum(1)), W * (b2.sum(1))), 1)

    A[:, :2 * n, :-10] = A1_
    A[:, :2 * n, -10:] = A2_
    B[:, :2 * n, ] = b_

    k0 = torch.ones([1, ], dtype=torch.float32).to(this_device)
    lam = torch.log(k0 / 3)
    lam = k0 * torch.exp(lam * (lambda_factor))

    # A[:, rot_reg_idx[0], rot_reg_idx[1]] = 1e0 * lam #rotational
    A[:, bet_reg_idx[0], bet_reg_idx[1]] = 1e0 * lam #shape
    # A[:, trans_reg_idx[0], trans_reg_idx[1]] = 1e0 #translation

    B[:, bet_reg_idx[0]] = 1e0 * lam * pred_beta
    # B[:,trans_reg_idx[0]] = t_pred_dl

    if is_train:
        q, r = torch.linalg.qr(A)
        b_solve = (q.permute(0, 2, 1) @ B[..., None])
        solution = torch.triangular_solve(b_solve, r, upper=True)[0][..., 0]
    else:
        solution = torch.linalg.lstsq(A, B[..., None])[0][..., 0]

    r_pred = rodrigues_smpl(
        torch.stack([solution[:, x * 6:(x) * 6 + 3] for x in range(key_size)]).permute(1, 0, 2).reshape(-1,
                                                                                                        3).unsqueeze(
            1)).reshape(batch_size, key_size, 3, 3)
    t_pred = torch.stack([solution[:, x * 6 + 3:(x) * 6 + 3 + 3] for x in range(key_size)]).permute(1, 0, 2)
    rt_pred = torch.eye(4).repeat(batch_size, key_size, 1, 1).to(solution.device)
    rt_pred[:, :, :3, :3] = r_pred
    r_curr = valid_R_list_stack
    r_curr = rt_pred @ r_curr
    beta = solution[:, -10:]
    return r_curr, t_pred, beta, r_pred


class PLIKS(nn.Module):
    def __init__(self, smpl, parent, kin_list, x_mean, b_std, blend_w, sparse=True, lambda_factor=1,
                 use_mean_shape=False, run_iters=2, run_pliks = True,
                 **kwargs):
        super(PLIKS, self).__init__()
        self.smpl = smpl
        self.parent = parent
        self.kin_list = kin_list
        self.x_mean = x_mean
        self.b_std = b_std
        self.blend_w = blend_w
        self.sparse = sparse
        self.lambda_factor = lambda_factor
        self.use_mean_shape = use_mean_shape
        self.run_iters = run_iters
        self.run_pliks = run_pliks

    def forward(self, pred_uvdw, pred_beta, pred_depth, pred_trans, K_fix, K_img,):
        this_device = pred_uvdw.device
        batch_size = pred_uvdw.shape[0]

        K = K_fix  # 4x4
        K1 = K[:, [0, 1, 3], ][..., [0, 1, 2]].clone()
        if self.run_pliks:
            K_out = K_img
        else:
            K_out = K_fix

        parent = self.parent.copy()
        kin_list = self.kin_list.copy()
        x_mean = self.x_mean.clone().detach()
        b_std = self.b_std.clone().detach()
        blend_w = self.blend_w.clone().detach()

        if self.sparse:
            x_mean_std = torch.matmul(b_std.view(-1, 10)[None, :].expand(pred_beta.shape[0], -1, -1),
                                      pred_beta[..., None]).view(-1, 934, 3) + x_mean[None,]
        else:
            x_mean_std = torch.matmul(b_std.view(-1, 10)[None, :].expand(pred_beta.shape[0], -1, -1),
                                      pred_beta[..., None]).view(-1, 6890, 3) + x_mean[None,]

        valid_R_list = []
        A_size = 0
        key_size = 0
        # ARE; to get intial rotation
        for key in kin_list.keys():
            if kin_list[key] is not None:
                indices = kin_list[key]
                uv_curr = pred_uvdw[:, indices, :2]
                depth_curr = pred_uvdw[:, indices, 2] + pred_depth[:, None]
                if self.sparse:  # Hard coded change
                    W_curr = pred_uvdw[:, indices, 3] * 0.95 + 0.05
                else:
                    W_curr = torch.clamp(pred_uvdw[:, indices, 3], 0.02, 1.)
                x_m_curr = x_mean[indices]
                x_m_std_curr = x_mean_std[:, indices]
                b_std_curr = b_std[indices]
                blend_curr_a = blend_w[indices, :]

                uvd_curr = (torch.inverse(K1) @ torch.cat(
                    [uv_curr,
                     torch.ones((uv_curr.shape[0], uv_curr.shape[1], 1), device=this_device, dtype=torch.float32)],
                    -1).permute(0, 2, 1)).permute(0, 2, 1)
                uvd_curr = (uvd_curr / uvd_curr[..., -1, None])[..., :3] * depth_curr[..., None]

                uv_c_torch = uvd_curr - uvd_curr.mean(1, keepdim=True)
                xm_c_torch = x_m_std_curr - x_m_std_curr.mean(1, keepdim=True)

                A_c = W_curr[..., None] * uv_c_torch * blend_curr_a[:, key, None]
                B_c = xm_c_torch * blend_curr_a[:, key, None]

                # Covariance matrix
                H = A_c.permute(0, 2, 1).bmm(B_c)
                U, S, V = torch.svd(H)
                R = V.bmm(U.permute(0, 2, 1))
                with torch.no_grad():
                    det_r = torch.linalg.det(R) < 0
                    sign = torch.ones((batch_size, 3, 3), device=this_device, dtype=V.dtype)
                    sign[det_r, :, -1] = -1
                V = V * sign
                R = V.bmm(U.permute(0, 2, 1))

                RT_var_curr = torch.eye(4, device=this_device, dtype=torch.float32).repeat(batch_size, 1, 1)
                RT_var_curr[:, :3, :3] = R.permute(0, 2, 1)

                valid_R_list.append(RT_var_curr)
                A_size += uv_curr.shape[1]
                key_size += 1
                kin_list[key] = [x_m_curr, b_std_curr, uv_curr, RT_var_curr, blend_curr_a, W_curr]

        if self.run_pliks:
            # Set fixed location for smpl params on the A matrix
            rot_reg_idx = [[A_size * 2 + y + 3 * x, y + 6 * x] for x in range(key_size) for y in range(3)]
            rot_reg_idx = list(zip(*rot_reg_idx))
            bet_reg_idx = [[-y - 1, -y - 1] for y in range(9, -1, -1)]
            bet_reg_idx = list(zip(*bet_reg_idx))
            trans_reg_idx = [[A_size * 2 + key_size * 3, A_size * 2 + key_size * 3 + 1, A_size * 2 + key_size * 3 + 2],
                             [3, 4, 5]]
            rot_reg_idx = torch.LongTensor(rot_reg_idx).to(this_device)
            bet_reg_idx = torch.LongTensor(bet_reg_idx).to(this_device)
            trans_reg_idx = torch.LongTensor(trans_reg_idx).to(this_device)

            valid_R_list_stack = torch.stack(valid_R_list).permute(1, 0, 2, 3)
            m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23 = K_out[:, [0, 1, 3]].reshape(batch_size, -1).T
            r_curr = None
            W_irls = None
            for itr in range(self.run_iters):
                if not r_curr is None:
                    valid_R_list_stack = r_curr
                _s_idx = 0
                xm_ = []
                b_s_ = []
                uv_ = []
                w_ = []
                bc_ = []
                for key in kin_list.keys():
                    if kin_list[key] is not None:
                        x_m_curr, b_std_curr, uv_curr, RT_var_curr, blend_curr, W = kin_list[key]
                        blend_curr = blend_curr[:, ]
                        #pre compute the rotation on indiviual segments
                        xm_.append(
                            blend_curr[:, :, None].permute(1, 0, 2).unsqueeze(0) * (valid_R_list_stack @ torch.cat(
                                [x_m_curr, torch.ones((x_m_curr.shape[0], 1), device=this_device, dtype=torch.float32)],
                                -1).T.unsqueeze(
                                0)).permute(0, 1, 3, 2)[..., :3])
                        b_s_.append((blend_curr[:, :, None, None].unsqueeze(1) * (valid_R_list_stack @ torch.cat(
                            [b_std_curr, torch.ones((b_std_curr.shape[0], 1, b_std_curr.shape[-1]), device=this_device,
                                                    dtype=torch.float32)], 1).unsqueeze(1).unsqueeze(1))).permute(1, 2,
                                                                                                                  0, 3,
                                                                                                                  4)[:,
                                    :, :, :3])
                        uv_.append(uv_curr)
                        w_.append(W)
                        bc_.append(blend_curr)

                xm_ = torch.cat(xm_, 2)
                b_s_ = torch.cat(b_s_, 2)
                uv_curr = torch.cat(uv_, 1)
                W = torch.cat(w_, 1)
                blend_curr = torch.cat(bc_, 0)

                t_pred_dl = torch.cat([pred_trans, pred_depth[..., None]], -1)
                if self.use_mean_shape:
                    pred_beta_ = torch.zeros_like(pred_beta)
                else:
                    pred_beta_ = pred_beta
                r_curr, t_pred, pred_beta_anal, r_pred = solve_lstq(uv_curr, xm_, b_s_, blend_curr, W, batch_size,
                                                                    this_device, m00, m11, m02, m12,
                                                                    0, 0, key_size, A_size, valid_R_list_stack,
                                                                    rot_reg_idx, bet_reg_idx, trans_reg_idx, t_pred_dl,
                                                                    pred_beta_, self.lambda_factor)

            results = [r_curr[:, 0, :3, :3]]
            for i in range(1, self.smpl.kintree_table.shape[1]):
                results.append(
                    torch.matmul(r_curr[:, parent[i], :3, :3].permute(0, 2, 1), r_curr[:, i, :3, :3])
                )
            results = torch.stack(results).permute(1, 0, 2, 3)

            v_shaped = torch.tensordot(pred_beta_anal, self.smpl.shapedirs, dims=([1], [2])) + self.smpl.v_template
            J = torch.matmul(self.smpl.J_regressor, v_shaped)
            trans_pred = t_pred[:, 0] - (J[:, 0] - ((results[:, 0] @ J[:, 0].unsqueeze(-1))[..., 0]))
            pred_beta = pred_beta_anal
            pred_pose = matrix_to_axis_angle(results).reshape(batch_size, -1)
        else:
            r_curr = torch.stack(valid_R_list).permute(1, 0, 2, 3)[..., :3, :3].reshape(-1, 24, 3, 3)
            results = [r_curr[:, 0, :3, :3]]
            for i in range(1, self.smpl.kintree_table.shape[1]):
                results.append(
                    torch.matmul(r_curr[:, parent[i], :3, :3].permute(0, 2, 1), r_curr[:, i, :3, :3])
                )
            results = torch.stack(results).permute(1, 0, 2, 3)
            pred_pose = matrix_to_axis_angle(results).reshape(batch_size, -1)
            trans_pred = torch.cat([pred_trans, pred_depth[..., None]], -1)
        return pred_pose, pred_beta, trans_pred, K_out
