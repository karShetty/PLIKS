import os
import torch
from pliks.smpl import SMPL
from  pliks.utils import  overlay_image, vtk_write, create_polydata
from pliks.dataset.dataloader_util import augmentation, process_bbox
from pliks.model.PLIKS import PLIKS
import pliks.config as cfg
import numpy as np
import vedo as vtkp
import cv2
# import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.patches import Rectangle


class DemoModule():

    def __init__(self, options):
        self.options = options
        self.device = torch.device("cuda")

        if not os.path.exists(options.out_dir):
            os.makedirs(options.out_dir)

        self.smpl = SMPL().to(self.device)
        kintree_table = self.smpl.kintree_table.cpu().data.numpy()
        id_to_col = {kintree_table[1, i]: i
                     for i in range(kintree_table.shape[1])}
        parent = {
            i: id_to_col[kintree_table[0, i]]
            for i in range(1, kintree_table.shape[1])
        }

        x_mean = self.smpl.v_template
        b_std = self.smpl.shapedirs
        blend_w = self.smpl.weights
        if self.options.model == 'MeshRegSparse':
            kin_list, v_sel = np.load("model_files/v_select_sparse.npy", allow_pickle=True)
            x_mean = x_mean[v_sel]
            b_std = b_std[v_sel]
            blend_w = blend_w[v_sel]
        else:
            seg_idx = self.smpl.weights.cpu().data.numpy().argmax(1)
            kin_list = {x: np.where(seg_idx == x)[0] for x in range(24)}

        self.pliks = PLIKS(self.smpl, parent, kin_list, x_mean, b_std, blend_w,
                           sparse=self.options.sparse,
                           lambda_factor=self.options.lambda_factor,
                           use_mean_shape=self.options.use_mean_shape,
                           run_iters=self.options.run_iters,
                           run_pliks=self.options.run_pliks
                           )

        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        # matplotlib.use('Qt5Agg')#Dont know why yolo creates an issue

        self.device = torch.device("cuda")
        self.model = Regressor().to(self.device)
        self.model.load_state_dict(torch.load(self.options.checkpoint)['state_dict'], strict=False)
        self.model.eval()

        self.smpl = SMPL().to(self.device)
        self.norm_img_mean = torch.FloatTensor(cfg.IMG_NORM_MEAN).to(self.device)
        self.norm_img_std = torch.FloatTensor(cfg.IMG_NORM_STD).to(self.device)
        K = np.asarray([[-1000 * 2 / self.options.img_res, 0, 0, 0],
                        [0, -1000 * 2 / self.options.img_res, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])
        self.K_fix = torch.FloatTensor(K).to(self.device)[None]




    def vis_step(self, input_path, viz=True):
        with torch.no_grad():
            input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
            out_name, ext = os.path.splitext(os.path.basename(input_path))

            xyxy = self.yolo(input_img).xyxy[0].cpu().data.numpy()
            xyxy = xyxy[(xyxy[:,4] > 0.6) & (xyxy[:,5] == 0)]
            if xyxy is None:
                return

            smpl_v_list = []

            for _p in xyxy:
                bbox = process_bbox([_p[0], _p[1], _p[2] - _p[0], _p[3] - _p[1] ], input_img.shape[1], input_img.shape[0])
                img, img2bb_trans, bb2img_trans, _, _, _, _ = augmentation(input_img, data_split='test', img_res=self.options.img_res, bbox=bbox)
                img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
                images = img[None,].to(self.device)
                images = (images - self.norm_img_mean[None, :, None, None]) / self.norm_img_std[None, :, None, None]
                batch_size = 1
                f = np.sqrt(input_img.shape[1]**2 + input_img.shape[0]**2) #default focal lenght based on size of image
                focal, princpt = [f, f], [input_img.shape[1] / 2, input_img.shape[0] / 2]
                K1 = np.asarray([[focal[0], 0, princpt[0], 0],
                                 [0, focal[1], princpt[1], 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]])
                K2 = np.asarray([[img2bb_trans[0, 0], img2bb_trans[0, 1], 0, img2bb_trans[0,-1]],
                                 [img2bb_trans[1, 0], img2bb_trans[1, 1], 0, img2bb_trans[1,-1]],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
                K3 = np.asarray([[-2 / (self.options.img_res - 1), 0, 0, 1],
                                 [0, -2 / (self.options.img_res - 1), 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
                K_img = K3 @ K2 @ K1
                K_img = torch.FloatTensor(K_img)[None,].to(self.device)

                with torch.cuda.amp.autocast():
                    pred_smpl_uvd, pred_beta, pred_scale, pred_trans = self.model(images)


                pred_depth =  (2*1000/(self.options.img_res * (pred_scale[:,0] + 0.9) +1e-9))
                K_ = self.K_fix.repeat(batch_size,1,1)
                pred_pose_anal, pred_beta_anal, trans_pred_anal, K_ = self.pliks(pred_smpl_uvd.float(),
                                                                                 pred_beta.float(),
                                                                                 pred_depth.float(),
                                                                                 pred_trans.float(),
                                                                                 K_.float(), K_img.float(),
                                                                                 )
                pred_v = self.smpl(pred_pose_anal, pred_beta_anal, False) + trans_pred_anal[:,None]
                smpl_v_list.append([pred_v[0].clone().detach(), K_[0].clone().detach(), trans_pred_anal[0,2].item()])

            if len(smpl_v_list) == 0:
                return
            color_list = np.asarray([[214, 39, 40],
                                     [44, 160, 44],
                                     [31, 119, 180],
                                     [255, 127, 14],
                                     [148, 103, 189],
                                     [23, 190, 207],
                                     [227, 119, 194],
                                     [210, 245, 60],
                                     [250, 190, 212],
                                     [0, 128, 128],
                                     [220, 190, 255],
                                     [170, 110, 40],
                                     [255, 250, 200],
                                     [128, 0, 0],
                                     [170, 255, 195],
                                     [128, 128, 0],
                                     [255, 215, 180], ]) / 255.

            if viz:
                from pytorch3d.renderer import PerspectiveCameras, Textures, RasterizationSettings, MeshRasterizer, \
                    HardPhongShader
                from pytorch3d.structures import Meshes

                mesh_list = []
                d_idx = np.argsort([x[2] for x in smpl_v_list])[::-1]
                np.random.seed(0)
                for _i, _j in enumerate(d_idx):
                    pred_vK = smpl_v_list[_j]
                    c = color_list[len(d_idx) - _i - 1]
                    mesh_list.append(vtkp.Mesh([pred_vK[0].cpu().data.numpy(), self.smpl.faces.cpu().data.numpy()], alpha=1, c = c))
                    # vtk_write(os.path.join(self.options.out_dir, out_name + '_res_%02d.ply') % (_i,),
                    #           create_polydata(pred_vK[0].cpu().data.numpy(), self.smpl.faces.cpu().data.numpy()))

                    cam = PerspectiveCameras(focal_length=(((-f), (-f)),),
                                             image_size=(np.asarray([input_img.shape[1], input_img.shape[0]]),),
                                             principal_point=(((input_img.shape[1] / 2, input_img.shape[0] / 2),)),
                                             device="cuda")

                    v_ = pred_vK[0][None]
                    c_ = torch.ones_like(v_) * torch.FloatTensor(c)[None,].to(self.device)
                    textures = Textures(verts_rgb=c_)

                    meshes_iuv = Meshes(verts=v_, faces=self.smpl.faces.repeat(batch_size, 1, 1), textures=textures)
                    raster_settings = RasterizationSettings(image_size=[input_img.shape[0], input_img.shape[1]],
                                                            max_faces_per_bin=len(self.smpl.faces))
                    rasterizer = MeshRasterizer(cameras=cam,
                                                raster_settings=raster_settings)
                    rgb_shader = HardPhongShader(device=self.device, cameras=cam)
                    fragments = rasterizer(meshes_iuv, cameras=cam)
                    rgb_images = rgb_shader(fragments, meshes_iuv)[:, :, :, :3]
                    boundary = fragments.pix_to_face.clone().detach()[..., 0]

                    boundary[boundary != -1] = 1
                    boundary[boundary == -1] = 0
                    if _i == 0:
                        real_img = torch.FloatTensor(input_img).permute(2, 0, 1)[None].to(self.device) / 255.
                    else:
                        real_img = rgb_img

                    rgb_img = overlay_image(real_img,
                                            rgb_images.permute(0, 3, 1, 2),
                                            boundary)

                cv2.imwrite( os.path.join(self.options.out_dir, out_name + '_res2D' + ext),
                             (rgb_img.permute(0, 2, 3, 1).cpu().data.numpy()[0]*255).astype(np.uint8)[...,::-1])
                vp = vtkp.Plotter(N=1, axes=0, bg='w', offscreen=True, screensize=(1920, 1080))
                vp.camera.SetPosition([4, -4, -2.5])
                vp.camera.SetViewUp([-0.27, -0.87, 0.39])
                vp.camera.SetFocalPoint([0, 0.0, 4])
                axs = vtkp.Axes(mesh_list,  # build axes for this set of objects
                                xtitle=" ",
                                ytitle=" ",
                                ztitle=" ",
                                htitle=' ',
                                hTitleFont='Kanopus',
                                hTitleJustify='bottom-right',
                                hTitleColor='red2',
                                hTitleSize=0.035,
                                hTitleOffset=(0, 0.075, 0),
                                hTitleRotation=45,
                                zHighlightZero=True,
                                xyGridTransparent=True,
                                yzGridTransparent=True,
                                zxGridTransparent=True,
                                xLabelSize=0,
                                yLabelSize=0,
                                zLabelSize=0,
                                yzGrid=True,
                                zxGrid=True,
                                zxShift=1.0,
                                xTitleJustify='bottom-right',
                                xTitleOffset=-1.175,
                                xLabelOffset=-1.75,
                                yLabelRotation=90,
                                zInverted=True,
                                tipSize=0.0025,
                                )
                mesh_list.append(axs)

                vp.show(mesh_list, at=0, interactive=False, resetcam=True, )
                vp.screenshot(os.path.join(self.options.out_dir, out_name + '_res3D' + ext),)
                vp.close()

                # f, axarr = plt.subplots(1, 2)
                # real_img = torch.FloatTensor(input_img).permute(2, 0, 1)[None].to(self.device) / 255.
                # axarr[0].imshow(rgb_img.permute(0, 2, 3, 1).cpu().data.numpy()[0])
                # axarr[1].imshow(real_img.permute(0, 2, 3, 1).cpu().data.numpy()[0])
                # for _i, _p in enumerate(xyxy[d_idx]):
                #     bbox = [_p[0], _p[1], _p[2] - _p[0], _p[3] - _p[1]]
                #     rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color_list[_i], facecolor='none')
                #     axarr[1].add_patch(rect)
                # real_img = (real_img.permute(0, 2, 3, 1).cpu().data.numpy()[0]*255).astype(np.uint8)
                # for _i, _p in enumerate(xyxy[d_idx]):
                #     c = color_list[len(d_idx) - _i - 1]
                #     real_img = cv2.rectangle(real_img, np.asarray([_p[0], _p[1]]).astype(int), np.asarray([_p[2], _p[3]]).astype(int), (c * 255))
                # plt.show()



if __name__ == '__main__':
    from pliks.options import Options
    import glob
    import random

    options = Options().parse_args()
    if options.model == 'MeshReg':
        from pliks.model.mesh_reg import Mesh_Regressor as Regressor
    else:
        from pliks.model.mesh_reg_sparse import Mesh_Regressor_Sparse as Regressor


    file_list = glob.glob(options.img_dir)
    demo = DemoModule(options)
    random.shuffle(file_list)




    for _i, path in enumerate(file_list):
        try:
            demo.vis_step(path)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(path, e)