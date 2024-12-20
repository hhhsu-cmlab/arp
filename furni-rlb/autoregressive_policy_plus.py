from math import ceil
from copy import deepcopy
import matplotlib.pyplot as plt
import wandb
from collections import defaultdict, ChainMap
from omegaconf import DictConfig
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from typing import Optional, Tuple
import torch
import torchvision.transforms as T
from torchvision import models

import torchvision
import numpy as np
import clip
from torch.cuda.amp import autocast, GradScaler
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.optim import Lamb, GradualWarmupScheduler
from utils.structure import ActResult
from argparse import Namespace
import utils.math3d as math3d
from utils.clip import clip_encode_text
from PIL import Image
from preprocess import CubePointCloudRenderer, preprocess_images_in_batch, \
    flatten_img_pc_to_points, clamp_pc_in_bound, place_pc_in_cube, generate_heatmap_from_screen_pts, \
    apply_se3_augmentation, transform_pc, grid_sample_from_heatmap, add_uniform_noise, denorm_rgb

from utils.layers import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    DenseBlock,
    FeedForward,
    FixedPositionalEncoding
)

from arp import AutoRegressivePolicy, TokenType, LayerType, ModelConfig, cat_uneven_blc_tensors
from autoregressive_policy import MultiViewTransformer, Policy


class PolicyNetwork(nn.Module):
    def __init__(self, model_cfg, env_cfg, render_device):
        '''
        model_cfg: see the "model.hp" key in config yaml file
        '''
        super().__init__()


        self.resnet = models.resnet50(pretrained=True)
        self.resnet.eval()  # Ensure the model is in evaluation mode
        self.fc = nn.Linear(1000, model_cfg.arp_cfg["n_embd"])

        # self._num_rotation_classes = model_cfg.num_rotation_classes
        # self._rotation_resolution = 360 / self._num_rotation_classes
        # self._image_resolution = [env_cfg.image_size, env_cfg.image_size]
        # self._transform_augmentation = model_cfg.transform_augmentation
        # self._place_with_mean = model_cfg.place_with_mean
        # self._transform_augmentation_xyz = torch.from_numpy(np.array(model_cfg.transform_augmentation_xyz))
        # self._transform_augmentation_rpy = model_cfg.transform_augmentation_rpy
        # self._transform_augmentation_rot_resolution = self._rotation_resolution

        # self.gt_hm_sigma = model_cfg.gt_hm_sigma
        # self.add_rgc_loss = model_cfg.add_rgc_loss
        # self.amp = model_cfg.amp

        # self.scene_bounds = env_cfg.scene_bounds
        # self.cameras = env_cfg.cameras
        # self.move_pc_in_bound = model_cfg.move_pc_in_bound

        # self.rotation_aug = model_cfg.rotation_aug # 2
        # self.stage2_zoom_scale = model_cfg.stage2_zoom_scale # st_sca
        # self.stage2_waypoint_label_noise = model_cfg.stage2_waypoint_label_noise # st_wpt_loc_aug
        # self.point_augment_noise = model_cfg.point_augment_noise # img_aug_2

        # self.num_all_rot = self._num_rotation_classes * 3
        # self.proprio_dim = model_cfg.proprio_dim

        self.img_size = model_cfg.img_size

        # self.img_patch_size = model_cfg.img_patch_size
        # self.renderer = CubePointCloudRenderer(render_device, (model_cfg.img_size, model_cfg.img_size), with_depth=model_cfg.add_depth, cameras=model_cfg.mvt_cameras)
        # self.num_cameras = len(model_cfg.mvt_cameras)
        # if model_cfg.render_with_cpp:
        #     assert model_cfg.mvt_cameras == ['top', 'left', 'front']
        #     self.render_with_cpp = True
        #     from point_renderer.rvt_renderer import RVTBoxRenderer
        #     self.cpp_renderer = RVTBoxRenderer(device=render_device,
        #                                        img_size=(model_cfg.img_size, model_cfg.img_size),
        #                                        three_views=True,
        #                                        with_depth=model_cfg.add_depth)
        # else:
        #     self.render_with_cpp = False

        # self.mvt1 = MultiViewTransformer(model_cfg, renderer=self.renderer)
        # self.mvt2 = MultiViewTransformer(model_cfg, renderer=self.renderer)

        # self.spatial_logits_buffer = []

        # def sample_callback(lst_of_spatial_logits):
        #     assert len(lst_of_spatial_logits) == 1
        #     self.spatial_logits_buffer.append(lst_of_spatial_logits[0])
        #     bs = len(lst_of_spatial_logits[0])
        #     dev = lst_of_spatial_logits[0].device
        #     return torch.zeros(bs, 1, 2, device=dev) # dummy output

        # self.sample_callback = sample_callback

        # # produce each xyz for stage 1
        # # then use xyz feature as a condition, to produce each xyz for stage 2
        # # then produce rot and grip separately

        self.device = render_device
        print(f"[PolicyNetwork] device: {self.device}")

        self.arp_cfg = model_cfg.arp_cfg

        self.policy = AutoRegressivePolicy(ModelConfig(
            n_embd=self.arp_cfg["n_embd"],
            embd_pdrop=self.arp_cfg["embd_pdrop"],
            max_seq_len=self.arp_cfg["max_seq_len"],
            max_chunk_size=self.arp_cfg["max_seq_len"],
            layers=[
                LayerType.make(**self.arp_cfg["layer_cfg"], condition_on='visual-tokens')
            ] * self.arp_cfg["num_layers"],
            tokens=[
                TokenType.make(
                    name='state_ee_position', is_continuous=True, dim=3, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='state_ee_quaternion', is_continuous=True, dim=4, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='state_ee_velocity', is_continuous=True, dim=3, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='state_ee_angular_velocity', is_continuous=True, dim=3, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='state_joint_positions', is_continuous=True, dim=7, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='state_joint_velocities', is_continuous=True, dim=7, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='state_joint_torques', is_continuous=True, dim=7, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='action_ee_delta_position', is_continuous=True, dim=3, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='action_ee_delta_quaternion', is_continuous=True, dim=4, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                ),
                TokenType.make(
                    name='action_gripper_range', is_continuous=True, dim=1, embedding='linear', predictor='gmm', 
                    predictor_kwargs={'num_latents': self.arp_cfg["num_latents"]}
                )
            ]
        )).to(self.device)

        print(f"[PolicyNetwork] policy device: {print(next(self.policy.parameters()).device)}")

        self.id_2_token_names = [""] * len(self.policy.token_name_2_ids)
        for name, id in self.policy.token_name_2_ids.items():
            self.id_2_token_names[id] = name
        
        # gripper state only depends on xyz, but not rotation
        # self.block_attn_directions = [(n, f'rot-{c}') for c in ['x', 'y', 'z'] for n in ['grip', 'collision']]
        # self.cfg = model_cfg
    
    
    # def multi_view_coordinate_sampler(self, lst_of_spatial_logits):
    #     hm_logits = torch.cat([a for a in lst_of_spatial_logits], dim=1)
    #     hm = F.softmax(hm_logits.flatten(2), dim=2)
    #     bs = len(hm_logits)
    #     hm = hm.view(bs, 3, 224, 224)
    #     pred_pt = [self.renderer.get_most_likely_point_3d(hm[i : i + 1]) for i in range(bs)]
    #     spatial_point = torch.cat(pred_pt, 0) # bs, 3
    #     screen_points = self.renderer.points3d_to_screen2d(spatial_point[:, None, :])
    #     screen_points = screen_points[:, 0]
    #     return spatial_point, screen_points
    
    # def to_tk_reg_ids(self, token_name_regs):
    #     result = []
    #     for v in token_name_regs:
    #         r = [self.token_name_2_ids[v[0]], v[1]]
    #         if len(v) == 3: r.append(v[2])
    #         result.append(r)
    #     return result

    # def get_gt_rot_grip_collision(
    #     self,
    #     batch_size,
    #     action_rot,
    #     action_grip,
    #     action_ignore_collisions,
    #     device,
    # ):
    #     """
    #     :param batch_size: int
    #     :param action_rot: np.array of shape (bs, 4), quternion xyzw format
    #     :param action_grip: torch.tensor of shape (bs)
    #     :param action_ignore_collisions: torch.tensor of shape (bs)
    #     :param device:
    #     """
    #     bs = batch_size
    #     assert action_rot.shape == (bs, 4)
    #     assert action_grip.shape == (bs,), (action_grip, bs)

    #     action_rot_x_one_hot = torch.zeros(
    #         (bs, self._num_rotation_classes), dtype=int, device=device
    #     )
    #     action_rot_y_one_hot = torch.zeros(
    #         (bs, self._num_rotation_classes), dtype=int, device=device
    #     )
    #     action_rot_z_one_hot = torch.zeros(
    #         (bs, self._num_rotation_classes), dtype=int, device=device
    #     )
    #     action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
    #     action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

    #     # fill one-hots
    #     for b in range(bs):
    #         gt_rot = action_rot[b]
    #         gt_rot = math3d.quaternion_to_discrete_euler(
    #             gt_rot, self._rotation_resolution
    #         )
    #         action_rot_x_one_hot[b, gt_rot[0]] = 1
    #         action_rot_y_one_hot[b, gt_rot[1]] = 1
    #         action_rot_z_one_hot[b, gt_rot[2]] = 1

    #         # grip
    #         gt_grip = action_grip[b]
    #         action_grip_one_hot[b, gt_grip] = 1

    #         # ignore collision (to one hot, if result = 0, then don't ignore collision)
    #         gt_ignore_collisions = action_ignore_collisions[b, :]
    #         action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

    #     return (
    #         action_rot_x_one_hot,
    #         action_rot_y_one_hot,
    #         action_rot_z_one_hot,
    #         action_grip_one_hot,
    #         action_collision_one_hot,
    #     )

    # def get_gt_translation_action(
    #     self,
    #     waypoint, # this is groundtruth 3d point
    #     dims,
    # ): # note: will be called separately for stage 1 / 2
    #     bs, nc, h, w = dims
    #     wpt_img = self.renderer.points3d_to_screen2d(waypoint.unsqueeze(1))
    #     assert wpt_img.shape[1] == 1
    #     wpt_img = wpt_img.squeeze(1)  # (bs, num_img, 2)
    #     action_trans = generate_heatmap_from_screen_pts(
    #         wpt_img.reshape(-1, 2), #! just the winning points
    #         (h, w),
    #         sigma=self.gt_hm_sigma,
    #         thres_sigma_times=3,
    #     )
    #     action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()
    #     return action_trans, wpt_img

    # def render(self, pc, img_feat, mvt: MultiViewTransformer):
    #     renderer = self.cpp_renderer if self.render_with_cpp else self.renderer
    #     with torch.no_grad():
    #         with autocast(enabled=False):
    #             if mvt.add_corr:
    #                 if mvt.norm_corr:
    #                     img = []
    #                     for _pc, _img_feat in zip(pc, img_feat):
    #                         max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
    #                         img.append(
    #                             renderer(_pc, torch.cat((_pc / max_pc, _img_feat), dim=-1)).unsqueeze(0) # [3, 224, 224, 7], 3 -> views, 7 -> feats
    #                         )
    #                 else:
    #                     img = [renderer(_pc, torch.cat((_pc, _img_feat), dim=-1)).unsqueeze(0) for _pc, _img_feat in zip(pc, img_feat)]
    #             else:
    #                 img = [renderer(_pc, _img_feat).unsqueeze(0) for _pc, _img_feat in zip(pc, img_feat)]

    #     img = torch.cat(img, 0)
    #     img = img.permute(0, 1, 4, 2, 3) # [1, 3, 7, 224, 224]

    #     if mvt.add_pixel_loc:
    #         bs = img.shape[0]
    #         pixel_loc = mvt.pixel_loc.to(img.device) # extra feature
    #         img = torch.cat(
    #             (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
    #         )
    #     return img

    def forward(self, batch):
        batch_size = batch["state_ee_position"].size(0)

        if self.training: # this is true if self.train() is called

            token_tensors = [batch[token_name] for token_name in self.id_2_token_names]

            # tk_vals: [batch_size, num_tokens, max_dim_of_given_tokens]
            # tk_ids: [batch_size, num_tokens, 1]
            tk_vals = cat_uneven_blc_tensors(*token_tensors) 
            tk_ids = torch.arange(len(self.id_2_token_names), device=self.device)[None, :, None].repeat(tk_vals.size(0), 1, 1)

            tks = torch.cat([tk_vals, tk_ids], dim=-1) # [batch_size, num_tokens, max_dim_of_given_tokens + 1]

            front_camera_rgb_image = batch["front_camera_rgb"]
            # print(type(front_camera_rgb_image))
            # print(front_camera_rgb_image.shape)

            transform = T.Compose([
                        T.ToPILImage(),  # 將 Tensor 轉換為 PIL Image
                        T.Resize((224, 224)),  # 調整大小
                        T.ToTensor(),  # 將 PIL Image 轉換為 Tensor
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
            ])

                        
            # # 假設 front_camera_rgb_image 的形狀為 [batch_size, 1, 224, 224, 3]
            # # 取出批次中的第一張影像，形狀為 [1, 224, 224, 3]
            # image_tensor = front_camera_rgb_image[0]

            # # 移除不必要的維度，形狀變為 [224, 224, 3]
            # image_tensor = image_tensor.squeeze(0)

            # # 將張量從 GPU 移動到 CPU（如果尚未在 CPU 上）
            # image_tensor = image_tensor.cpu() /255.0
            # # 將張量數據類型轉換為 numpy 陣列
            # image_np = image_tensor.numpy()

            # # print(image_np)


            # # 顯示影像
            # plt.imshow(image_np)
            # plt.title("Front Camera RGB Image")
            # plt.axis("off")
            # plt.show()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 對每張圖片進行編碼
            encoder_out=[]
            for i in range(batch_size):
                img = front_camera_rgb_image[i]
                img = img.squeeze(0)  # 擠壓掉不必要的維度，現在 img 形狀應該是 [224, 224, 3]
                img = img.permute(2, 0, 1)
                # print(img.shape)  # 確認 img 的形狀
                img = transform(img)  # 現在可以正常進行轉換
                img = img.unsqueeze(0)  # 增加批次維度

                img = img.half()  # 將輸入數據轉換為 float16

                with torch.no_grad():
                    img = img.to(device)
                    features = self.resnet(img)
                    # print(features.shape) # 查看輸出形狀
                    embedding = self.fc(features)
                encoder_out.append(embedding)

            encoder_out = torch.stack(encoder_out)


            # encoder_out = torch.rand(batch_size, 1, self.arp_cfg["n_embd"], device=self.device)
            # print(encoder_out.shape)

            loss_dict = self.policy.compute_loss(tks, chk_ids=None, contexts={'visual-tokens': encoder_out})

            return loss_dict
        else:
            # # TODO: evaluation code
            pass

            # prompt_seq = torch.zeros([bs, 0, 3], device=dev, dtype=torch.float32)
            # future_tk_chk_ids = [dict(chk_id=0, tk_id=self.policy.token_name_2_ids['stage2-screen-pts'])]
            # for view_id in range(3):
            #     self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='cross', sample_function=self.sample_callback,
            #                         contexts={
            #                                 'visual-tokens': visual_featmap_2[:, view_id].flatten(-2, -1).permute(0, 2, 1),
            #                                 'visual-featmap': visual_featmap_2[:, view_id],
            #                         })
            #     assert len(self.spatial_logits_buffer) == (view_id + 1)

            # hms = torch.cat([F.softmax(hm_logits.reshape(bs, -1), dim=1).reshape(bs, 1, 224, 224) 
            #                  for hm_logits in self.spatial_logits_buffer], dim=1)
            # pred_pt = [self.renderer.get_most_likely_point_3d(hms[i : i + 1]) for i in range(bs)]
            # waypoint_stage2 = torch.cat(pred_pt, 0) # bs, 3
            # self.spatial_logits_buffer.clear()

            # screen_waypoint_stage2 = self.renderer.points3d_to_screen2d(waypoint_stage2[:, None, :])[:, 0]

            # prompt_features = torch.cat([ # [bs, 6, 128]
            #         grid_sample_from_heatmap(screen_waypoint_stage2.reshape(-1, 1, 2) / self.img_patch_size, 
            #                                 visual_featmap_2.flatten(0, 1))[0].reshape(bs, -1, 128),
            #         visual_featmap_2.max(dim=-1)[0].max(dim=-1)[0]], dim=1)
            
            # prompt_seq = torch.as_tensor([(i, self.policy.token_name_2_ids['prompt-features']) for i in range(6)], 
            #                       device=dev).reshape(1, 6, 2).repeat(bs, 1, 1)
            # future_tk_chk_ids = [dict(chk_id=chk_id, tk_id=self.policy.token_name_2_ids[tk_name]) 
            #                      for chk_id, tk_name in zip(range(6, 11), ['rot-x', 'rot-y', 'rot-z', 'grip', 'collision'])]
            
            # result_seq_stage2 = self.policy.generate(prompt_seq, future_tk_chk_ids, match_layer='self', 
            #                                         sample=False,  block_attn_directions=self.block_attn_directions,  
            #                                         contexts={
            #                                             'prompt-features': prompt_features
            #                                         })

        
