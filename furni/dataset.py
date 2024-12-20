import random
import os
import pickle

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler

import os
import random

# # this implementation of dataset is wrong
# def glue_dicts(dicts: List[dict]):
#     glued = {}
#     for d in dicts:
#         for k, v in d.items():
#             if k not in glued:
#                 glued[k] = []
#             glued[k].append(v)
#     return glued
#
# class FurnitureOfflineDataset(Dataset):
#     def __init__(self, data_dir, batch_size=16):
#         super().__init__()
#         self.bs = batch_size
#         self.data_dir = data_dir

#         self.filenames = [fn for fn in os.listdir(self.data_dir) if fn.endswith('.pkl')]

#     def __len__(self):

#     def __getitem__(self, idx):
#         batch = {
#             "ee_delta_position": [],
#             "ee_delta_quaternion": [],
#             "gripper_width": []
#         }

#         b = 0
#         while b < self.bs:
#             # sample a random file
#             fn = random.choice(self.filenames)
#             with open(os.path.join(self.data_dir, fn), 'rb') as f:
#                 data = pickle.load(f)

#             if idx < len(data):
#                 batch["ee_delta_position"].append(data[idx]["ee_pos"]) # EEF position (3,)
#                 batch["ee_delta_quaternion"].append(data[idx]["ee_quat"]) # EEF orientation (4,)
#                 batch["gripper_width"].append(data[idx]["gripper_width"]) # Gripper width (1,)
#                 b += 1

#         return batch
            

# your new implementation should consider token shape (bs, L, d+1),
# where L is sequence length.
# I guess L is determined from config?
class FurnitureOfflineDataset(Dataset):
    def __init__(self, data_dir, batch_size, seq_len, batch_num: int=1000):
        super().__init__()
        self.bs = batch_size
        self.seq_len = seq_len
        self.data_dir = data_dir
        self.batch_num = batch_num

        self.filenames = [fn for fn in os.listdir(self.data_dir) if fn.endswith('.pkl')]

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        batch = {
            "state_ee_position": [],
            "state_ee_quaternion": [],
            "state_ee_velocity": [],
            "state_ee_angular_velocity": [],
            "state_joint_positions": [],
            "state_joint_velocities": [],
            "state_joint_torques": [],
            "state_gripper_width": [],
            "action_ee_delta_position": [],
            "action_ee_delta_quaternion": [],
            "action_gripper_range": [],
            "wrist_camera_rgb": [],
            "front_camera_rgb": []
        }

        n_tries = 0 # number of tries to sample a subsequence from episode
        for _ in range(self.bs):
            if n_tries > 3 * self.batch_num:
                raise ValueError("Cannot sample enough subsequences for batch. seq_len might be too large.")

            # sample a random file
            fn = random.choice(self.filenames)
            with open(os.path.join(self.data_dir, fn), 'rb') as f:
                data = pickle.load(f)

            state_traj = [item["robot_state"] for item in data["observations"]]
            action_traj = data["actions"]
            episode_len = len(action_traj) # notice that len(state_traj) == len(action_traj) + 1

            # get a subsequence of length self.seq_len
            n_tries += 1
            if episode_len < self.seq_len:
                continue
            start_idx = random.randint(0, episode_len - self.seq_len)

            # Each item below is of the following form:
            # {
            #     'ee_pos': EEF position (3,)
            #     'ee_quat': EEF orientation (4,)
            #     'ee_pos_vel': EEF linear velocity (3,)
            #     'ee_ori_vel': EEF angular velocity (3,)
            #     'joint_positions': Joint positions (7,)
            #     'joint_velocities': Joint velocities (7,)
            #     'joint_torques': Joint torques (7,)
            #     'gripper_width': Gripper width (1,)
            # }
            sub_state_traj = state_traj[start_idx:(start_idx+self.seq_len)]

            wrist_camera_rgbs = []
            for i in range(start_idx, start_idx+self.seq_len):
                wrist_camera_rgbs.append(data["observations"][i]["color_image1"])
            front_camera_rgbs = []
            for i in range(start_idx, start_idx+self.seq_len):
                front_camera_rgbs.append(data["observations"][i]["color_image2"])

            state_ee_positions = [item["ee_pos"] for item in sub_state_traj]
            state_ee_quaternions = [item["ee_quat"] for item in sub_state_traj]
            state_ee_velocities = [item["ee_pos_vel"] for item in sub_state_traj]
            state_ee_angular_velocities = [item["ee_ori_vel"] for item in sub_state_traj]
            state_joint_positions = [item["joint_positions"] for item in sub_state_traj]
            state_joint_velocities = [item["joint_velocities"] for item in sub_state_traj]
            state_joint_torques = [item["joint_torques"] for item in sub_state_traj]
            state_gripper_widths = [item["gripper_width"] for item in sub_state_traj]

            # each item below is an array of shape (8,)
            # 3D EE delta position, 4D EE delta rotation (quaternion), and 1D gripper.Range to [-1, 1].
            action_ee_delta_positions = [item[:3] for item in action_traj[start_idx:start_idx+self.seq_len]]
            action_ee_delta_quaternions = [item[3:7] for item in action_traj[start_idx:start_idx+self.seq_len]]
            action_gripper_ranges = [item[7:] for item in action_traj[start_idx:start_idx+self.seq_len]]

            batch["state_ee_position"].append(state_ee_positions)
            batch["state_ee_quaternion"].append(state_ee_quaternions)
            batch["state_ee_velocity"].append(state_ee_velocities)
            batch["state_ee_angular_velocity"].append(state_ee_angular_velocities)
            batch["state_joint_positions"].append(state_joint_positions)
            batch["state_joint_velocities"].append(state_joint_velocities)
            batch["state_joint_torques"].append(state_joint_torques)
            batch["state_gripper_width"].append(state_gripper_widths)
            batch["action_ee_delta_position"].append(action_ee_delta_positions)
            batch["action_ee_delta_quaternion"].append(action_ee_delta_quaternions)
            batch["action_gripper_range"].append(action_gripper_ranges)

            batch["wrist_camera_rgb"].append(wrist_camera_rgbs)
            batch["front_camera_rgb"].append(front_camera_rgbs)

        # change to torch tensor
        for k, v in batch.items():
            batch[k] = torch.tensor(np.array(v), dtype=torch.float32)

        return batch
    
    def dataloader(self, num_workers=1, pin_memory=True, distributed=False, pin_memory_device=''):
        if distributed:
            sampler = DistributedSampler(self)
        else:
            sampler = RandomSampler(range(len(self)))
        if pin_memory and pin_memory_device != '':
            pin_memory_device = f'cuda:{pin_memory_device}'
        return DataLoader(self, batch_size=None, shuffle=False, pin_memory=pin_memory,
                        sampler=sampler, num_workers=num_workers, pin_memory_device=pin_memory_device), sampler
    
if __name__ == "__main__":
    dataset = FurnitureOfflineDataset(
        data_dir="./furniture-bench/data/low_compressed/lamp",
        batch_size=16,
        seq_len=32
    )
    dataloader, sampler = dataset.dataloader(num_workers=0, pin_memory=False, distributed=False)

    print(f"Number of batches: {len(dataloader)}\n")

    print(f"The keys and the values' shapes of each batch:")
    for batch in dataloader:
        for k, v in batch.items():
            print(f"{k}: {v.shape}          dtype: {v.dtype}")
        break
