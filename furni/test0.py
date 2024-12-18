from rlb.autoregressive_policy_plus import PolicyNetwork, Policy

import furniture_bench
import gym

import os
import wandb
import os.path as osp
import torch
from copy import copy
from tqdm import tqdm
import logging
from time import time
import sys, shlex
# from utils import configurable, DictConfig, config_to_dict
# import torch.multiprocessing as mp
# from utils.dist import find_free_port
# import torch.distributed as dist
# from dataset import TransitionDataset
# from utils.structure import RLBENCH_TASKS
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data import DataLoader
# from runstats import Statistics

from omegaconf import DictConfig, OmegaConf

"""
# Input
action: torch.Tensor or np.ndarray (shape: [num_envs, action_dim]) # Action space is 8-dimensional (3D EE delta position, 4D EE delta rotation (quaternion), and 1D gripper.Range to [-1, 1].

# Output
obs: Dictionary of observations. The keys are specified in obs_keys. The default keys are: ['color_image1', 'color_image2', 'robot_state'].
reward: torch.Tensor or np.ndarray (shape: [num_envs, 1])
done: torch.Tensor or np.ndarray (shape: [num_envs, 1])
info: Dictionary of additional information.
"""



def main0(cfg: DictConfig, log_dir:str):
    device = "cuda:0" # should come from cfg

    env = gym.make(
        "FurnitureSim-v0",
        furniture='one_leg',
        num_envs=1,
    )

    net = PolicyNetwork(cfg.model.hp, cfg.env, render_device=f"cuda:{device}").to(device)

    agent = Policy(net, cfg.model.hp, log_dir=log_dir)
    agent.build(training=True, device=device)

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    if cfg.train.eval_mode:
        agent.eval()
        torch.set_grad_enabled(False)
    else:
        agent.train()

    ob = env.reset()
    done = False

    while not done:
        batch = {
            "gripper_action": ob[]
            action_grip = observation["gripper_action"].int() # (b,) of int
            action_ignore_collisions = observation["ignore_collisions"].view(-1, 1).int()  # (b, 1) of int
            action_gripper_pose = observation["gripper_pose"]  # (b, 7)
            action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3), translation in xyz
            action_rot = action_gripper_pose[:, 3:7] 
        }

        {
                "lang_goal_tokens": clip.tokenize(load_pkl(osp.join(episode_path, DESC_PICKLE))[0])[0].numpy(),
                "lang_goal_embs": episode['lang_emb'],
                "keypoint_idx": kp,
                "kp_frame_idx": kp_frame_id,
                "frame_idx": obs_frame_id,
                "episode_idx": episode_idx,
                "variation_idx": variation_id,
                "task_idx": self.tasks.index(task),

                "gripper_pose": essential_kp_obs.gripper_pose,
                "ignore_collisions": int(essential_kp_obs.ignore_collisions),

                "gripper_action": int(essential_kp_obs.gripper_open),
                "low_dim_state": curr_low_dim_state,

                **obs_media_dict
            }
        ac = net() # net(???????????????)
        ob, rew, done, _ = env.step(ac)



def main_single(rank: int, cfg: DictConfig, port: int, log_dir:str):
    if cfg.wandb and rank == 0:
        wandb.init(project=cfg.wandb, name='/'.join(log_dir.split('/')[-2:]), config=config_to_dict(cfg))

    world_size = cfg.train.num_gpus
    assert world_size > 0
    ddp, on_master = world_size > 1, rank == 0
    print(f'Rank - {rank}, master = {on_master}')
    if ddp:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = rank
    if on_master:
        logfile = open(osp.join(log_dir, 'log.txt'), "w")

    def log(msg, printer=print):
        if on_master:
            print(msg, file=logfile, flush=True)
            printer(msg)

    env_cfg = cfg.env
    if env_cfg.tasks == 'all':
        tasks = RLBENCH_TASKS
    else:
        tasks = env_cfg.tasks.split(',')

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    cfg.model.hp.lr *= (world_size * cfg.train.bs)
    cfg.model.hp.cos_dec_max_step = cfg.train.epochs * cfg.train.num_transitions_per_epoch // cfg.train.bs // world_size

    py_module = cfg.py_module
    from importlib import import_module
    MOD = import_module(py_module)
    Policy, PolicyNetwork = MOD.Policy, MOD.PolicyNetwork

    net = PolicyNetwork(cfg.model.hp, cfg.env, render_device=f"cuda:{device}").to(device)
    if ddp:
        net = DistributedDataParallel(net, device_ids=[device])
    agent = Policy(net, cfg.model.hp, log_dir=log_dir)
    agent.build(training=True, device=device)

    start_step = 0
    if cfg.model.weights:
        start_step = agent.load(cfg.model.weights)
        log(f"Resuming from step {start_step}")
    if ddp: dist.barrier()

    total_batch_num = cfg.train.num_transitions_per_epoch * cfg.train.epochs // cfg.train.bs #(cfg.train.bs * world_size)
    total_batch_num -= (start_step * world_size)
    dataset = TransitionDataset(cfg.train.demo_folder, tasks, cameras=env_cfg.cameras,
            batch_num=total_batch_num, batch_size=cfg.train.bs, scene_bounds=env_cfg.scene_bounds,
            voxel_size=env_cfg.voxel_size, rotation_resolution=env_cfg.rotation_resolution,
            cached_data_path=cfg.train.cached_dataset_path, time_in_state=cfg.env.time_in_state,
            episode_length=cfg.env.episode_length, k2k_sample_ratios=cfg.train.k2k_sample_ratios, 
            origin_style_state=cfg.env.origin_style_state)

    log("Begin Training...")
    dataloader, sampler = dataset.dataloader(num_workers=cfg.train.num_workers, 
                                             pin_memory=False, distributed=ddp)
    log(f"Total number of batches: {len(dataloader)}")

    if ddp: sampler.set_epoch(0)
    if cfg.train.eval_mode:
        agent.eval()
        torch.set_grad_enabled(False)
    else:
        agent.train()

    train(agent, dataloader, log, device, freq=cfg.train.disp_freq, rank=rank, save_freq=cfg.train.save_freq, 
        start_step=start_step, use_wandb=cfg.wandb and rank == 0)


@configurable()
def main(cfg: DictConfig):
    if cfg.train.num_gpus <= 1:
        main_single(0, cfg, -1, cfg.output_dir)
    else:
        port = find_free_port()
        mp.spawn(main_single, args=(cfg, port, cfg.output_dir),  nprocs=cfg.train.num_gpus, join=True)

if __name__ == "__main__":
    main()
