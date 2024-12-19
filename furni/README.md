git clone this repo and move it to one layer under arp's directory.

## dataset.py
This contains a dataloader that provides batches made from the furniture-bench trajectory datasets. 

Run `python dataset.py` to see the keys and values' shape of each batch. You probably need to change the input of the `data_dir` argument, since it points to the directory where the furniture-bench trajectories are stored. The output should look like
```
Number of batches: 1000

The keys and the values' shapes of each batch:
state_ee_position: torch.Size([16, 32, 3])          dtype: torch.float32
state_ee_quaternion: torch.Size([16, 32, 4])          dtype: torch.float32
state_ee_velocity: torch.Size([16, 32, 3])          dtype: torch.float32
state_ee_angular_velocity: torch.Size([16, 32, 3])          dtype: torch.float32
state_joint_positions: torch.Size([16, 32, 7])          dtype: torch.float32
state_joint_velocities: torch.Size([16, 32, 7])          dtype: torch.float32
state_joint_torques: torch.Size([16, 32, 7])          dtype: torch.float32
state_gripper_width: torch.Size([16, 32, 1])          dtype: torch.float32
action_ee_delta_position: torch.Size([16, 32, 3])          dtype: torch.float32
action_ee_delta_quaternion: torch.Size([16, 32, 4])          dtype: torch.float32
action_gripper_range: torch.Size([16, 32])          dtype: torch.float32
wrist_camera_rgb: torch.Size([16, 32, 224, 224, 3])          dtype: torch.float32
front_camera_rgb: torch.Size([16, 32, 224, 224, 3])          dtype: torch.float32
```

## training_pseudocode.ipynb
This shows the pseudocode for the training loop of imitation learning using arp. You probably need to move this file to the parent directory of arp so that the imports in the beginning can work.
