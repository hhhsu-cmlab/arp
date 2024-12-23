export LD_LIBRARY_PATH=~/anaconda3/envs/sim/lib:$LD_LIBRARY_PATH

python furniture_bench/scripts/download_dataset.py --randomness low --furniture all --out_dir ./furniture_dataset
python furniture_bench/scripts/download_dataset.py --randomness med --furniture all --out_dir ./furniture_dataset
python furniture_bench/scripts/download_dataset.py --randomness high --furniture all --out_dir ./furniture_dataset

python furniture_bench/scripts/preprocess_data.py --in-data-path furniture_dataset/low/lamp --out-data-path furniture_dataset_processed/low/lamp

python -m run run_prefix=lamp_full_bc_resnet18_low_sim rolf.demo_path=furniture_dataset_processed/low/lamp/ env.furniture=lamp rolf.encoder_type=resnet18 rolf.finetune_encoder=True gpu=0

python -m run run_prefix=$(date "+%Y-%m-%d-%H-%M-%S") rolf.demo_path=furniture_dataset_processed/low/lamp/ env.furniture=lamp rolf.encoder_type=resnet18 rolf.finetune_encoder=False gpu=0

python -m run env.id=FurnitureSim-v0  run_prefix=lamp_full_bc_resnet18_low_sim env.furniture=lamp rolf.encoder_type=resnet18 gpu=0 is_train=False init_ckpt_path=log/FurnitureDummy-v0.bc.lamp_full_bc_resnet18_low_sim.123/ckpt/ckpt_00000000050.pt

python -m run env.id=FurnitureSim-v0  run_prefix=lamp_full_bc_resnet18_low_sim env.furniture=lamp rolf.encoder_type=resnet18 gpu=0 is_train=False init_ckpt_path=log/FurnitureDummy-v0.bc.2024-12-20-13-30-57.123/ckpt/ckpt_00000000030.pt


python -m run run_prefix=$(date "+%Y-%m-%d-%H-%M-%S") rolf.demo_path=furniture_dataset_processed/low/lamp/ env.furniture=lamp  gpu=0

python -m run run_prefix=$(date "+%Y-%m-%d-%H-%M-%S") rolf.demo_path=furniture_dataset_processed/low/lamp/ env.furniture=lamp  gpu=0 is_train=False init_ckpt_path=model_8000.pth

python -m furniture_bench.scripts.run_sim_env --furniture lamp  --scripted


    vision_encoder: "dinov2"