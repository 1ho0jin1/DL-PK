config=node_250217.yaml
runname=node_250217
weight=/home/hj/DL-PK/Experiments/runs/train/node_250209/best.pt

nohup python train.py --data_dir dataset --yaml_path configs/$config --ckpt_path $weight --run_name $runname > nohup.log 2>&1 &