config=node_250216.yaml
runname=node_250216
nohup python train.py --data_dir dataset --yaml_path configs/$config --run_name $runname > nohup.log 2>&1 &