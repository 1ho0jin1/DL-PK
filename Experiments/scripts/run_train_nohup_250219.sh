# dose 2222: 1000mg per 12 hours
dataset=dataset/dose_2222

config=gru_250219_noaug.yaml
runname=gru_250219_noaug
nohup python train.py --data_dir $dataset --yaml_path configs/$config --run_name $runname > $runname.log 2>&1

config=gru_250219_aug.yaml
runname=gru_250219_aug
nohup python train.py --data_dir $dataset --yaml_path configs/$config --run_name $runname > $runname.log 2>&1

config=node_250219_noaug.yaml
runname=node_250219_noaug
nohup python train.py --data_dir $dataset --yaml_path configs/$config --run_name $runname > $runname.log 2>&1

config=node_250219_aug.yaml
runname=node_250219_aug
nohup python train.py --data_dir $dataset --yaml_path configs/$config --run_name $runname > $runname.log 2>&1