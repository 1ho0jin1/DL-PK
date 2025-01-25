runname=(
    # gru_250122
    # gru_250122_curricular
    gru_250122_curricular_L1
    gru_250123_curricular
    gru_250124_2_curricular
    gru_250124_curricular
    gru_250124_curricular_longrun
    # gru_250125_longrun
)

for name in "${runname[@]}"
do
    echo Processing $name
    python inference.py --source_dir dataset/train --ckpt_path runs/train/$name/best.pt --run_name $name/train --plot &
    python inference.py --source_dir dataset/test --ckpt_path runs/train/$name/best.pt --run_name $name/test --plot &
    wait
done