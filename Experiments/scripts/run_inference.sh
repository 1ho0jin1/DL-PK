runname=(
    gru_250223_aug
    gru_250223_noaug
    node_250223_aug
    node_250223_noaug
)
postfix=(
    0000 0101 0202 0303 0404
    1010 1111 1212 1313 1414
    2020 2121 2222 2323 2424
    3030 3131 3232 3333 3434
    4040 4141 4242 4343 4444
)

for post in "${postfix[@]}"
do
    for name in "${runname[@]}"
    do
        echo Processing $name$post
        python inference.py --source_dir dataset/dose_$post/test --ckpt_path runs/train/$name/best.pt --run_name $name/$post/test --plot &
    done
    wait
done