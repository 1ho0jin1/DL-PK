runname=(
    gru_250219_aug
    gru_250219_noaug
    node_250219_aug
    node_250219_noaug
)
postfix=(
    0000
    0202
    0404
    1212
    2222
    3232
    3434
    4444
)

for name in "${runname[@]}"
do
    for post in "${postfix[@]}"
    do
        echo Processing $name$post
        python inference.py --source_dir dataset$post/test --ckpt_path runs/train/$name/best.pt --run_name $name/$post/test --plot &
    done
    wait
done