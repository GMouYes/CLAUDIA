EXPNAME='binary_extrasensory_lr1e-4'
echo "EXPNAME $EXPNAME"

python3 -u trainer.py \
    --config_path "../config/config_lr1e-4.yml" \
    --expName $EXPNAME \
    --outputPath '../output/'$EXPNAME \
    --lossPath $EXPNAME'.jpg' \
    1> "../log/"$EXPNAME".log" \
    2> "../log/"$EXPNAME".err" &

