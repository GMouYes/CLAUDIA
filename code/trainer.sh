EXPNAME='binary'

CUDA_VISIBLE_DEVICES=0 nohup python3 -u trainer.py \
	--config_path "../config/config_lr3e-4_lpp_0.06_lu_0.6.yml" \
    --expName $EXPNAME \
    --outputPath '../output/'$EXPNAME \
    --lossPath $EXPNAME'.jpg' \
    1> "../log/"$EXPNAME".log" \
    2> "../log/"$EXPNAME".err" &

