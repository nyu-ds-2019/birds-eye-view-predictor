# you must train the autoencoder first and select the path to the desired model in ae-model below
python3 train_rm_resnet.py 
    --batch-size 32 \
    --num-workers 2 \
    --learning-rate 1e-3 \
    --num-epochs 50 \
    --train-views front_left,front,front_right,back_left,back_back_right
    --ae-model ae_1.pt \
    --data-directory ../data \
    --model-directory ../models