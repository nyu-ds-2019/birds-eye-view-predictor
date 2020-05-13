# you must train the autoencoder first
python3 road_view_prediction/train_rm_resnet.py \
    --batch-size 32 \
    --num-workers 2 \
    --learning-rate 1e-3 \
    --num-epochs 50 \
    --train-views front_left,front,front_right,back_left,back,back_right \
    --ae-model ../artifacts/models/autoencoder/best_performing.pt \
    --data-directory ../artifacts/data \
    --model-directory ../artifacts/models/topview_resnet \
    --save-freq 1