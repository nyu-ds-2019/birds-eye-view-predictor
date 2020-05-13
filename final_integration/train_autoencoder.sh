python3 road_view_prediction/train_autoencoder.py \
    --batch-size 32 \
    --num-workers 2 \
    --learning-rate 1e-3 \
    --num-epochs 50 \
    --random-seed 666 \
    --data-directory ../artifacts/data \
    --model-directory ../artifacts/models/autoencoder \
    --save-freq 1
