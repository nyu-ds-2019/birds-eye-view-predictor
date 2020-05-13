python3 bounding_boxes/mono/train.py \
    --type dynamic \
    --data_path ../artifacts/data_mono/front_left \
    --save_path ../artifacts/models/mono/front_left \
    --rotation_model_path ../artifacts/models/rotation_ssl/best_performing.pt \
    --num_epochs 50 \
    --lr 1e-2 \
    --lr_D 1e-2 \
    --scheduler_step_size 20 \
    --save_freq 2

python3 bounding_boxes/mono/train.py \
    --type dynamic \
    --data_path ../artifacts/data_mono/front \
    --save_path ../artifacts/models/mono/front \
    --rotation_model_path ../artifacts/models/rotation_ssl/best_performing.pt \
    --num_epochs 50 \
    --lr 1e-2 \
    --lr_D 1e-2 \
    --scheduler_step_size 20 \
    --save_freq 2

python3 bounding_boxes/mono/train.py \
    --type dynamic \
    --data_path ../artifacts/data_mono/front_right \
    --save_path ../artifacts/models/mono/front_right \
    --rotation_model_path ../artifacts/models/rotation_ssl/best_performing.pt \
    --num_epochs 50 \
    --lr 1e-2 \
    --lr_D 1e-2 \
    --scheduler_step_size 20 \
    --save_freq 2

python3 bounding_boxes/mono/train.py \
    --type dynamic \
    --data_path ../artifacts/data_mono/back_left \
    --save_path ../artifacts/models/mono/back_left \
    --rotation_model_path ../artifacts/models/rotation_ssl/best_performing.pt \
    --num_epochs 50 \
    --lr 1e-2 \
    --lr_D 1e-2 \
    --scheduler_step_size 20 \
    --save_freq 2

python3 bounding_boxes/mono/train.py \
    --type dynamic \
    --data_path ../artifacts/data_mono/back \
    --save_path ../artifacts/models/mono/back \
    --rotation_model_path ../artifacts/models/rotation_ssl/best_performing.pt \
    --num_epochs 50 \
    --lr 1e-2 \
    --lr_D 1e-2 \
    --scheduler_step_size 20 \
    --save_freq 2

python3 bounding_boxes/mono/train.py \
    --type dynamic \
    --data_path ../artifacts/data_mono/back_right \
    --save_path ../artifacts/models/mono/back_right \
    --rotation_model_path ../artifacts/models/rotation_ssl/best_performing.pt \
    --num_epochs 50 \
    --lr 1e-2 \
    --lr_D 1e-2 \
    --scheduler_step_size 20 \
    --save_freq 2

	