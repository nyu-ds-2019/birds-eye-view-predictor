python3 bounding_boxes/mono/train.py \
	--type dynamic \
	--data_path ../../artifacts/data_mono/front_left \
	--save_path ../../artifacts/models/mono/front_left \
	--resnet_model_path ../
	--num_epochs 50 \
	--lr 1e-2 \
	--lr_D 1e-2 \
	--scheduler_step_size 20

python3 bounding_boxes/mono/train.py \
	--type dynamic \
	--data_path ../../artifacts/data_mono/front \
	--save_path ../../artifacts/models/mono/front \
	--num_epochs 50 \
	--lr 1e-2 \
	--lr_D 1e-2 \
	--scheduler_step_size 20

python3 bounding_boxes/mono/train.py \
	--type dynamic \
	--data_path ../../artifacts/data_mono/front_right \
	--save_path ../../artifacts/models/mono/front_right \
	--num_epochs 50 \
	--lr 1e-2 \
	--lr_D 1e-2 \
	--scheduler_step_size 20

python3 bounding_boxes/mono/train.py \
	--type dynamic \
	--data_path ../../artifacts/data_mono/back_left \
	--save_path ../../artifacts/models/mono/back_left \
	--num_epochs 50 \
	--lr 1e-2 \
	--lr_D 1e-2 \
	--scheduler_step_size 20

python3 bounding_boxes/mono/train.py \
	--type dynamic \
	--data_path ../../artifacts/data_mono/back \
	--save_path ../../artifacts/models/mono/back \
	--num_epochs 50 \
	--lr 1e-2 \
	--lr_D 1e-2 \
	--scheduler_step_size 20

python3 bounding_boxes/mono/train.py \
	--type dynamic \
	--data_path ../../artifacts/data_mono/back_right \
	--save_path ../../artifacts/models/mono/back_right \
	--num_epochs 50 \
	--lr 1e-2 \
	--lr_D 1e-2 \
	--scheduler_step_size 20
