PORT1=31337

MINIBATCH_SIZE=256
BATCH_SIZE=2048
N_EPOCHS=800
N_GPUS=8
WARMUP_EPOCHS=40

IMAGENET_SIZE=1281167
WARMUP_STEPS=$(($IMAGENET_SIZE * $WARMUP_EPOCHS / $BATCH_SIZE))
MAX_STEPS=$(($IMAGENET_SIZE * $N_EPOCHS / $BATCH_SIZE))
GRAD_ACCUM=$(($BATCH_SIZE / $MINIBATCH_SIZE / $N_GPUS))

torchrun --nproc_per_node=$N_GPUS --master_port=$PORT1 main.py \
--output_dir $SNAPSHOT_PATH \
--overwrite_output_dir \
--dataset_class ImageFolder \
--train_dir $INPUT_PATH/mnt/data/imagenet/train \
--validation_dir $INPUT_PATH/mnt/data/imagenet/val \
--dataloader_num_workers 32 \
--num_readers 4 \
--do_train \
--fp16 True \
--save_steps 6250 \
--save_total_limit 3 \
--learning_rate 2e-3 \
--num_train_epochs $N_EPOCHS \
--per_device_train_batch_size $MINIBATCH_SIZE \
--warmup_steps $WARMUP_STEPS \
--max_steps $MAX_STEPS \
--gradient_accumulation_steps $GRAD_ACCUM \
--seed 1337 \
--n_layers_to_average 8 \
--momentum 0.9998 \
--patch_size 16 \
--image_size 224 \
--mask_ratio 0.6 \
 \
--ddp_find_unused_parameters True \
--report_to wandb \
--logging_steps 5 \
--run_name nirvana_batch256_fp32 \
--disable_tqdm True \
