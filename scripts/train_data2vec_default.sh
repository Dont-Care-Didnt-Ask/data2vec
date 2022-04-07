PORT1=31337

MINIBATCH_SIZE=128
BATCH_SIZE=2048
N_EPOCHS=10
N_GPUS=2
WARMUP_EPOCHS=40

#IMAGENET_SIZE=1281167
IMAGENET_SIZE=50000
WARMUP_STEPS=$(($IMAGENET_SIZE * $WARMUP_EPOCHS / $BATCH_SIZE))
MAX_STEPS=$(($IMAGENET_SIZE * $N_EPOCHS / $BATCH_SIZE))
GRAD_ACCUM=$(($BATCH_SIZE / $MINIBATCH_SIZE / $N_GPUS))

torchrun --nproc_per_node=$N_GPUS --master_port=$PORT1 main.py \
--output_dir ./outputs/ \
--overwrite_output_dir \
--dataset_class ImageFolder \
--train_dir /mnt/data/imagenet/val \
--validation_dir /mnt/data/imagenet/val \
--dataloader_num_workers 16 \
--num_readers 4 \
--do_train \
--fp16 True \
--save_steps 25 \
--save_total_limit 2 \
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
--ddp_find_unused_parameters False \
--report_to none \
--logging_steps 5 \
--run_name default_debugging \
--disable_tqdm True \
