cd ..
TRAIN_DIR=./train_log
DATASET_DIR=./tfrecords
PRETRAINED_CHECKPOINT_DIR=./pretrain_model


python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=fisher \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
