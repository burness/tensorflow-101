cd ..
TRAIN_DIR=./train_log
DATASET_DIR=./tfrecords
PRETRAINED_CHECKPOINT_DIR=./pretrain_model

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=fisher \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3
