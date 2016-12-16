## using tf.slim to finetuning a model to new task

`python covert_somedata_to_tfrecord.py --dataset_name=train --dataset_dir=. --nFold=4`  Split the train dataset to Train and val in 4 folds.
In folder `tfrecords`, we get the fish_train_00000-of-nFold-*-00001.tfrecord and fish_validation_00000-of-nFold-*-00001.tfrecord.

Then, run `sh run.sh` to finetuning some layers to fit the new task (8 classification task)
After finetuning the model, run `sh run_eval.sh` to eval the model.

run_all.sh and run_all_eval.sh train the all layers and eval the model.

**PS**: When you train or eval the model, make tfrecords include only one folder tfrecord.

In `fish_inference.py`, we make an inference with the finetuning model.

## make RESTful api with your model

In flask_inference.py, we build an serving model with flask. It is too simple to get a image file path in your computer to make inference, 
the model is hold in your memory when the scrip is runing.
