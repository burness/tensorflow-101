## using tf.slim to finetuning a model to new task

1. Download the train dataset [fisher data](https://pan.baidu.com/s/1nvyLmx7) and [pretrained model](https://pan.baidu.com/s/1pLRh2DP), uncompress the dataset in train folder.

2. run the command `cd covert_pys;python covert_somedata_to_tfrecord.py --dataset_name=train --dataset_dir=. --nFold=4`  to split the train dataset to Train and val in 4 folds.
Then, in folder `tfrecords`, we get the fish_train_00000-of-nFold-*-00001.tfrecord and fish_validation_00000-of-nFold-*-00001.tfrecord.

3. run `cd run_scripts; sh run.sh` to finetuning some layers to fit the new task (8 classification task).After finetuning the model, run `sh run_eval.sh` to eval the model.

4. (if you want to finetuning the all layers)run run_all.sh and run_all_eval.sh train the all layers and eval the model.

**PS**: When you train or eval the model, make tfrecords include only one folder tfrecord.

5. In `fish_inference.py`, we make an inference with the finetuning model.

## make RESTful api with your model

In flask_inference.py, we build an serving model with flask. It is too simple to get a image file path in your computer to make inference, 
the model is hold in your memory when the scrip is runing.


I deploy a image classification in [demo page](http://demo.duanshishi.com). Feel free to try.




