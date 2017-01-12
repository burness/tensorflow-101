## algorithm

MLP for mnist done

## covert to tfrecord

test on 102flowers done
command like this:

`python covert_somedata_to_tfrecord.py --dataset_name=102flowers --dataset_dir='./'`

## serving
test on MLP model on mnist done

download the mnist jpgs on baidu yunpan [mnist jpgs](https://pan.baidu.com/s/1o8EWkVS)

![](./images/mnist_server.png)
![](./images/mnist_client_result.png)




Reference: [https://github.com/tobegit3hub/deep_recommend_system](https://github.com/tobegit3hub/deep_recommend_system)


## finetuning and deploy model with flask

![](./images/flask_with_pretrain_model.png)
![](./images/flask_with_pretrain_model_00.png)

In folder finetuning, we use tf.slim to finetuning the pretrain model (I use the same method in my porn detection) and use flask to buid a very simple inference system.



## Inference Demo
![](./images/demo_result.png)
I deploy a image classification in [demo page](http://demo.duanshishi.com). It is based on Tensorflow and Flask. Feel free to try.

## Chinese rec

![](http://images.duanshishi.com/mac_blogs_chinese_rec_example.png)

You can get the detailed introductation in [TensorFlow与中文手写汉字识别](http://hacker.duanshishi.com/?p=1753)