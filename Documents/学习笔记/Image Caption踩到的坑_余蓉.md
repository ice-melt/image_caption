# Image Caption踩到的坑

## 1、num_examples_per_epoch（configuration.py）

此参数的含义是每个epoch中examples的数量，即训练数据集中图片数量。

原代码使用的数据集是MSCOCO，训练数据集数量为117215，而原代码此参数设为586363，约为训练数据数量的5倍。

此参数的主要作用是用来计算学习速率衰减。



## 2、inception_v4(image_embedding.py)

原代码在建立CNN模型时，调用image_embedding.py的inception_v3()接口实现。

我们希望将inception_v3替换为inception_v4，故在image_embedding.py代码中仿照inception_v3()实现inception_v4()接口。

**出现错误：**

```python
Traceback (most recent call last):
  File "./train.py", line 25, in <module>
    from im2txt import show_and_tell_model
  File "C:\Users\YuRong\AI实战\Image Caption看图说话机器人\Code\im2txt\im2txt\show_and_tell_model.py", line 29, in <module>
    from im2txt.ops import image_embedding
  File "C:\Users\YuRong\AI实战\Image Caption看图说话机器人\Code\im2txt\im2txt\ops\image_embedding.py", line 27, in <module>
    from tensorflow.contrib.slim.python.slim.nets.inception_v4 import inception_v4_base
ModuleNotFoundError: No module named 'tensorflow.contrib.slim.python.slim.nets.inception_v4'
```

**原因：**

查看python的site-packages下slim的nets里，没有inception_v4，故报此错误。

**解决方案：**

将slim下的nets和preprocessing文件夹拷贝到代码所在路径下，通过代码

```python
from nets import inception
```

导入inception包，而inception.py中，会通过其代码

```python
from nets.inception_v4 import inception_v4
from nets.inception_v4 import inception_v4_arg_scope
from nets.inception_v4 import inception_v4_base
```

导入inception_v4_base。

## 3、进入eval阶段后，循环进行eval问题

模型运行日志如下：

```python
################    eval    ################
INFO:tensorflow:Creating eval directory: /output/eval
INFO:tensorflow:Prefetching values from 1 files matching /data/fandichao1998/flickr8k/val-00000-of-00001
INFO:tensorflow:Starting evaluation at 2018-11-10-19:38:39
2018-11-10 19:38:40.003090: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-11-10 19:38:40.003584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:00:04.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2018-11-10 19:38:40.003661: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-11-10 19:38:40.324242: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-11-10 19:38:40.324356: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 
2018-11-10 19:38:40.324374: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N 
2018-11-10 19:38:40.324744: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10750 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
INFO:tensorflow:Loading model from checkpoint: /output/ckpt/model.ckpt-5049
INFO:tensorflow:Restoring parameters from /output/ckpt/model.ckpt-5049
INFO:tensorflow:Successfully loaded model.ckpt-5049 at global step = 5049.
INFO:tensorflow:Computed losses for 1 of 32 batches.
INFO:tensorflow:Perplexity = 17.806959 (13 sec)
INFO:tensorflow:Finished processing evaluation at global step 5049.
INFO:tensorflow:Starting evaluation at 2018-11-10-19:48:39
2018-11-10 19:48:39.924730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-11-10 19:48:39.924860: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-11-10 19:48:39.924902: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 
2018-11-10 19:48:39.924918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N 
2018-11-10 19:48:39.925054: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10750 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
INFO:tensorflow:Loading model from checkpoint: /output/ckpt/model.ckpt-5049
INFO:tensorflow:Restoring parameters from /output/ckpt/model.ckpt-5049
INFO:tensorflow:Successfully loaded model.ckpt-5049 at global step = 5049.
INFO:tensorflow:Computed losses for 1 of 32 batches.
INFO:tensorflow:Perplexity = 17.800203 (13 sec)
INFO:tensorflow:Finished processing evaluation at global step 5049.
INFO:tensorflow:Starting evaluation at 2018-11-10-19:58:40
2018-11-10 19:58:40.014333: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-11-10 19:58:40.014438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-11-10 19:58:40.014500: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0 
2018-11-10 19:58:40.014563: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N 
2018-11-10 19:58:40.014802: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10750 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
INFO:tensorflow:Loading model from checkpoint: /output/ckpt/model.ckpt-5049
INFO:tensorflow:Restoring parameters from /output/ckpt/model.ckpt-5049
INFO:tensorflow:Successfully loaded model.ckpt-5049 at global step = 5049.
INFO:tensorflow:Computed losses for 1 of 32 batches.
INFO:tensorflow:Perplexity = 17.800204 (13 sec)
INFO:tensorflow:Finished processing evaluation at global step 5049.
```

**原因：**

evaluate.py的run()函数中，实际运行eval过程的代码的写在一个while ture死循环中，目的是令eval过程每间隔指定时间后evaluate一次。

```python
while True:
	start = time.time()
	tf.logging.info("Starting evaluation at " + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
	run_once(model, saver, summary_writer, summary_op)
	time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
	if time_to_next_eval > 0:
		time.sleep(time_to_next_eval)
```

**解决方案：**

代码改为，若命令行输入参数eval_interval_secs为0，则只evaluate一次。

```python
eval_once = (FLAGS.eval_interval_secs == 0)
# Run a new evaluation run every eval_interval_secs.
while True:
	start = time.time()
	tf.logging.info("Starting evaluation at " + time.strftime(
		"%Y-%m-%d-%H:%M:%S", time.localtime()))
	run_once(model, saver, summary_writer, summary_op)
	if eval_once:
		break
	time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
	if time_to_next_eval > 0:
		time.sleep(time_to_next_eval)
```

## 4、InceptionV4的checkpoint加载失败

**错误信息：**

```python
INFO:tensorflow:Restoring parameters from E:\01人工智能学习\0数据\output\train\model.ckpt-0
2018-11-12 10:23:55.239769: W T:\src\github\tensorflow\tensorflow\core\framework\op_kernel.cc:1275] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key InceptionV4/Conv2d_1a_3x3/BatchNorm/beta not found in checkpoint
INFO:tensorflow:Error reported to Coordinator: <class 'tensorflow.python.framework.errors_impl.NotFoundError'>, Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key InceptionV4/Conv2d_1a_3x3/BatchNorm/beta not found in checkpoint
	 [[Node: save_1/RestoreV2 = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save_1/Const_0_0, save_1/RestoreV2/tensor_names, save_1/RestoreV2/shape_and_slices)]]
	 [[Node: save_1/RestoreV2/_231 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device_incarnation=1, tensor_name="edge_312_save_1/RestoreV2", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"]()]]

Caused by op 'save_1/RestoreV2', defined at:
  File "./train.py", line 122, in <module>
    tf.app.run()
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\platform\app.py", line 125, in run
    _sys.exit(main(argv))
  File "./train.py", line 107, in main
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\training\saver.py", line 1281, in __init__
    self.build()
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\training\saver.py", line 1293, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\training\saver.py", line 1330, in _build
    build_save=build_save, build_restore=build_restore)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\training\saver.py", line 778, in _build_internal
    restore_sequentially, reshape)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\training\saver.py", line 397, in _AddRestoreOps
    restore_sequentially)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\training\saver.py", line 829, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\ops\gen_io_ops.py", line 1546, in restore_v2
    shape_and_slices=shape_and_slices, dtypes=dtypes, name=name)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\util\deprecation.py", line 454, in new_func
    return func(*args, **kwargs)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py", line 3155, in create_op
    op_def=op_def)
  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py", line 1717, in __init__
    self._traceback = tf_stack.extract_stack()

NotFoundError (see above for traceback): Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key InceptionV4/Conv2d_1a_3x3/BatchNorm/beta not found in checkpoint
	 [[Node: save_1/RestoreV2 = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save_1/Const_0_0, save_1/RestoreV2/tensor_names, save_1/RestoreV2/shape_and_slices)]]
	 [[Node: save_1/RestoreV2/_231 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device_incarnation=1, tensor_name="edge_312_save_1/RestoreV2", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"]()]]
```

初步判断为代码建立的图和checkpoint中的不匹配，因此初始化数据加载不进去。

在本地加载ckpt报错，在tinymind上可以正常加载和运行。

**原因：**

本地环境之前运行过InceptionV3模型，在TrainDir中自动保存了CheckPoint，因此，在我运行代码时，创建的是InceptionV4的网络结构，而加载的是TrainDir最新训练得到的InceptionV3的CheckPoint，因此，加载失败。

解决方案：

清空TrainDir下的CheckPoint。