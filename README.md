
## 项目操作说明
   
    step1：
        使用labelimg标注数据为VOC格式；

####    
    step2:
        >>2.1 安装依赖环境，将requirements.txt的库安装在docker中，我这边的onnx库没有安装官方规定的版本（因为会报错）；我认为安装了requirements.txt就没有必要再执行setup.py了；
        >>2.2 安装apex库：这个库是用于混合精度训练，达到提升GPU上训练速度的目的；
        安装参考链接：
        https://blog.csdn.net/ccbrid/article/details/103207676
        【Github】https://github.com/NVIDIA/apex

        关于step2的说明：可以直接用做好的docker进行训练，做好的docker中没有安装apex（因为版本不匹配报错了），所以在训练时候不要设置混合精度的参数。

        执行下面代码测试环境是否配置好：
        python3 tools/demo.py image -n yolox-s -c weights/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [gpu]

        正常运行且在YOLOX_outputs文件夹中保存了测试的图片。
####    
    step3:
        对step1中标注好的VOC格式的数据进行处理工作,其目录结构如下所示：

            |-- Annotations
            |-- ImageSets
                `-- Main
                    |-- train.txt
                    `-- val.txt
            |-- JPEGImages
            |-- annotations_cache
            |-- results
            `-- train_val_data_split.py

        其中，Annotations存放的是标注所得的xml标签数据；JPEGImages存放的是图片数据；ImageSets/Main文件夹下存放了训练和验证集的txt文件（ImageSets/Main需要手动建立）；annotations_cache和results都是训练所得。
        由于自己标注的数据都放在一个文件夹里，没有区分训练集和验证集，所以需要执行train_val_data_split.py脚本来划分数据集，并生成tarin.txt和val.txt两个文件。那么train_val_data_split.py文件如下所示(我是按照1：9划分的，可以修改)：

```python3
import os
import random

images_path = "JPEGImages/"
xmls_path = "Annotations/"
train_val_txt_path = "ImageSets/Main/"
val_percent = 0.1

images_list = os.listdir(images_path)
random.shuffle(images_list)

#　划分训练集和验证集的数量
train_images_count = int((1-val_percent)*len(images_list))
val_images_count = int(val_percent*len(images_list))

#　生成训练集的train.txt文件
train_txt = open(os.path.join(train_val_txt_path,"train.txt"),"w")
train_count = 0
for i in range(train_images_count):
    text = images_list[i].split(".jpg")[0] + "\n"
    train_txt.write(text)
    train_count+=1
    print("train_count: " + str(train_count))
train_txt.close()

#　生成验证集的val.txt文件
val_txt = open(os.path.join(train_val_txt_path,"val.txt"),"w")
val_count = 0
for i in range(val_images_count):
    text = images_list[train_images_count + i].split(".jpg")[0] + "\n"
    val_txt.write(text)
    val_count+=1
    print("val_count: " + str(val_count))
val_txt.close()
```

    step4:
        修改训练工程代码(以人头检测单目标为例)
        >>4.1 修改yolox/data/datasets/voc_classes.py中的标签信息：

```python3
        #修改标签信息，类别后都要加一个逗号
        VOC_CLASSES = (
            "head",
        )
```
        >>4.2 修改exps/example/yolox_voc/yolox_voc_s.py中的类别数量：
```python3
        class Exp(MyExp):
            def __init__(self):
                super(Exp, self).__init__()
                self.num_classes = 1           #修改类别数量，voc原始为20
                self.depth = 0.33
                self.width = 0.50
                self.warmup_epochs = 1
```
        >>4.3 修改yolox/exp/yolox_base.py中的类别数量：
```python3
        class Exp(BaseExp):
            def __init__(self):
                super().__init__()

                # ---------------- model config ---------------- #
                # detect classes number of model
                self.num_classes = 1            #修改类别数量，COCO原始为80
```
        >>4.4 修改exps/example/yolox_voc/yolox_voc_s.py中的训练集路径信息和目标个数：
```python3
        with wait_for_the_master(local_rank):
                    dataset = VOCDetection(
                        data_dir="/home/dataset/head_det/",   #训练集所在的绝对路径（docker内）
                        image_sets=[('train')],               #训练集名称
                        img_size=self.input_size,
                        preproc=TrainTransform(
                            max_labels=100,       #表示单图中最多出现的目标个数，修改为100，原始值为50
                            flip_prob=self.flip_prob,
```
        >>4.5 修改yolox/data/datasets/voc.py中的年份信息（删除、自训练数据集永久如下）：
```python3
        #原始的rootpath中带有年份信息，自己标注的数据集中没有年份信息，所以删掉:
                for name in image_sets:
                    rootpath = self.root
                    for line in open(
                        os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
                    ):
                        self.ids.append((rootpath, line.strip()))
```
        >>4.6 修改exps/example/yolox_voc/yolox_voc_s.py中的验证集路径信息：
```python3
        def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
            from yolox.data import VOCDetection, ValTransform

            valdataset = VOCDetection(
                data_dir="/home/dataset/head_det/",        #验证集的绝对路径（docker中）
                image_sets=[('val')],                      #验证集的名字
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy),
            )
```     
        >>4.7 【关于训练yolox-m、yolox-nano等模型需要进行的修改】
        >>4.7.1 比如我要训练yolo-nano，则需要在exps/example/yolox_voc/目录下新建一个yolox_voc_nano.py的文件，先复制yolox_voc_s.py中的内容，然后修改下述代码中的self.depth和self.width，将这两个值修改的同exps/default/yolox_nano.py这两个参数一致即可。
```python3
        class Exp(MyExp):
            def __init__(self):
                super(Exp, self).__init__()
                self.num_classes = 1           #修改类别数量，voc原始为20
                self.depth = 0.33
                self.width = 0.50
                self.warmup_epochs = 1
```
        >>4.7.2 修改yolox/exp/yolox_base.py中的input参数(修改为416)：【检查下是否需要修改】
```python3
        self.input_size = (640, 640)  # (height, width)

        self.test_size = (640, 640)
```
        >>4.7.3 修改yolox/exp/yolox_base.py中的self.depth和self.width，将这两个值修改的同exps/default/yolox_nano.py这两个参数一致即可。
```python3
        class Exp(BaseExp):
            def __init__(self):
                super().__init__()

                # ---------------- model config ---------------- #
                # detect classes number of model
                self.num_classes = 1            #修改类别数量，COCO原始为80
                # factor of model depth
                self.depth = 0.33               #修改深度，原始值为1.00
                # factor of model width
                self.width = 0.5               #修改宽度，原始值为1.00
```
        >>4.8 修改yolox/exp/yolox_base.py中的self.max_epoch、self.print_interval和self.eval_interval三个参数：
```python3
        self.max_epoch = 300       #总的迭代的epoch
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 5        #表示每迭代5次保存一下pth权重，原始值为10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 5        #表示每迭代5次对验证集做一次验证，原始值为10
        # save history checkpoint or not.
```
        >>4.9 修改yolox/data/datasets/voc.py中的_do_python_eval函数：
```python3
        def _do_python_eval(self, output_dir="output", iou=0.5):
                rootpath = self.root                #去掉了rootpath中原有的年份信息
                name = self.image_set[0]            #由于自有数据集中没有年份信息，所以名字不是image_set[0][1]，而是image_set[0]
                annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
                imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
                cachedir = os.path.join(
                    self.root, "annotations_cache"              #去掉cachedir中原有的年份信息
                )
                if not os.path.exists(cachedir):
                    os.makedirs(cachedir)
                aps = []
                # The PASCAL VOC metric changed in 2010
                use_07_metric = True  #改为True，之前是根据年份信息进行判断的
                print("Eval IoU : {:.2f}".format(iou))
```
        >>4.10 修改yolox/data/datasets/voc.py中的_get_voc_results_file_template(self)函数，去除年份信息：
```python3
        def _get_voc_results_file_template(self):
                filename = "comp4_det_test" + "_{:s}.txt"
                filedir = os.path.join(self.root, "results")        #由于自建数据集中没有年份信息，所以将年份信息的参数删除
                if not os.path.exists(filedir):
```

    step5:
        开始训练
        下载预训练权重放置在weights文件夹中，在命令终端执行如下命令开始训练：
        python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 2 -b 32 -c weights/yolox_s.pth

        其中，-d 表示使用的显卡数量，如果GPU服务器只有一张卡，则将devices的default修改为0；
              -b 表示batchsize，根据显存大小而定；
              -c 表示预训练权重

####
    step6:
        训练后测试：(有两种测试手段，即执行demo.py或eval.py文件)：
        >>6.1 执行demo.py测试（测试结果都保存在YOLOX_outputs文件夹下）
        使用demo.py进行测试时，注意修改demo.py里的标签。
        测试视频：python tools/demo.py video -n yolox-s -c YOLOX_outputs/yolox_voc_s_0509/best_ckpt.pth --path assets/head_det.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
            其中，-n 表示模型名字；
                  -c 表示权重位置；
                  --path 表示测试视频；
                  --conf 表示测试的阈值；
                  --nms 表示nms阈值；
                  --tsize 表示测试image大小。
        测试图片：python tools/demo.py image -n yolox-s -c YOLOX_outputs/yolox_voc_s_0509/best_ckpt.pth --path assets/head_det.png --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu

        >>6.2 执行eval.py可以测出map
        执行命令:python tools/eval.py -n yolox-s -f exps/example/yolox_voc/yolox_voc_s.py -d 2 -c YOLOX_outputs/yolox_voc_s_0509/best_ckpt.pth --conf 0.25 --nms 0.45 --tsize 640
            其中，-f 表示模型配置文件，要与训练时一致

####
    step7:
        一些工具
        >>7.1 使用tensorboard可视化loss；
        首先进入YOLOX_outputs目录，然后执行如下命令：
        tensorboard --logdir yolox_voc_s_0509/tensorboard/ --bind_all
        此时，会在vscode终端出现如下内容：
```
        (base) root@24956317f57e:/home/yolox-master/YOLOX_outputs# tensorboard --logdir yolox_voc_s_0509/tensorboard/ --bind_all
        TensorFlow installation not found - running with reduced feature set.

        NOTE: Using experimental fast data loading logic. To disable, pass
            "--load_fast=false" and report issues on GitHub. More details:
            https://github.com/tensorflow/tensorboard/issues/4784

        TensorBoard 2.9.0 at http://24956317f57e:6006/ (Press CTRL+C to quit)
```     
        如上内容中：http://24956317f57e:6006/表示容器24956317f57e中的端口6006，然后在端口那栏添加端口在本地浏览器打开即可，具体参考assets/tensorboard.png。
       
        >>7.2 Resume训练
        如遇特殊情况模型训练中断，需要继续训练，执行如下命令：（没跑过，有待测试可行性）
        python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 2 -b 32 -c OLOX_outputs/yolox_voc_s/epoch_10_ckpt.pth --resume （ -e 10 可去掉）

        >>7.3 转onnx模型
        执行如下命令：
        python tools/export_onnx.py --output-name YOLOX_outputs/yolox_voc_s_0509/head_det.onnx -n yolox-s -c YOLOX_outputs/yolox_voc_s_0509/best_ckpt.pth
        关于如何转不是标准的yolox模型或者如何使用onnx模型进行推理，请参考demo/ONNXRuntime/README.md文档。

####
    参考链接：
    【官方】https://github.com/Megvii-BaseDetection/YOLOX/issues
    【江大白人头检测教程,讲解训练过程中常见问题】https://zhuanlan.zhihu.com/p/397499216
    【一个比较详细的教程，讲yolox-s/l/...等如何替换和测试】https://blog.csdn.net/qq_40716944/article/details/120409457
    【讲yolox-s/l/...等如何替换、windows上训练yolox】https://blog.csdn.net/dear_queen/article/details/120172174
    【转ONNX模型】https://positive.blog.csdn.net/article/details/119915460
    【以脚本的形式对工程进行整理、此作者的其他文章也可学习】https://zhuanlan.zhihu.com/p/402210371



