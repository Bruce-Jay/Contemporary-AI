```
+--requirements.txt
+--run_only_text.sh  任务运行脚本，只运行文本
+--pre_trained_model
|      +--resnet152.pth  预训练模型，需要从pytorch下载
+--test_overfit.py       查看各个种类 label 数据的数量
+--run.py      主程序
+--generate_data.py     数据处理，从 txt 生成 tsv
+--predict
|      +--test_without_label_legacy.txt   利用原始数据生成的预测数据
|      +--test_without_label.txt    利用增补过标签并且翻译的数据生成的预测数据
+--fix_overfit.py      对于较少的标签进行增补
+--output
|      +--pytorch_model.bin    保存的最好模型
|      +--pytorch_encoder.bin    保存的最好模型
+--run_test.sh   任务运行脚本，生成预测文件
+--data_txt
|      +--valid_legacy.tsv    处理后的原始验证集数据
|      +--valid.tsv    补过标签并且翻译的验证集数据
|      +--train.tsv    补过标签并且翻译的训练集数据
|      +--test_without_label.txt   需要预测数据的 guid 和对应的空情感标签
|      +--test.tsv       处理后的测试集数据
|      +--train_legacy.tsv    处理后的原始训练集数据
|      +--train.txt   数据的 guid 和对应的情感标签
+--models
|      +--model.py    定义了 bert model 的各个部分
|      +--resnet.py    定义了 resnet 如何进行图像转换
+--data 	里面存放单个的文本和图像数据，数目过多不在此展示
|      +...txt
|      +...jpg
+--README.md
+--run.sh    任务运行脚本，运行文本 + 图像
+--run_only_image.sh   任务运行脚本，只运行图像
+--file_tree.py   生成文件树
+--processors
|      +--util.py     处理如何对数据进行训练和验证
```

运行模型需要先下载预训练模型 ResNet-152，并放入 pre_trained_model 文件下

链接 (https://download.pytorch.org/models/resnet152-b121ed2d.pth)

以下命令默认对于未增补过标签并且翻译的数据，输入模型并且运行。

```
sh run.sh #文本+图像
```

```
sh run_text_only.sh #仅文本
```

```
sh run_image_only.sh #仅图像
```

```
sh run_test.sh #输出预测结果
```

如果需要运行增补过标签并且翻译的数据，需要修改以上四个文件：

去除文件末尾的 `--train_file train_legacy.tsv --valid_file valid_legacy.tsv` 参数。

