文件树结构：（关键文件）

```
+--data
|      +--dataloader.py # 把数据使用 DataLoader 定义 batch_size, num_workers 等等训练参数加载训练
|      +--my_dataset.py # 定义如何处理数据使其符合 batch 以及 bos, eos 等等各种 token
|      +--process_data.py # 把数据从 csv 转换为列表的文件
|      +--test.csv
|      +--train.csv
|      +--words_test.csv # sentencepiece 处理过的测试集
|      +--words_train.csv # sentencepiece 处理过的训练集
+--dictionary # 用于sentencepiece生成词表的文件目录（但是因为时间仓促没有成功）
|      +--vocabulary.csv # 一个数字对应一个汉字，用于sentencepiece的生成
|      +--2000kanji.txt # 网上找到的2000汉字常用词
|      +--toVocabulary.py # 生成 csv
|      +--vocabulary.txt # csv 的 txt 版本
+--main.py # 运行，包含训练和测试两大步骤
+--main_str.py # 运行 sentencepiece 生成的词表处理过的数据，但因为后面时间仓促未能实现
+--model # 存放预训练后模型参数的目录
|      +--model
|      |      +--facebook
|      |      |      +--bart-base-0.pth
|      |      |      +--bart-base-1.pth
|      |      |      +--bart-base-2.pth
|      |      |      +--bart-base-3.pth
|      |      |      +--bart-base-4.pth
|      |      |      +--bart-base-9.pth
|      +--small_bart.py # 定义从 Huggingface 库选择哪个模型进行预训练
+--opts.py # 各种超参数的配置文件
+--predict.py # 运行这个代码，生成预测文件
+--report # 历史生成的预测文件，不具有效力
|      +--report_1.csv
|      +--report_2.csv
|      +--report_3.csv
+--report.csv # 最终的预测文件
+--requirements.txt
+--utils
|      +--build_lr_scheduler.py
|      +--build_optimizer.py
|      +--save_model.py # 使用 torch.save 保存模型，并且定义保存的路径

```

如何运行代码？

运行训练和验证步骤：进入文件根目录并且运行

```
python main.py
```

可以指定超参数 --num_epochs, --lr, --batch_size 等等

如果单独运行训练或者验证步骤，需要注释掉相关代码

具体在 main.py 的 59-68 行

```python
    for epoch in range(args.epochs):
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('lr:', cur_lr)
        print('weight decay:', optimizer.state_dict()['param_groups'][0]['weight_decay'])
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, device=device, is_adversial=False, scaler=scaler) # 如果只运行验证，注释这两行
        valid_loss = valid(valid_loader, model, device, epoch, args.num_beams, args.file_valid) # 如果只运行训练，注释这两行

        tb_writer.add_scalar('train loss', train_loss, epoch) # 如果只运行验证，注释这两行
        tb_writer.add_scalar('valid loss', valid_loss, epoch) # 如果只运行训练，注释这两行
        tb_writer.add_scalar('lr', cur_lr, epoch)
```



如何生成预测数据：

进入文件根目录并且运行：

```
python predict.py
```

会在文件根目录下生成 report.csv

最终的预测文件在项目根目录下的 report.csv