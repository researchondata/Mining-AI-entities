# Exploring developments of the AI field from the perspective of methods, datasets, and metrics
This is the code for the paper entitled 'Exploring developments of AI field from the perspective of methods, datasets, and metrics'.

## Code details

**Requirements:**  
>Python=3.6, Tensorflow=1.10.0, numpy==1.15.4, and pandas==0.20.3

## Configurations
    - Running Mode: [`train`/`test`/`interactive_predict`/`api_service`]
    - Datasets(Input/Output): 
    - Labeling Scheme: 
        - [`BIO`/`BIESO`]
        - [`PER`|`LOC`|`ORG`]
        - ...
    - Model Configuration: 
        - encoder: BGU/Bi-LSTM, layer, Bi/Uni-directional
        - decoder: crf/softmax
        - embedding level: char/word
        - with/without self attention
        - hyperparameters
    - Training Settings: 
        - subscribe measuring metrics: [precision,recall,f1,accuracy]
        - optimazers: GD/Adagrad/AdaDelta/RMSprop/Adam
    - Testing Settings
    - Api service Settings
    
see more in [HandBook](HandBook.md).

## Module Structure
```

├── main.py
├── system.config
├── HandBook.md
├── README.md
│
├── checkpoints
│   ├── BILSTM-CRFs-datasets1
│   │   ├── checkpoint
│   │   └── ...
│   └── ...
├── data
│   ├── example_datasets6
│   │   ├── logs
│   │   ├── vocabs
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── dev.csv
│   └── ...
├── demo_webapp
│   ├── demo_webapp
│   ├── interface
│   └── manage.py
├── engines
       ├── BiLSTM_CRFs.py
     ├── Configer.py
     ├── DataManager.py
     └── utils.py

```

- Folds
    - in `engines` fold, 提供模型架构、数据处理、参数设置等程序.
    - in `data` fold, logs文件夹记录模型训练每轮数据，vocabs文件夹存放字典信息.
    - in `checkpoints` fold, 存放最终训练好的模型.
    
- Files
    - `main.py` 是主程序，无论是训练还是测试，只需要运行main.py即可.
    - `system.config` 是模型参数设置文件，按照自己的需求设置参数，同时，模型选择“训练”和“测试”也是在该程序上修改控制.
    - `BiLSTM_CRFs.py` 是模型整体结构程序，包括网络搭建，模型训练测试设置，以及AI实体归一化自动化处理.
    - `Configer.py` 是将`system.config`文件中设置的参数写入的程序.
    - `DataManager.py` 数据预处理程序，包括将训练集、验证集和测试集转化成模型可直接输入的数据，以及构建相关词典，设置词向量获取方式等.
 
## Quick Start

Under following steps:

#### step 1. 将相关模型参数设置在 `system.config`文件中.

- 设置数据输入输出位置和名称.
- 设置标注模式[BIO||BIEO].
- 设置模型运行状态[train||test].
- 按照注释设置模型网络相关参数[编码器解码器选择等].

#### step 2. 训练模型

- 在`system.config`中将mode=train.
- 在`system.config`中设置与模型训练相关参数.
- 运行 `main.py`.

#### step 3. 测试模型

- 在`system.config`中将mode=test和is_real_test=False.
- 在`system.config`中设置与模型测试相关参数.
- 运行 `main.py`.
    
#### step 4. 模型抽取大规模AI文献中AI实体

- 在`system.config`中将mode=test和is_real_test=True.
- 在`system.config`中设置最终数据存储位置.
- 运行 `main.py`.
