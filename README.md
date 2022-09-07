# Exploring developments of the AI field from the perspective of methods, datasets, and metrics
This is the code for the paper entitled 'Exploring developments of AI field from the perspective of methods, datasets, and metrics.'

## Code details

**Requirements:**  
>Python=3.6, Tensorflow=1.10.0, numpy==1.15.4, and pandas==0.20.3

**Configurations:** 
>Running Mode

        [`train`/`test`/`interactive_predict`/`api_service`]

>Labeling Scheme: 

        [`BIO`/`BIESO`]

>Model Configuration:

        encoder: CNN/BiLSTM
        decoder: crf/softmax
        embedding level: char/word
        with/without self attention
        hyperparameters

>Training Settings: 

        Metrics: [precision,recall,f1,accuracy]
        Optimizer: GD/Adagrad/AdaDelta/RMSprop/Adam

    
>See more in [HandBook](HandBook.md).

**Train and Test:**  
>1. When training the model, you need to download [SciBERT](https://github.com/allenai/scibert), [BERT](https://github.com/google-research/bert), or [GloVe](https://nlp.stanford.edu/projects/glove/) if you want to use the pretrained vectors.

>2. Module Structure

        ├── main.py
        ├── system.config
        ├── HandBook.md
        ├── README.md
        ├── engines
            ├── Model_Structure.py
           ├── Configer.py
           ├── DataManager.py
           └── utils.py
    
>3. Run the following command to train or evaluate the model.

        Step 1. Train the model
            ├── Set mode=train in 'system.config'
            ├── python main.py

        Step 2. Test the model
            ├── Set mode=test and is_real_test=False in 'system.config'
            ├── python main.py

        Step 3. Extract entities using the trained model
            ├── Set mode=test and is_real_test=True in 'system.config'
            ├── python main.py
