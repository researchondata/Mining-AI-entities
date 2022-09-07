
import os, logging
import numpy as np
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
import re
import xlrd, xlwt
from engines.utils import read_csv_
import jieba
jieba.setLogLevel(logging.INFO)
from itertools import chain

def getTestDataFromExcel(filepath):
    nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05')
    text_para_data = xlrd.open_workbook(filepath, 'rb').sheet_by_name(u'Sheet1')
    text_metrics_list = xlrd.open_workbook('metrics_list.xls', 'rb').sheet_by_name(u'sheet1')
    para_row = text_para_data.nrows
    input_sens = text_para_data.col_values(6)
    metrics_list = text_metrics_list.col_values(0)

    """by stanfordcorenlp to get sentences tokenizes and lables"""
    num_input_sens = len(input_sens)
    print("the length of data is : ", num_input_sens)
    sens_data = []
    sens_pos_data = []
    metrics_entities = []
    for row in range(num_input_sens):
        sentence = input_sens[row].replace('“', '"').replace('”', '"').replace('’', '\'').replace('‘','\'').replace(
        '！', '!').replace('\t', ' ')
        sentence = re.sub("\[.*[0-9]{4}\s*\]", " ", sentence)
        metrics_entity=[metrics_val for metrics_val in metrics_list if (' ' not in metrics_val and ' '+metrics_val+' ' in sentence) or (' ' in metrics_val and ' '+metrics_val+' ' in sentence.lower())]
        metrics_entities.append(metrics_entity)
        pos_sentence = nlp.pos_tag(sentence)
        ner_sentence = nlp.ner(sentence)
        sentence = [tp[0] for tp in pos_sentence]
        sentence_pos = [tp[1] for tp in pos_sentence]
        sentence_ner = [tp[1] for tp in ner_sentence]
        print(row)
        sen_data = []
        sen_pos_data = []
        for i in range(len(sentence)):
            if ' ' not in sentence[i] and sentence[i].lower() != 'n/a' and sentence[i].lower() != 'null' and sentence[
                i].lower() != 'nan' and sentence[i].lower() != 'na' and sentence_ner[i] != 'PERSON':
                sen_data.append(sentence[i])
                sen_pos_data.append(sentence_pos[i])
        sens_data.append(sen_data)
        sens_pos_data.append(sen_pos_data)

    nlp.close()
    print("the sens_length is: ", len(sens_data))
    return sens_data,sens_pos_data,metrics_entities



class DataManager:
    def __init__(self, configs, logger):
        self.configs=configs
        self.train_file = configs.train_file
        self.logger = logger
        self.hyphen = configs.hyphen

        self.UNKNOWN = "<UNK>"
        self.PADDING = "<PAD>"

        self.train_file = configs.datasets_fold + "/" + configs.train_file
        self.dev_file = configs.datasets_fold + "/" + configs.dev_file
        self.test_file = configs.datasets_fold + "/" + configs.test_file
        self.is_real_test = configs.is_real_test
        self.raw_real_sens_file = configs.raw_real_sens_file

        self.output_test_file = configs.datasets_fold + "/" + configs.output_test_file
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = configs.datasets_fold + "/" + configs.output_sentence_entity_file

        self.label_scheme = configs.label_scheme
        self.label_level = configs.label_level
        self.suffix = configs.suffix
        self.labeling_level = configs.labeling_level

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim
        self.max_char_len = configs.max_char_len

        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = self.vocabs_dir + "/token2id"
        self.label2id_file = self.vocabs_dir + "/label2id"
        self.char2id_file = self.vocabs_dir + "/char2id"

        self.token2id, self.id2token, self.label2id, self.id2label, self.char2id, self.id2char = self.loadVocab()

        self.max_token_number = len(self.token2id)
        self.max_char_num = len(self.char2id)
        self.max_label_number = len(self.label2id)

        jieba.load_userdict(self.token2id.keys())

        self.logger.info("dataManager initialed...\n")

    def loadVocab(self):
        if not os.path.isfile(self.token2id_file):
            self.logger.info("vocab files not exist, building vocab...")
            return self.buildVocab(self.train_file)

        self.logger.info("loading vocab...")
        token2id = {}
        id2token = {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id = {}
        id2label = {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                label = row.split('\t')[0]
                label_id = int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label

        char2id = {}
        id2char = {}
        with open(self.char2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                char = row.split('\t')[0]
                char_id = int(row.split('\t')[1])
                char2id[char] = char_id
                id2char[char_id] = char

        return token2id, id2token, label2id, id2label, char2id, id2char

    def buildVocab(self, train_path):
        df_train = read_csv_(train_path, names=["token", "label"],delimiter=self.configs.delimiter)
        tokens = list(set(df_train["token"][df_train["token"].notnull()]))
        labels = list(set(df_train["label"][df_train["label"].notnull()]))
        char_tokens = list(chain(*[[c for c in str] for str in tokens]))
        char_dic = []
        for c in char_tokens:
            if c not in char_dic:
                char_dic.append(c)

        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        char2id = dict(zip(char_dic,range(1, len(char_dic) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        id2char = dict(zip(range(1,len(char_dic) + 1), char_dic))
        id2token[0] = self.PADDING
        id2label[0] = self.PADDING
        id2char[0] = self.PADDING
        token2id[self.PADDING] = 0
        label2id[self.PADDING] = 0
        char2id[self.PADDING] = 0
        id2token[len(tokens) + 1] = self.UNKNOWN
        id2char[len(char_dic) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1
        char2id[self.UNKNOWN] = len(char_dic) + 1

        self.saveVocab(id2token, id2label, id2char)

        return token2id, id2token, label2id, id2label, char2id, id2char

    def saveVocab(self, id2token, id2label, id2char):
        with open(self.token2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + "\t" + str(idx) + "\n")
        with open(self.label2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + "\t" + str(idx) + "\n")
        with open(self.char2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2char:
                outfile.write(id2char[idx] + "\t" + str(idx) + "\n")

    def getEmbedding(self, embed_file):
        print("begin to embedding")
        emb_matrix = np.random.normal(loc=0.0, scale=0.08, size=(len(self.token2id.keys()), self.embedding_dim))
        emb_matrix[self.token2id[self.PADDING], :] = np.zeros(shape=(self.embedding_dim))

        with open(embed_file, "r", encoding="utf-8") as infile:
            for row in infile:
                row = row.rstrip()
                items = row.split()
                token = items[0]
                assert self.embedding_dim == len(
                    items[1:]), "embedding dim must be consistent with the one in `token_emb_dir'."
                emb_vec = np.array([float(val) for val in items[1:]])
                if token in self.token2id.keys():
                    emb_matrix[self.token2id[token], :] = emb_vec
        print("the shape of embedding is:",emb_matrix.shape)

        return emb_matrix

    def nextTestBatch(self, X,X_pos, y,X_str, X_char,start_index,is_less_batch=False):
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        X_pos_batch = list(X_pos[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        X_str_batch = list(X_str[start_index:min(last_index, len(X))])
        X_char_batch = list(X_char[start_index:min(last_index, len(X))])
        if last_index > len(X):
            is_less_batch = True
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                X_pos_batch.append(X_pos[index])
                y_batch.append(y[index])
                X_str_batch.append(X_str[index])
                X_char_batch.append(X_char[index])
        X_batch = np.array(X_batch)
        X_pos_batch = np.array(X_pos_batch)
        y_batch = np.array(y_batch)
        X_str_batch = np.array(X_str_batch)
        X_char_batch = np.array(X_char_batch)
        return X_batch,X_pos_batch, y_batch,X_str_batch,X_char_batch,is_less_batch

    def nextBatch(self, X, y,X_str, start_index):
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        X_str_batch = list(X_str[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
                X_str_batch.append(X_str[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        X_str_batch = np.array(X_str_batch)
        return X_batch, y_batch, X_str_batch

    def nextRandomBatch(self, X, y):
        X_batch = []
        y_batch = []
        for i in range(self.batch_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def padding(self, sample):
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def padding_char(self,char_sample):
        for l in range(len(char_sample)):
            if len(char_sample[l]) < self.max_sequence_length:
                char_sample[l] += (self.max_sequence_length-len(char_sample[l]))*[self.max_char_len*[self.char2id[self.PADDING]]]
        return char_sample

    def prepare(self, tokens, labels, is_padding=True,is_char=True, return_psyduo_label=False):
        X = []
        y = []
        y_psyduo = []
        tmp_x = []
        tmp_y = []
        tmp_y_psyduo = []
        X_char = []
        temp_X_char = []
        size_char_all = []
        tem_size_char = []
        for record in zip(tokens, labels):
            c = record[0]
            l = record[1]
            if is_char:
                if c != -1:
                    x_c = [self.mapFunc(s, self.char2id) for s in c]
                    size_char = len(c)
                    if len(x_c) > self.max_char_len:
                        x_c = x_c[0:self.max_char_len]
                        size_char = self.max_char_len
                    x_c += [self.char2id[self.PADDING] for _ in range(self.max_char_len - len(x_c))]
            if c == -1:
                if len(tmp_x) > self.max_sequence_length:
                    tmp_x = tmp_x[0:self.max_sequence_length]
                    tmp_y = tmp_y[0:self.max_sequence_length]
                    if is_char:
                        tem_size_char = tem_size_char[0:self.max_sequence_length]
                        temp_X_char = temp_X_char[0:self.max_sequence_length]

                X.append(tmp_x)
                y.append(tmp_y)
                X_char.append(temp_X_char)
                size_char_all.append(tem_size_char)
                if return_psyduo_label: y_psyduo.append(tmp_y_psyduo)
                tmp_x = []
                tmp_y = []
                temp_X_char = []
                tem_size_char = []
                if return_psyduo_label: tmp_y_psyduo = []
            else:
                tmp_x.append(c)
                tmp_y.append(l)
                if is_char:
                    tem_size_char.append(size_char)
                    temp_X_char.append(x_c)
                if return_psyduo_label: tmp_y_psyduo.append(self.label2id["O"])
        if is_padding:
            X = np.array(self.padding(X))
            if is_char:
                X_char = np.array(self.padding_char(X_char))
                size_char_all = np.array(self.padding(size_char_all))
                print("the shape of char is :", X_char.shape, size_char_all.shape)
        y = np.array(self.padding(y))

        if return_psyduo_label:
            y_psyduo = np.array(self.padding(y_psyduo))
            return X, y_psyduo

        return X,y,X_char

    def prepareForRealTest(self, tokens,tokens_id,tokens_pos, is_padding=True,is_char=True, return_psyduo_label=True):
        X_char = []
        y_psyduo = []
        for i in range(len(tokens)):
            if len(tokens[i]) > self.max_sequence_length:
                tokens[i] = tokens[i][0:self.max_sequence_length]
                tokens_id[i] = tokens_id[i][0:self.max_sequence_length]
                tokens_pos[i] = tokens_pos[i][0:self.max_sequence_length]
            X_char_one = []
            y_psyduo_one = []
            if is_char:
                for j in range(len(tokens[i])):
                    c = tokens[i][j]
                    if return_psyduo_label:
                        y_psyduo_one.append(self.label2id['O'])
                    x_c = [self.mapFunc(s, self.char2id) for s in c]
                    if len(x_c) > self.max_char_len:
                        x_c = x_c[0:self.max_char_len]
                    x_c += [self.char2id[self.PADDING] for _ in range(self.max_char_len - len(x_c))]
                    X_char_one.append(x_c)
            X_char.append(X_char_one)
            y_psyduo.append(y_psyduo_one)
        if is_padding:
            X_str = np.array(self.padding(tokens))
            X_pos = np.array(self.padding(tokens_pos))
            X_id = np.array(self.padding(tokens_id))
            if is_char:
                X_char = np.array(self.padding_char(X_char))
        if return_psyduo_label:
            y_psyduo = np.array(self.padding(y_psyduo))
        print("the shape of test_token_id is:",X_id.shape,X_id[0])
        print("the shape of test_label_id is:",y_psyduo.shape,y_psyduo[0])
        print("the shape of test_token is:",X_str.shape,X_str[0])
        print("the shape of test_token_char is:",X_char.shape,X_char[0])
        return X_id,X_pos,y_psyduo,X_str,X_char

    def getTrainingSet(self, train_val_ratio=0.9):
        print("begin to load data")
        df_train = read_csv_(self.train_file, names=["token", "label"],delimiter=self.configs.delimiter)

        # map the token and label into id
        df_train["token_id"] = df_train.token.map(lambda x: -1 if str(x) == str(np.nan) else self.token2id[x])
        df_train["token"] = df_train.token.map(lambda x: -1 if str(x) == str(np.nan) else x)
        df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])
        # print(df_train["token"])

        # convert the data in maxtrix
        X,y,_ = self.prepare(df_train["token_id"],df_train["label_id"],is_char=False)
        _,_,X_char = self.prepare(df_train["token"],df_train["label_id"],is_char=True)

        # shuffle the samples
        num_samples = len(X)
        indexs = np.arange(num_samples)
        np.random.shuffle(indexs)
        X = X[indexs]
        y = y[indexs]
        X_char = X_char[indexs]

        if self.dev_file != None:
            X_train = X
            y_train = y
            X_train_char = X_char
            X_val, y_val, X_val_char = self.getValidingSet()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]

        self.logger.info("\ntraining set size: %d, validating set size: %d\n" % (len(X_train), len(y_val)))
        print("the input of x_id is:",X_train[0])
        print("the input of y_id is : ", y_train[0])
        print("the input of x_char_id is: ", X_train_char[0])

        return X_train, y_train,X_train_char, X_val, y_val,X_val_char

    def getValidingSet(self):
        df_val = read_csv_(self.dev_file, names=["token","label"],delimiter=self.configs.delimiter)

        df_val["token_id"] = df_val.token.map(lambda x: self.mapFunc(x, self.token2id))
        df_val["token"] = df_val.token.map(lambda x: -1 if str(x) == str(np.nan) else x)
        df_val["label_id"] = df_val.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])

        X_val, y_val,_ = self.prepare(df_val["token_id"], df_val["label_id"],is_char=False)
        _,_, X_val_char = self.prepare(df_val["token"], df_val["label_id"],is_char=True)
        return X_val, y_val,X_val_char

    def getTestingSet(self):
        if self.is_real_test:
            test_token,test_token_pos,rule_metrics_entities = getTestDataFromExcel(self.raw_real_sens_file)
            self.logger.info("\ntesting set size: %d\n" % (len(test_token)))
            test_token_id = [[self.mapFuncToken(val,self.token2id) for val in val_list] for val_list in test_token]
            X_test_id,X_test_pos, y_test_psyduo_label, X_test_token, X_test_char = self.prepareForRealTest(test_token,test_token_id,test_token_pos)
            return X_test_id,X_test_pos,y_test_psyduo_label,X_test_token,X_test_char,rule_metrics_entities
        else:
            df_test = read_csv_(self.test_file, names=None, delimiter=self.configs.delimiter)
            if len(list(df_test.columns)) == 2:
                df_test.columns = ["token", "label"]
                df_test["token_id"] = df_test.token.map(lambda x: self.mapFunc(x, self.token2id))
                df_test["token"] = df_test.token.map(lambda x: -1 if str(x) == str(np.nan) else x)
                df_test["label_id"] = df_test.label.map(lambda x: -1 if str(x) == str(np.nan) else self.label2id[x])

                X_test_id, y_test_psyduo_label, _ = self.prepare(df_test["token_id"], df_test["label_id"],
                                                                 is_char=False)
                X_test_token, _, X_test_char = self.prepare(df_test["token"], df_test["label_id"], is_char=True)

                self.logger.info("\ntesting set size: %d\n" % (len(X_test_id)))
                return X_test_id, y_test_psyduo_label, X_test_token, X_test_char

    def mapFuncToken(self,x,token2id):
        if x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def mapFunc(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def mapFuncLable(self, x, label2id):
        if str(x) == str(np.nan):
            return -1
        else:
            return label2id[x]

    def prepare_single_sentence(self, sentence):
        if self.labeling_level == 'word':
            if self.check_contain_chinese(sentence):
                sentence = list(jieba.cut(sentence))
            else:
                sentence = list(sentence.split())
        elif self.labeling_level == 'char':
            sentence = list(sentence)

        gap = self.batch_size - 1

        x_ = []
        y_ = []

        for token in sentence:
            try:
                x_.append(self.token2id[token])
            except:
                x_.append(self.token2id[self.UNKNOWN])
            y_.append(self.label2id["O"])

        if len(x_) < self.max_sequence_length:
            sentence += ['x' for _ in range(self.max_sequence_length - len(sentence))]
            x_ += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(x_))]
            y_ += [self.label2id["O"] for _ in range(self.max_sequence_length - len(y_))]
        elif len(x_) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]
            x_ = x_[:self.max_sequence_length]
            y_ = y_[:self.max_sequence_length]

        X = [x_]
        Sentence = [sentence]
        Y = [y_]
        X += [[0 for j in range(self.max_sequence_length)] for i in range(gap)]
        Sentence += [['x' for j in range(self.max_sequence_length)] for i in range(gap)]
        Y += [[self.label2id['O'] for j in range(self.max_sequence_length)] for i in range(gap)]
        X = np.array(X)
        Sentence = np.array(Sentence)
        Y = np.array(Y)

        return X, Sentence, Y

    def check_contain_chinese(self, check_str):
        for ch in list(check_str):
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

