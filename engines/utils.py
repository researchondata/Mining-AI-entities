import re, logging, datetime, csv
import pandas as pd
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report


def get_logger(log_dir):
    log_file = log_dir + "/" + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d: %H %M %S'))

    return logger


def extractEntity_(sentence,sentence_pos, labels_, reg_str, label_level):
    entitys = []
    entitys_pos = []
    labled_labels = []
    labled_indexs = []
    judge_len = len(labels_)
    seq_len = 0
    judeg_str = 'O'
    if judge_len != 0:
        seq_len = len(sentence)
        judeg_str = labels_[-1]
    labels__ = [('%03d' % (ind)) + lb for lb, ind in zip(labels_, range(len(labels_)))]
    labels = " ".join(labels__)
    re_entity = re.compile(reg_str)
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        if label_level == 1:
            labled_labels.append("_")
        elif label_level == 2:
            labled_labels.append(entity_labels.split()[0][5:])

        start_index = int(entity_labels.split()[0][:3])

        if len(entity_labels.split()) != 1:
            end_index = int(entity_labels.split()[-1][:3]) + 1
        else:
            end_index = start_index + 1
        if judge_len != 0:
            if end_index == seq_len - 1 and judeg_str != 'O':
                end_index += 1

        entity = ' '.join(sentence[start_index:end_index])
        entity_pos = ' '.join(sentence_pos[start_index:end_index])
        labels = labels__[end_index:]
        labels = " ".join(labels)
        entitys.append(entity)
        entitys_pos.append(entity_pos)
        labled_indexs.append((start_index, end_index))
        m = re_entity.search(labels)

    return entitys, labled_labels, labled_indexs,entitys_pos


def extractEntity(x, x_pos, y, dataManager):
    label_scheme = dataManager.label_scheme
    label_level = dataManager.label_level
    label_hyphen = dataManager.hyphen

    if label_scheme == "BIO":
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*'

        elif label_level == 2:
            tag_bodys = ["(" + tag + ")" for tag in dataManager.suffix]
            tag_str = "(" + ('|'.join(tag_bodys)) + ")"
            reg_str = r'([0-9][0-9][0-9]B'+label_hyphen + tag_str + r' )([0-9][0-9][0-9]I'+label_hyphen + tag_str + r' )*'

    elif label_scheme == "BIESO":
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*([0-9][0-9][0-9]E' + r' )|([0-9][0-9][0-9]S' + r' )'

        elif label_level == 2:
            tag_bodys = ["(" + tag + ")" for tag in dataManager.suffix]
            tag_str = "(" + ('|'.join(tag_bodys)) + ")"
            reg_str = r'([0-9][0-9][0-9]B'+label_hyphen + tag_str + r' )([0-9][0-9][0-9]I'+label_hyphen + tag_str + r' )*([0-9][0-9][0-9]E'+label_hyphen + tag_str + r' )|([0-9][0-9][0-9]S'+label_hyphen + tag_str + r' )'

    return extractEntity_(x,x_pos, y, reg_str, label_level)


def metrics(y_true, y_pred, measuring_metrics, dataManager):
    true_labels = []
    pred_labels = []
    count = 0
    for i in range(len(y_true)):
        y = [str(dataManager.id2label[val]) for val in y_true[i] if val != dataManager.label2id[dataManager.PADDING]]
        y_hat = [str(dataManager.id2label[val]) for val in y_pred[i] if val != dataManager.label2id[dataManager.PADDING]]
        if len(y) == len(y_hat):
            count+=1
        true_labels.append(y)
        pred_labels.append(y_hat)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    results = {}
    for measu in measuring_metrics:
        results[measu] = vars()[measu]
    return results

def read_csv_(file_name, names, delimiter='t'):
    if delimiter=='t':
        sep = "\t"
    elif delimiter=='b':
        sep = " "
    else:
        sep = delimiter

    return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                       names=names)


def save_csv_(df_, file_name, names, delimiter='t'):
    if delimiter == 't':
        sep = "\t"
    elif delimiter == 'b':
        sep = " "
    else:
        sep = delimiter

    df_.to_csv(file_name, quoting=csv.QUOTE_NONE,
               columns=names, sep=sep, header=False,
               index=False)
