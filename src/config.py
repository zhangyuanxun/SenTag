import os
import torch

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT_PATH, "../outputs")
BERT_MODEl = 'bert-base-uncased'
PORT_NUMBER = 8005
MAX_SEQ_LEN = 16
MAX_SENT_LEN = 16
BATCH_SIZE = 8
LABEL_LIST = ['B-f', 'I-f', 'O', 'B-s', 'I-s']
DEVICE = torch.device("cpu")