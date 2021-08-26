import torch

from transformers import (
    AutoTokenizer,
)
from transformers.models.bert.modeling_bert import (
    BertConfig,
)
from transformers import WEIGHTS_NAME
import nltk
from torch.utils.data import DataLoader
from .utils import *
from .models.sentag import SenTag
from config import *
import collections


class InferModel:
    def __init__(self, device, bert_model, label_list, model_dir, max_seq_length, max_sent_length, batch_size):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.config = BertConfig.from_pretrained(bert_model)
        self.max_seq_length = max_seq_length
        self.max_sent_length = max_sent_length
        self.label2id = {k: i for i, k in enumerate(label_list)}
        self.id2label = {i: k for i, k in enumerate(label_list)}
        self.sentag_model = SenTag.from_pretrained(bert_model, config=self.config,
                                                   label_list=label_list, device=self.device)
        self.sentag_model.load_state_dict(torch.load(os.path.join(model_dir, WEIGHTS_NAME), map_location="cpu"))
        self.sentag_model.eval()
        self.sentag_model.to(self.device)
        self.batch_size = batch_size

    def convert_features(self, example):
        sentences = nltk.sent_tokenize(example)
        sentences_tokens = [nltk.word_tokenize(sent) for sent in sentences]

        sentences_input_ids = list()
        sentences_input_mask = list()
        sentences_type_ids = list()
        features = list()

        for sent in sentences_tokens:
            sent_feature = self.tokenizer(sent, is_split_into_words=True, max_length=self.max_seq_length,
                                          padding="max_length", truncation=True)

            sentences_input_ids.append(sent_feature['input_ids'])
            sentences_input_mask.append(sent_feature['attention_mask'])
            sentences_type_ids.append(sent_feature['token_type_ids'])

        if len(sentences_input_ids) % self.max_sent_length != 0:
            empty_sentence = self.tokenizer([], is_split_into_words=True, max_length=self.max_seq_length,
                                       padding="max_length", truncation=True)

            while len(sentences_input_ids) % self.max_sent_length != 0:
                sentences_input_ids.append(empty_sentence['input_ids'])
                sentences_input_mask.append(empty_sentence['attention_mask'])
                sentences_type_ids.append(empty_sentence['token_type_ids'])

        for i in range(0, len(sentences_input_ids), self.max_sent_length):
            features.append({'sentences_input_ids': sentences_input_ids[i: i + self.max_sent_length],
                             'sentences_input_mask': sentences_input_mask[i: i + self.max_sent_length],
                             'sentences_type_ids': sentences_type_ids[i: i + self.max_sent_length],
                             'sentences_input_len': self.max_sent_length})

        def collate_fn(batch):
            def convert_to_tensor(key):
                if isinstance(key, str):
                    tensors = [torch.tensor(o[1][key], dtype=torch.long) for o in batch]
                else:
                    tensors = [torch.tensor(o, dtype=torch.long) for o in key]

                return torch.stack(tensors)

            ret = dict(sentences_input_ids=convert_to_tensor('sentences_input_ids'),
                       sentences_input_mask=convert_to_tensor('sentences_input_mask'),
                       sentences_type_ids=convert_to_tensor('sentences_type_ids'),
                       sentences_input_len=convert_to_tensor('sentences_input_len'))

            return ret

        dataloader = DataLoader(list(enumerate(features)), batch_size=self.batch_size, collate_fn=collate_fn)
        return dataloader

    def predict(self, example):
        dataloader = self.convert_features(example)
        prediction_results = list()
        output_results = list()

        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                output = self.sentag_model(**inputs)
                logits = output['logits']
                pred_labels = self.sentag_model.crf.obtain_labels(logits, self.id2label)

            sentences_input_ids = inputs['sentences_input_ids']

            for i in range(len(pred_labels)):
                original_notes = list()
                results = list()
                predictions = get_entities(pred_labels[i], self.id2label, 'bio')
                predict_notes = collections.defaultdict(list)

                for j in range(len(pred_labels[i])):
                    sentence = self.tokenizer.convert_ids_to_tokens(sentences_input_ids[i][j])
                    sentence = self.tokenizer.convert_tokens_to_string(sentence)
                    sentence = clean_text(sentence)
                    results.append([sentence, 'O'])
                    original_notes.append(sentence)

                for label, start, end in predictions:
                    for k in range(start, end + 1):
                        results[k][1] = label
                    predict_notes[label].append(", ".join(original_notes[start: end + 1]))

                prediction_results.append(results)
                predict_notes = '' if not predict_notes else predict_notes
                output_results.append({"Original Notes": clean_text(" ".join(original_notes)),
                                       "Predict Notes": predict_notes})

        return output_results


model = InferModel(device=DEVICE, bert_model=BERT_MODEl,
                   label_list=LABEL_LIST, model_dir=MODEL_DIR,
                   max_seq_length=MAX_SEQ_LEN, max_sent_length=MAX_SENT_LEN,
                   batch_size=BATCH_SIZE)


def get_model():
    return model


if __name__ == "__main__":
    model = get_model()
    example = "Not all that Mrs. Bennet, however, with the assistance of her five daughters, could ask on the subject, was " \
           "sufficient to draw from her husband any satisfactory description of Mr. Bingley. They attacked him in various " \
           "ways with barefaced questions, ingenious suppositions, and distant surmises; but he eluded the skill of them all," \
           " and they were at last obliged to accept the second-hand intelligence of their neighbour,"

    model.predict(example)
