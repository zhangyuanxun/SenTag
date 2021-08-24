import json
import copy
import os
import nltk


class InputFeatures(object):
    def __init__(self, sentences_input_ids, sentences_input_mask,
                 sentences_type_ids, sentences_input_len, label_ids):
        self.sentences_input_ids = sentences_input_ids
        self.sentences_input_mask = sentences_input_mask
        self.sentences_type_ids = sentences_type_ids
        self.label_ids = label_ids
        self.sentences_input_len = sentences_input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputExample(object):
    def __init__(self, guid, sentences, labels):
        self.guid = guid
        self.sentences = sentences
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SenTagProcessor(object):
    def get_debug_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "notes_debug.json")), 'train')

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "notes_train.json")), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "notes_dev.json")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "notes_test.json")), 'test')

    def get_labels(self):
        return ['B-f', 'I-f', 'O', 'B-s', 'I-s']

    def _word_tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def _read_data(self, input_file):
        with open (input_file) as fp:
            jObj = json.load((fp))
        data = []
        for note in jObj:
            # each note may have multiple paragraph, and each paragraph may has multiple sentences
            for para in jObj[note]:
                sentences = list()
                labels = list()

                for label, sentence in para:
                    tokens = self._word_tokenize(sentence.strip())
                    sentences.append(tokens)

                    if label == 1:
                        if labels and (labels[-1] == 'B-f' or labels[-1] == 'I-f'):
                            labels.append('I-f')
                        else:
                            labels.append('B-f')
                    elif label == 2:
                        if labels and (labels[-1] == 'B-s' or labels[-1] == 'I-s'):
                            labels.append('I-s')
                        else:
                            labels.append('B-s')
                    else:
                        labels.append('O')

                assert len(sentences) == len(labels)
                data.append({"sentences": sentences, "labels": labels})

        return data

    def _create_examples(self, data, data_type):
        examples = []
        for (i, line) in enumerate(data):
            guid = "%s-%s" % (data_type, i)
            sentences = line['sentences']
            labels = line['labels']
            examples.append(InputExample(guid=guid, sentences=sentences, labels=labels))

        return examples




