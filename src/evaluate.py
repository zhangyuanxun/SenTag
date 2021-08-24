from collections import defaultdict
import torch
from metrics import SeqEntityScore
from utils import get_entities, extract_text_from_bert_padding

import json
import os


def evaluate(args, model, tokenizer, dataloader, labels_list):
    print("Start to evaluate the model...")
    args.id2label = {i: label for i, label in enumerate(labels_list)}
    args.label2id = {label: i for i, label in enumerate(labels_list)}
    metric = SeqEntityScore(args.id2label, markup='bio')

    eval_loss = 0.0
    nb_eval_steps = 0
    all_predictions = defaultdict(dict)
    all_labels = defaultdict(dict)
    all_sentences_input_ids = defaultdict(dict)

    idx = 0
    for batch in dataloader:
        model.eval()

        inputs = {k: v.to(args.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']
            eval_loss += outputs['loss'].item()
            pred_labels = model.crf.obtain_labels(logits, args.id2label)
            print(pred_labels)

        true_label_ids = inputs['label_ids'].cpu().numpy().tolist()
        print(true_label_ids)
        sentences_input_ids = inputs['sentences_input_ids']

        for i in range(len(pred_labels)):
            all_predictions[idx] = get_entities(pred_labels[i], args.id2label, 'bio')
            all_labels[idx] = get_entities(true_label_ids[i], args.id2label, 'bio')
            all_sentences_input_ids[idx] = sentences_input_ids[i]
            idx += 1

        nb_eval_steps += 1

        for i, labels in enumerate(true_label_ids):
            true_label_path = []
            pred_label_path = []

            for j, m in enumerate(labels):
                true_label_path.append(args.id2label[true_label_ids[i][j]])
                pred_label_path.append(pred_labels[i][j])
            metric.update(pred_paths=[pred_label_path], label_paths=[true_label_path])

    assert len(all_predictions) == len(all_sentences_input_ids) == len(all_labels)

    prediction_results = list()

    for i in range(idx):
        predictions = all_predictions[i]
        labels = all_labels[i]
        results = list()
        sentences_input_ids = all_sentences_input_ids[i]
        sentences_context = list()

        for j in range(len(sentences_input_ids)):
            sentence = tokenizer.convert_ids_to_tokens(sentences_input_ids[j])
            sentence = tokenizer.convert_tokens_to_string(sentence)
            results.append([sentence, 'O', 'O'])

        for label, start, end in labels:
            for k in range(start, end + 1):
                results[k][1] = label

        for label, start, end in predictions:
            for k in range(start, end + 1):
                results[k][2] = label

        prediction_results.append(results)

    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    performance_results = {f'{key}': value for key, value in eval_info.items()}
    performance_results['loss'] = eval_loss

    print("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in performance_results.items()])
    print(info)

    # save the file
    with open(os.path.join(args.output_dir, "performance_results.json"), "w") as f:
        json.dump(performance_results, f)

    with open(os.path.join(args.output_dir, "prediction_results.json"), "w") as f:
        json.dump(prediction_results, f, indent=4)