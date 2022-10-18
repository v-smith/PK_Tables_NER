import itertools
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy
import ujson
from matplotlib import pyplot
from nervaluate import Evaluator
from sklearn.metrics import confusion_matrix
# from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from spacy import Language
from spacy.tokens.doc import Doc
from spacy.training import offsets_to_biluo_tags
from termcolor import colored


# from termcolor import colored
# from sty import fg

def get_biluo_labels(inp_doc: Doc) -> List[str]:
    ch_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in inp_doc.ents]
    return offsets_to_biluo_tags(inp_doc, ch_entities)


def read_jsonl(file_path):
    # Taken from prodigy support
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def write_jsonl(file_path, lines):
    # Taken from prodigy
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))


def character_annotations_to_spacy_doc(inp_annotation: Dict, tokenizer_model) -> Tuple[Doc, bool]:
    """
    Converts an input sentence annotated at the character level for NER to a spaCy doc object
    It assumes that the inp_annotation has:
        1. "text" field
        2. "spans" field with a list of NER annotations in the form of  {"start": <ch_idx>, "end": <ch_idx>,
        "label": <NER label name>}
    """
    text = inp_annotation["text"]  # extra
    doc = tokenizer_model.make_doc(text)  # extra
    ents = []  # extra
    misaligned = False
    if "spans" in inp_annotation.keys():
        for entities_sentence in inp_annotation["spans"]:
            start = entities_sentence["start"]
            end = entities_sentence["end"]
            label = entities_sentence["label"]
            span = doc.char_span(start, end, label=label)
            if span is None:
                misaligned = True
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character" \
                      f" span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n "
                warnings.warn(msg)
            else:
                ents.append(span)
    doc.ents = ents
    return doc, misaligned


def character_annotations_to_char_doc(annot_table: Dict, char_model: Language) -> Tuple[Doc, bool]:
    """
    Converts an input sentence annotated at the character level for NER to a spaCy doc object
    It assumes that the inp_annotation has:
        1. "text" field
        2. "spans" field with a list of NER annotations in the form of  {"start": <ch_idx>, "end": <ch_idx>,
        "label": <NER label name>}
    """
    ents = []
    text = annot_table["text"]
    chars = [char for char in text]
    doc = char_model.make_doc(text)
    if "spans" in annot_table.keys():
        for entities_sentence in annot_table["spans"]:
            # add up how many whitespaces from character gaps are in a word and add to this
            start = entities_sentence["start"]
            end = entities_sentence["end"]
            if start != 0:
                start = start + (start - 1)
            end = end + (end - 1)
            label = entities_sentence["label"]
            span = doc.char_span(start, end, label=label)  # experiment with strict/contract/expand
            if span is not None:
                ents.append(span)
            a = 1
            # for i in range(start, end):
            # span = doc.char_span(i, (i + 1), label=label)
            # if span is not None:
            # ents.append(span)

    doc.ents = ents
    return doc


def get_iob_labels(inp_doc: Doc) -> List[str]:
    return [token.ent_iob_ + "-" + token.ent_type_ if token.ent_type_ else token.ent_iob_ for token in inp_doc]


def get_biluo_labels(inp_doc: Doc) -> List[str]:
    ch_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in inp_doc.ents]
    return offsets_to_biluo_tags(inp_doc, ch_entities)


def check_predictions_vs_labels(inp_labels: List, inp_predictions: List):
    uq_labels = set(list(itertools.chain(*inp_labels)))
    uq_predictions = set(list(itertools.chain(*inp_predictions)))
    not_predicted = sorted([i for i in uq_labels if i not in uq_predictions])
    print(f"The token labels for this corpus are:\n{uq_labels}")
    # assert uq_labels == uq_predictions
    if len(not_predicted) >= 1:
        print(f"These labels are not being predicted: {not_predicted}")
    for labels, predictions in zip(inp_labels, inp_predictions):
        if len(labels) != len(predictions):
            print(labels, predictions, "\n")
        # assert len(labels) == len(predictions)


def clean_instance_span(instance_spans: Dict):
    return [dict(start=x['start'], end=x['end'], label=x['label']) for x in instance_spans]


def get_ner_scores(pred_ents_ch: List[List[Dict]], true_ents_ch: List[List[Dict]], inp_tags: List[str],
                   original_annotations: List[Dict],
                   display_errors: bool, display_all: bool = False):
    """
    @param pred_ents_ch: entities predicted at the character level expressed as:
    [
    [{'start': 3, 'end': 12, 'label': 'PK'},
    {'start': 17, 'end': 39, 'label': 'PK'}].
    [],
    [{'start': 15, 'end': 18, 'label': 'PK'}]
    ] one list per sentence
    @param inp_tags: list of entities to consider. e.g., ["PK", "VALUE", "UNITS"]
    @param true_ents_ch: true entities in the same format
    @param original_annotations: Original annotations with _task_hash and text
    @param display_errors: whether to display prediction errors in the terminal window
    @return: prints the results
    :param display_all:
    """
    assert len(pred_ents_ch) == len(true_ents_ch)
    evaluator = Evaluator(true_ents_ch, pred_ents_ch, tags=inp_tags)
    _, results_agg = evaluator.evaluate()

    print('=================== nervaluate metrics ==================')
    print_ner_scores(inp_dict=results_agg)

    print("\n===== Printing discrepancies between annotations and model prediction =====")

    if display_errors or display_all:
        i = 0
        for instance, predicted_ent, true_ent in zip(original_annotations, pred_ents_ch, true_ents_ch):
            sentence_text = instance["text"]
            if predicted_ent != true_ent or display_all:
                i += 1
                instance["_task_hash"] = 8888 if "_task_hash" not in instance.keys() else instance["_task_hash"]
                print(10 * "=", f"Example with task hash {instance['_task_hash']} n={i}", 10 * "=")
                print("REAL LABELS:")
                print(view_all_entities_terminal(inp_text=sentence_text, character_annotations=true_ent))
                print("MODEL PREDICTIONS:")
                print(view_all_entities_terminal(inp_text=sentence_text, character_annotations=predicted_ent))

    print_ner_scores(inp_dict=results_agg)


def view_all_entities_terminal(inp_text: str, character_annotations: List[Dict]):
    if character_annotations:
        character_annotations = sorted(character_annotations, key=lambda anno: anno['start'])
        sentence_text = ""
        end_previous = 0
        for annotation in character_annotations:
            sentence_text += inp_text[end_previous:annotation["start"]]
            label = annotation["label"]
            label_colours = {"PK": "green", "Measure_Type": "white", "Dose/Route": "red", "Population": "yellow",
                             "Chem/Drug": "magenta", "Units": "cyan",
                             "Num_Value": "white", "Num_Sub": "green", "Sample_Type": "blue"}
            term_colour = label_colours[label]
            sentence_text += colored(inp_text[annotation["start"]:annotation["end"]],
                                     term_colour, attrs=['reverse', 'bold'])
            end_previous = annotation["end"]
        sentence_text += inp_text[end_previous:]
        return sentence_text
    return inp_text


def print_ner_scores(inp_dict: Dict):
    """
    @param inp_dict: Dictionary with keys corresponding to entity types and subkeys to metrics
    e.g. {'PK': {'ent_type': {..},{'partial': {..},{'strict': {..} }}
    @return: Prints summary of metrics
    """
    overall_table_data = [["metric", "ent", "precision", "recall", "f1"]]
    for ent_type in inp_dict.keys():
        # print(f"====== Stats for {ent_type} ======")
        for metric_type in inp_dict[ent_type].keys():
            if metric_type in ['partial', 'strict']:
                # print(f" === {metric_type} match: === ")
                precision = inp_dict[ent_type][metric_type]['precision']
                recall = inp_dict[ent_type][metric_type]['recall']
                f1 = inp_dict[ent_type][metric_type]['f1']
                p = round(precision * 100, 2)
                r = round(recall * 100, 2)
                f1 = round(f1 * 100, 2)
                overall_table_data.append([metric_type, ent_type, p, r, f1])
                # print(f" Precision:\t {p}%")
                # print(f" Recall:\t {r}%")
                # print(f" F1:\t\t {f1}%")
                a = 1

    for row in overall_table_data:
        print("{: >20} {: >20} {: >20} {: >20} {: >20}".format(*row))


def print_spacy_ner_scores(inp_dict: Dict):
    """
    @param inp_dict: Dictionary with keys corresponding to entity types and subkeys to metrics
    e.g. {'PK': {'ent_type': {..},{'partial': {..},{'strict': {..} }}
    @return: Prints summary of metrics
    """
    table_data = [[" ", "precision", "recall", "f1-score"]]
    token_acc = round(inp_dict['token_acc'] * 100, 2)
    print(f"Token accuracy: {token_acc}")
    per_entity_metrics = inp_dict['ents_per_type']
    for ent_type in per_entity_metrics.keys():
        # support = get_support()
        # print(f" ==== Stats for {ent_type} ====")
        p = round(per_entity_metrics[ent_type]['p'] * 100, 2)
        r = round(per_entity_metrics[ent_type]['r'] * 100, 2)
        f1 = round(per_entity_metrics[ent_type]['f'] * 100, 2)
        table_data.append([ent_type, p, r, f1])
        # print(f" Precision:\t {p}%")
        # print(f" Recall:\t {r}%")
        # print(f" F1:\t\t {f1}%")
    for row in table_data:
        print("{: >20} {: >20} {: >20} {: >20}".format(*row))


def seperate_class_tokens(raw_annotations, tokenizer):
    all_ent_labels = {"PK": [], "Measure_Type": [], "Dose/Route": [], "Population": [],
                      "Demographics": [], "Chem/Drug": [], "Units": [],
                      "Num_Value": [], "Num_Sub": [], "Sample_Type": []}
    for annot_sentence in raw_annotations:
        doc, misaligned = character_annotations_to_spacy_doc(inp_annotation=annot_sentence, tokenizer_model=tokenizer)
        ents = doc.ents
        for ent in ents:
            if str(ent).isascii():
                label = ent.label_
                for k, v in all_ent_labels.items():
                    if k == label:
                        ent = str(ent).lower().strip()
                        v.append(ent)
                        a = 1

    return all_ent_labels


# def counts_per_class()

def join_spans(join_indexes, consec_indexes, same_spans):
    new_spans = []
    for (start, end) in consec_indexes:
        thiselem = same_spans[start]
        nextelem = same_spans[end]
        joined_span = dict(start=thiselem["start"], end=nextelem["end"],
                           token_start=thiselem["token_start"], token_end=nextelem["token_end"],
                           label=thiselem["label"])
        new_spans.append(joined_span)

    leftover_spans = [x for x in same_spans if same_spans.index(x) not in join_indexes]
    final_spans = leftover_spans + new_spans
    final_spans = sorted(final_spans, key=lambda d: d['start'])

    return final_spans


def is_consecutive(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def which_spans_join(text, same_spans):
    indexes_to_join = []
    """returns list index of spans to join"""
    a = 1
    for idx, ele in enumerate(same_spans):
        if len(same_spans) > idx + 1:
            nextele = same_spans[idx + 1]
            if nextele["label"] == ele["label"]:
                joining_text = (text[ele["end"]:
                                     nextele["start"] - 1])
                if not joining_text:
                    indexes_to_join.append(idx)
                    indexes_to_join.append(idx + 1)
                elif joining_text.isspace():
                    indexes_to_join.append(idx)
                    indexes_to_join.append(idx + 1)
                elif joining_text.strip().lower() in ["and", "or", "with", "also", "including", "together with",
                                                      "along with"]:
                    indexes_to_join.append(idx)
                    indexes_to_join.append(idx + 1)
    indexes_to_join = list(set(indexes_to_join))
    return indexes_to_join


def clean_spans(text, spans):
    """cleans any 'n=' or 'pk parameter' incorrect labels"""
    cleaned_spans = []
    for ele in spans:
        subject_text = (text[ele["start"]: ele["end"]])
        subject_text = subject_text.replace(" ", "").lower().strip()
        if subject_text in ["n=", "n =", "n ="]:
            clean_span = dict(start=ele["start"], end=(ele["start"] + 1),
                              token_start=ele["token_start"], token_end=ele["token_start"] + 1, label=ele["label"])
            cleaned_spans.append(clean_span)
        elif subject_text in ["pkparameters", "parameters", "pharmacokinetics", "pharmacokinetic",
                              "pharmacokineticvariables",
                              "pharmacokineticsparameters", "pharmacokineticparameters", "pkparameter",
                              "pharmacokineticparameter", "kineticparameters"]:
            pass
            a = 1
        else:
            cleaned_spans.append(ele)

    return cleaned_spans


def get_cleaned_label(label: str):
    if "-" in label:
        return label.split("-")[1]
    else:
        return label


def create_total_target_vector(docs, nlp):
    target_vector = []
    for doc in docs:
        new = nlp(doc[0])
        entities = doc[1]["entities"]
        bilou_entities = offsets_to_biluo_tags(new, entities)
        final = []
        for item in bilou_entities:
            final.append(get_cleaned_label(item))
        '''
        for i in final:
            if not i:
                print([i])
                print(bilou_entities)
                print(final)
                a=1
        '''
        target_vector.extend(final)
    return target_vector


def create_prediction_vector(text, nlp):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(text, nlp=nlp)]


def create_total_prediction_vector(docs: list, nlp):
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(doc[0], nlp=nlp))
    return prediction_vector


def get_all_ner_predictions(text, nlp):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities


def get_model_labels(nlp):
    labels = list(nlp.get_pipe("ner").labels)
    labels.append("O")
    return sorted(labels)


def get_dataset_labels(docs, nlp):
    # vect =create_total_target_vector(docs, nlp=nlp)
    return sorted(set(create_total_target_vector(docs, nlp=nlp)))


def generate_confusion_matrix(docs, nlp):
    classes = sorted(set(create_total_target_vector(docs, nlp=nlp)))
    y_true = create_total_target_vector(docs, nlp)
    y_pred = create_total_prediction_vector(docs, nlp)
    # print(y_true)
    # print(y_pred)
    return confusion_matrix(y_true, y_pred,
                            labels=classes)


def plot_confusion_matrix(docs, classes, nlp, normalize, cmap=pyplot.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    https://dulaj.medium.com/confusion-matrix-visualization-for-spacy-ner-9e9d99120ee9
    """

    title = 'Confusion Matrix, for SpaCy NER'

    # Compute confusion matrix
    cm = generate_confusion_matrix(docs, nlp)
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]) * 100
        a = 1

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.show()
    return cm, ax, pyplot


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
