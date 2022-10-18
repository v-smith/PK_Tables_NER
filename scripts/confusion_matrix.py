import warnings

import spacy

from table_ner.utils import read_jsonl, character_annotations_to_spacy_doc, clean_instance_span, plot_confusion_matrix, \
    get_dataset_labels

model_trained = spacy.load("en_spacy_table_ner")
base_tokenizer = model_trained
test_corpus = list(read_jsonl("../data/model_vs_annotator/PopDemCombined/model_compare_review_test.jsonl"))

misaligned_sentences = 0
bilou_data = []
for example in test_corpus:
    label_span = clean_instance_span(instance_spans=example["spans"])
    entities = []
    for label in label_span:
        entity = (label["start"], label["end"], label["label"])
        entities.append(entity)

    # check for misaligned spans
    doc, misaligned = character_annotations_to_spacy_doc(inp_annotation=example, tokenizer_model=model_trained)
    if misaligned:
        misaligned_sentences += 1
    else:
        # convert to bilou data
        bilou_tup = (example["text"], {"entities": entities})
        bilou_data.append(bilou_tup)
    a = 1

if misaligned_sentences > 0:
    warnings.warn(f"Number of misaligned sentences: {misaligned_sentences}"
                  f"({round(misaligned_sentences*100 / len(test_corpus), 2)}%)")

plot_confusion_matrix(bilou_data, classes=get_dataset_labels(bilou_data, nlp=model_trained), normalize=True,
                      nlp=model_trained)

