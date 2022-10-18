"""
N.B. Spacy uses exact matching to evaluate NER models, need custom script to allow for partial matching.
"""
import warnings
from pathlib import Path

import spacy
import typer
from nervaluate import Evaluator
from seqeval.metrics import classification_report
from spacy.scorer import Scorer
from spacy.training import Example

from table_ner.partial_split_tokenizer import create_tokenizer
from table_ner.utils import read_jsonl, character_annotations_to_spacy_doc, check_predictions_vs_labels, \
    print_ner_scores, clean_instance_span, print_spacy_ner_scores, plot_confusion_matrix, get_iob_labels, \
    get_dataset_labels


def main(
        input_model_path: Path = typer.Option(default="../data/trained_models2/PopDemCombo/model-best",
                                              help="Path to the input model"),
        test_file_path: Path = typer.Option(
            default="../data/model_vs_annotator/PopDemCombined/model_compare_review_test.jsonl",
            help="Path to the jsonl file of the test/evaluation set")
):
    """
    Apply your trained NER model to the test/development set
    """
    # 1. load model and tokenizer
    model_trained = spacy.load(input_model_path)
    base_tokenizer = spacy.blank("en")
    base_tokenizer.tokenizer = create_tokenizer(base_tokenizer)

    #model_trained = spacy.load("en_spacy_table_ner")
    #base_tokenizer = model_trained

    # base_tokenizer = spacy.blank("en")
    # base_tokenizer.tokenizer = WhitespaceTokenizer(base_tokenizer.vocab)

    # Check that the tokenization rules are the same for the trained model than the base tokenizer
    if not ((base_tokenizer.tokenizer.infix_finditer == model_trained.tokenizer.infix_finditer) and
            (base_tokenizer.tokenizer.suffix_search == model_trained.tokenizer.suffix_search) and
            (base_tokenizer.tokenizer.prefix_search == model_trained.tokenizer.suffix_search)):
        msg = "The tokenizer used to split your text and generate IOB labels has different rules than the tokenizer " \
              "used in your trained model"
        warnings.warn(msg)

    test_corpus = list(read_jsonl(test_file_path))
    labels_spans = [clean_instance_span(instance_spans=instance["spans"]) for instance in test_corpus]

    texts_to_predict = [model_trained(annot["text"]) for annot in test_corpus]
    predictions_spans = []
    for prediction in model_trained.pipe(texts_to_predict):
        tmp_ents = [dict(start=ent.start_char, end=ent.end_char, label=ent.label_) for ent in prediction.ents]
        predictions_spans.append(tmp_ents)

    assert len(predictions_spans) == len(labels_spans)

    # Apply nervaluation metrics
    evaluator = Evaluator(labels_spans, predictions_spans,
                          tags=["PK", "Measure_Type", "Dose/Route", "Chem/Drug", "Population", "Units",
                                "Num_Value", "Num_Sub", "Sample_Type"])
    _, results_agg = evaluator.evaluate()
    print('=================== nervaluate metrics ==================')
    print_ner_scores(inp_dict=results_agg)

    # Spacy evaluation
    labelled_docs = []
    misaligned_sentences = 0
    for annot_sentence in test_corpus:
        doc, misaligned = character_annotations_to_spacy_doc(inp_annotation=annot_sentence,
                                                             tokenizer_model=base_tokenizer)
        if misaligned:
            misaligned_sentences += 1
        labelled_docs.append(doc)

    if misaligned_sentences > 0:
        warnings.warn(f"Number of misaligned sentences: {misaligned_sentences}"
                      f"({round(misaligned_sentences * 100 / len(test_corpus), 2)}%)")

    iob_labels = [get_iob_labels(spacy_doc) for spacy_doc in labelled_docs]
    iob_predictions = [get_iob_labels(spacy_doc_p) for spacy_doc_p in texts_to_predict]

    check_predictions_vs_labels(iob_labels, iob_predictions)

    examples = []
    for i in range(len(texts_to_predict)):
        examples.append(Example(texts_to_predict[i], labelled_docs[i]))
    scorer = Scorer(model_trained)
    scores = scorer.score(examples)
    print('=================== nervaluate metrics ==================')
    print_ner_scores(inp_dict=results_agg)
    print('=================== spaCy metrics ==================')
    print_spacy_ner_scores(inp_dict=scores)

    # seqeval
    print('=================== seqeval metrics ==================')
    print(classification_report(iob_labels, iob_predictions, digits=4))

    ###### Confusion Matrix ######
    bilou_data = []
    for example in test_corpus:
        label_span = clean_instance_span(instance_spans=example["spans"])
        entities = []
        for label in label_span:
            entity = (label["start"], label["end"], label["label"])
            entities.append(entity)
        bilou_tup = (example["text"], {"entities": entities})
        bilou_data.append(bilou_tup)
        a = 1

    get_dataset_labels(bilou_data, nlp=model_trained)

    plot_confusion_matrix(bilou_data, classes=get_dataset_labels(bilou_data, nlp=model_trained), normalize=True,
                          nlp=model_trained)


if __name__ == "__main__":
    typer.run(main)
