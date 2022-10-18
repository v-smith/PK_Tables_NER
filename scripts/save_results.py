"""
N.B. Spacy uses exact matching to evaluate NER models, need custom script to allow for partial matching.
"""
import warnings
import os
import typer
from pathlib import Path
import spacy
from table_ner.partial_split_tokenizer import create_tokenizer
from table_ner.whitespace_tokenizer import WhitespaceTokenizer
from table_ner.utils import read_jsonl, clean_instance_span, write_jsonl


def main(
        input_model_path: Path = typer.Option(default="../data/trained_models/model-last",
                                              help="Path to the input model"),
        file_path: Path = typer.Option(default="../data/split_data/train.jsonl",
                                       help="Path to the jsonl file of the set")
):
    """
    Apply your trained NER model to the test/development set
    """
    # 1. load model and tokenizer
    model_trained = spacy.load(input_model_path)
    base_tokenizer = spacy.blank("en")
    base_tokenizer.tokenizer = create_tokenizer(base_tokenizer)
    # base_tokenizer = spacy.blank("en")
    # base_tokenizer.tokenizer = WhitespaceTokenizer(base_tokenizer.vocab)

    # Check that the tokenization rules are the same for the trained model as the base tokenizer
    if not ((base_tokenizer.tokenizer.infix_finditer == model_trained.tokenizer.infix_finditer) and
            (base_tokenizer.tokenizer.suffix_search == model_trained.tokenizer.suffix_search) and
            (base_tokenizer.tokenizer.prefix_search == model_trained.tokenizer.suffix_search)):
        msg = "The tokenizer used to split your text and generate IOB labels has different rules than the tokenizer " \
              "used in your trained model"
        warnings.warn(msg)

    final_corpus = []
    corpus = list(read_jsonl(file_path))
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    for instance in corpus:
        instance["_session_id"] = "annotators" + "_" + file_name
        instance["_view_id"] = "blocks"
        final_corpus.append(instance)
        # labels_span = clean_instance_span(instance_spans=instance["spans"])
        texts_to_predict = model_trained(instance["text"])
        predictions_spans = []
        for prediction in model_trained.pipe([texts_to_predict]):
            tmp_ents = [
                dict(start=ent.start_char, end=ent.end_char,
                     label=ent.label_) for ent in prediction.ents] #token_start=ent.start_char, token_end=(ent.end_char - 1),
            predictions_spans.extend(tmp_ents)
        # assert len(predictions_spans) == len(labels_span)
        new_instance = instance.copy()
        new_instance["spans"] = predictions_spans
        new_instance["_session_id"] = "ner_model" + "_" + file_name
        final_corpus.append(new_instance)
        a = 1

    write_jsonl('../data/model_vs_annotator/train_model_vs_annotator.jsonl', final_corpus)


if __name__ == "__main__":
    typer.run(main)
