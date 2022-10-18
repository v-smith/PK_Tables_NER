"""Convert entity annotation from character-level NER to spaCy v3 .spacy format."""
import warnings
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from table_ner.utils import read_jsonl, character_annotations_to_char_doc, clean_spans
from table_ner.character_tokenizer import CharacterTokenizer


def convert(
        input_path: Path = typer.Option(default='../data/model_vs_annotator/reviewed_out/model_compare_review_dev.jsonl'),
        output_path: Path = typer.Option(default='../data/spacy2/dev.spacy/'),
        scispacy_base_tok: bool = typer.Option(default=False)
):
    if scispacy_base_tok:
        nlp = spacy.load("en_core_sci_lg")
    else:
        nlp = spacy.blank("en")
        nlp.tokenizer = CharacterTokenizer(nlp.vocab)
        db = DocBin()

        raw_annotations = list(read_jsonl(input_path))
        for annot_table in raw_annotations:

            spans = annot_sentence["spans"]
            text = annot_sentence["text"]

            if spans:
                join_indexes = which_spans_join(text=text, same_spans=spans)
                # check for consecutive indexes
                consec_indexes = is_consecutive(join_indexes)
                # combine consecutive entries from same spans
                final_spans = join_spans(join_indexes=join_indexes, consec_indexes=consec_indexes, same_spans=spans)
                cleaned_spans = clean_spans(text=text, spans=final_spans)
                annot_sentence.update({"spans": cleaned_spans})

            doc = character_annotations_to_char_doc(annot_table, nlp)
            db.add(doc)
            a=1

    output_path.parents[0].mkdir(parents=True, exist_ok=True)
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)
