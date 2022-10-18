from pathlib import Path
import spacy
import typer

from table_ner.partial_split_tokenizer import create_tokenizer
from table_ner.utils import read_jsonl


def main(
        model_path: Path = typer.Argument("../data/trained_models/model-last", exists=True, dir_okay=False),
        output_path: Path = typer.Argument("../spacy_table_ner/", dir_okay=True),
):
    nlp = spacy.load(model_path)
    nlp.tokenizer = create_tokenizer(nlp)
    #ruler = nlp.add_pipe("entity_ruler")
    nlp.to_disk(output_path)
    print(f"Rule-Based model saved at {output_path}.")


if __name__ == "__main__":
    typer.run(main)
