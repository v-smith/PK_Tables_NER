import prodigy
from prodigy.components.preprocess import add_tokens
import spacy
import pickle
from prodigy.components.loaders import JSONL


@prodigy.recipe("table-ner")
def table_ner(source, dataset, lang="en"):
    # We can use the blocks to override certain config and content, and set "text": None for the choice interface so it doesn't also render the text
    blocks = [
        {"view_id": "ner_manual", "ner_manual_highlight_chars": True, "fontWeight": "bold"},
        {"view_id": "html", "smallText": 20},
        {"view_id": "text_input", "field_rows": 3, "field_label": "Comments"}]

    nlp = spacy.blank(lang)

    stream = JSONL(source)
    stream = add_tokens(nlp, stream, use_chars=True)
    stream = list(stream)

    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        # "validate_answer": validate_answer,
        "config": {
            "labels": ["PK", "Units", "Measure_Type", "Num_Value", "Chem/Drug", "Dose/Route", "Num_Sub",
                       "Population", "Demographics", "Sample_Type"],
            "custom_theme": {"cardMinWidth": 300, "cardMaxWidth": 1500, "show_flag": False},
            "global_css": ".prodigy-button-reject, .prodigy-button-ignore {display: none}",
            "blocks": blocks,
            "batch_size": 10,
        }
    }


def get_stream(source):
    my_pickle = pickle.load(open('./data/ActiveLearning/table_hashes_remaining_ner.pkl', 'rb'))
    res = JSONL(source)
    for eg in res:
        html_hash = eg["html"]
        html = my_pickle[html_hash]
        eg["html"] = html
        yield {"text": eg["text"], "html": eg["html"], "table_id": eg["table_id"],
               "col": eg["col"], "row": eg["row"], "meta": eg["meta"], "spans": eg["ents"]}
