import pickle
import copy
import spacy
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens, split_sentences
from prodigy.util import split_string, set_hashes
from table_ner.partial_split_tokenizer import create_tokenizer


@prodigy.recipe("table-ner-correct")
def table_ner_correct(source, dataset, spacy_model, label):
    # We can use the blocks to override certain config and content, and set "text": None for the choice interface so it doesn't also render the text
    blocks = [
        {"view_id": "ner_correct", "fontWeight": "bold"},
        {"view_id": "html", "smallText": 20},
        {"view_id": "text_input", "field_rows": 3, "field_label": "Comments"}]

    # nlp = spacy.blank(lang)
    nlp = spacy.load(spacy_model)
    base_tokenizer = spacy.blank("en")
    base_tokenizer.tokenizer = create_tokenizer(base_tokenizer)
    labels = label

    stream = get_stream(source)
    stream = add_tokens(nlp, stream)
    #
    stream = make_tasks(nlp, stream, labels)
    stream = list(stream)

    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        # "validate_answer": validate_answer,
        "config": {
            "labels": ["PK", "Units", "Measure_Type", "Num_Value", "Chem/Drug", "Dose/Route", "Num_Sub",
                       "Population", "Demographics"],
            "custom_theme": {"cardMinWidth": 300, "cardMaxWidth": 1500, "show_flag": False},
            "global_css": ".prodigy-button-reject, .prodigy-button-ignore {display: none}",
            "blocks": blocks,
            "batch_size": 10,
            "show_flag": True,
            "feed_overlap": True,
            "force_stream_order": True
        }
    }


def get_stream(source):
    my_pickle = pickle.load(open('./data/vicky/pkl_files/table_hashes_test_ner_1000.pkl', 'rb'))
    res = JSONL(source)
    for eg in res:
        html_hash = eg["html"]
        html = my_pickle[html_hash]
        eg["html"] = html
        yield {"text": eg["text"], "html": eg["html"], "table_id": eg["table_id"],
               "col": eg["col"], "row": eg["row"], "meta": eg["meta"], "spans": eg["ents"]}


def make_tasks(nlp, stream, labels):
    """Add a 'spans' key to each example, with predicted entities."""
    # Process the stream using spaCy's nlp.pipe, which yields doc objects.
    # If as_tuples=True is set, you can pass in (text, context) tuples.
    texts = ((eg["text"], eg) for eg in stream)
    for doc, eg in nlp.pipe(texts, as_tuples=True):

        task = copy.deepcopy(eg)
        spans = []
        for ent in doc.ents:
            # Ignore if the predicted entity is not in the selected labels.
            if labels and ent.label_ not in labels:
                continue
            # Create a span dict for the predicted entity.
            spans.append(
                {
                    "token_start": ent.start,
                    "token_end": ent.end - 1,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "label": ent.label_,
                }
            )
        task["spans"] = spans
        # Rehash the newly created task so that hashes reflect added data.
        task = set_hashes(task)
        yield task
