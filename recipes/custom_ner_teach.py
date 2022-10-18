import pickle
import prodigy
from prodigy.util import combine_models, split_string, get_labels, log, copy_nlp
from typing import List, Optional, Union, Iterable
from prodigy.components.loaders import JSONL
from prodigy.models.ner import EntityRecognizer
from prodigy.recipes.ner import teach
from prodigy.models.matcher import PatternMatcher
from prodigy.components.preprocess import split_sentences
from prodigy.components.sorters import prefer_uncertain
from prodigy.util import combine_models, split_string, set_hashes
import spacy


# from prodigy.components.loaders import get_stream
# https://github.com/explosion/prodigy-recipes/blob/master/ner/ner_teach.py

@prodigy.recipe(
    "table-teach-ner",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline with an entity recognizer", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    patterns=("Path to match patterns file", "option", "pt", str),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    unsegmented=("Don't split sentences", "flag", "U", bool),
    # fmt: on
)
def table_teach_ner(
        dataset: str,
        spacy_model: str,
        source: Union[str, Iterable[dict]],
        label: Optional[List[str]] = None,
        patterns: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        unsegmented: bool = False,
):
    blocks = [{"view_id": "ner"}, {"view_id": "html", "smallText": 20}]

    nlp = spacy.load(spacy_model)
    stream = get_stream(source)
    model = EntityRecognizer(nlp, label=label)

    if patterns is None:
        # No patterns are used, so just use the NER model to suggest examples
        # and only use the model's update method as the update callback
        predict = model
        update = model.update
    else:
        # Initialize the pattern matcher and load in the JSONL patterns
        matcher = PatternMatcher(nlp).from_disk(patterns)
        # Combine the NER model and the matcher and interleave their
        # suggestions and update both at the same time
        predict, update = combine_models(model, matcher)

    if not unsegmented:
        # Use spaCy to split text into sentences
        stream = split_sentences(nlp, stream)

    stream = prefer_uncertain(predict(stream))
    # stream = list(stream)

    return {
        "dataset": dataset,
        "view_id": "blocks",
        "stream": stream,
        "update": update,
        "exclude": exclude,
        "config": {
            "labels": ["PK", "Units", "Measure_Type", "Num_Value", "Chem/Drug", "Dose/Route", "Num_Sub",
                       "Population", "Demographics"],
            "custom_theme": {"cardMinWidth": 300, "cardMaxWidth": 1500, "show_flag": False},
            "blocks": blocks,
            "lang": nlp.lang
        }
    }


def get_stream(source):
    my_pickle = pickle.load(open('./data/ActiveLearning/table_hashes_remaining_ner.pkl', 'rb'))
    res = JSONL(source)
    for eg in res:
        html_hash = eg["html"]
        if my_pickle[html_hash]:
            html = my_pickle[html_hash]
        else:
            html = {}
        eg["html"] = html
        task = {"text": eg["text"], "html": eg["html"], "table_id": eg["table_id"],
                "col": eg["col"], "row": eg["row"], "meta": eg["meta"]}
        task = set_hashes(task)
        yield task


'''
return {
        "view_id": "ner",
        "dataset": dataset,
        "stream": (eg for eg in stream),
        "update": update,
        "exclude": exclude,
        "config": {
            "lang": nlp.lang,
            "label": ", ".join(label) if label is not None else "all",
        },
    }
components = teach(dataset=dataset, spacy_model=spacy_model, label=label, source=source, loader=loader,
                       patterns=patterns, exclude=exclude, unsegmented=unsegmented)
    print(components)

    components["stream"] = get_stream(source)

    components["view_id"] = "blocks"
    components["config"]["blocks"] = [{"view_id": "ner"}, {"view_id": "html", "smallText": 20}]
    components["config"]["custom_theme"] = {"cardMinWidth": 300, "cardMaxWidth": 1500, "show_flag": False}
    return components
'''
