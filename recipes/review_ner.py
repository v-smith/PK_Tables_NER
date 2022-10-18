import prodigy
from prodigy.recipes.review import review as rv
from typing import List, Dict, Optional, Any
from prodigy.util import split_string, get_labels


@prodigy.recipe(
    "review-ner",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    input_sets=("Comma-separated names of datasets to review", "positional", None, split_string),
    view_id=(
            "View ID (e.g. 'ner' or 'ner_manual') to use if none present in the task or to overwrite existing",
            "option", "v",
            str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    fetch_media=("Load images, audio or video files from local paths or URLs", "flag", "FM", bool),
    show_skipped=(
            "Include skipped answers, e.g. if annotator hit ignore or rejected manual annotation", "flag", "S", bool),
)
def review_ner(
        dataset: str,
        input_sets: List[str],
        view_id: Optional[str] = None,
        label: Optional[List[str]] = None,
        fetch_media: bool = False,
        show_skipped: bool = False, ):
    components = rv(dataset=dataset,
                    input_sets=input_sets,
                    label=label,
                    view_id=view_id,
                    fetch_media=fetch_media,
                    show_skipped=show_skipped)
    print(components)

    components["view_id"] = "blocks"
    components["config"]["blocks"] = [{"view_id": "review"}, {"view_id": "html", "smallText": 20}]
    components["config"]["custom_theme"] = {"cardMinWidth": 300, "cardMaxWidth": 1500, "show_flag": False}

    return components


#x = list(components["stream"])
#{"view_id": "text_input", "field_rows": 6, "field_label": "Comments", "user_input": x}
