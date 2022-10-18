import os
from pathlib import Path
import typer
from table_ner.utils import join_spans, which_spans_join, is_consecutive
from table_ner.utils import read_jsonl, write_jsonl


def convert(
        input_dir: Path = typer.Option(default='../data/model_vs_annotator/reviewed_out/'),
        output_dir: Path = typer.Option(default='../data/model_vs_annotator/PopDemCombined/'),
):
    dir_list = os.listdir(input_dir)
    for file in dir_list:
        filename = os.path.splitext(file)[0]
        raw_annotations = list(read_jsonl(str(input_dir) + "/" + str(file)))
        # raw_annotations = test_data
        final_annotations = []
        for annot_sentence in raw_annotations:
            # remove Pop and Dem labels
            for x in annot_sentence["spans"]:
                if x["label"] == "Demographics":
                    x["label"] = "Population"

            # define rules to combine spans if there are gaps....
            spans = annot_sentence["spans"]
            text = annot_sentence["text"]

            # check elements in list with same label
            # same_spans = [x for x in spans if any(spans[i]["label"] == spans[i + 1]["label"] for i in range(len(spans) - 1))]
            if spans:
                join_indexes = which_spans_join(text=text, same_spans=spans)
                # check for consecutive indexes
                consec_indexes = is_consecutive(join_indexes)
                # combine consecutive entries from same spans
                final_spans = join_spans(join_indexes=join_indexes, consec_indexes=consec_indexes, same_spans=spans)
                # update original annotated example
                annot_sentence.update({"spans": final_spans})

            final_annotations.append(annot_sentence)

        for x in final_annotations:
            print(x["text"])
            print(x["spans"])

        write_jsonl(str(output_dir) + "/" + str(filename) + ".jsonl", raw_annotations)

        a = 1


if __name__ == "__main__":
    typer.run(convert)


