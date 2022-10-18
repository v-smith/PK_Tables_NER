# imports
# from azure.storage.blob import BlobClient
import numpy as np
from nervaluate import Evaluator
from table_ner.utils import read_jsonl, print_ner_scores, clean_instance_span
import collections
from more_itertools import pairwise
from table_ner.utils import which_spans_join, is_consecutive, join_spans, view_all_entities_terminal, clean_spans

annotations = list(read_jsonl("../data/vm_annotations_out/table_ner_test_1000_3-output.jsonl"))

uq_annotators = set([x["_session_id"] for x in annotations if x["_session_id"][-5:] != "vicky"])

split_annotators = []
split_annotators_text = []
for annotator in uq_annotators:
    sub_annotations = [an for an in annotations if an["_session_id"] == annotator]
    # replace spans
    updated_sub_annotations = []
    for annot_sentence in sub_annotations:
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
            cleaned_spans = clean_spans(text=text, spans=final_spans)
            annot_sentence.update({"spans": cleaned_spans})

        updated_sub_annotations.append(annot_sentence)
        a = 1

    split_annotators_text.append(updated_sub_annotations)

    updated_sub_annotations = [{x["_task_hash"]: x["spans"]} for x in updated_sub_annotations]
    updated_sub_annotations = {k: v for d in updated_sub_annotations for k, v in d.items()}
    split_annotators.append(updated_sub_annotations)

most_anns = max(len(y) for y in split_annotators_text)
split_annotators_text = [b for b in split_annotators_text if len(b) == most_anns]

split_annotators = [b for b in split_annotators if len(b) == most_anns]

# Apply nervaluation metrics
pairwise_list = []
for item1 in split_annotators_text[0]:
    matching_d = [d for d in split_annotators_text[1] if d["_task_hash"] == item1["_task_hash"]][0]
    if matching_d:
        my_tup = (item1, matching_d)
        pairwise_list.append(my_tup)
for tup in pairwise_list:
    assert tup[0]["_task_hash"] == tup[1]["_task_hash"]

list1 = [x[0] for x in pairwise_list]
list2 = [x[1] for x in pairwise_list]
list_annotations2 = [list1, list2]

ordered_annotators = [collections.OrderedDict(sorted(lst.items())) for lst in split_annotators]
pairwise_keys = [(list1, list2) for (list1, list2) in pairwise(ordered_annotators)]
list_annotations = [[v for k, v in l.items()] for l in ordered_annotators]

labels_spans = [[clean_instance_span(instance_spans=instance["spans"]) for instance in sublist] for sublist in
                list_annotations2]
evaluators = [Evaluator(ele1, ele2,
                        tags=["PK", "Measure_Type", "Dose/Route", "Population",
                              "Chem/Drug", "Units", "Num_Value", "Num_Sub", "Sample_Type"])
              for (ele1, ele2) in pairwise(labels_spans)]

for evaluator in evaluators:
    _, results_agg = evaluator.evaluate()
    micro_f1_partial = np.mean([v["partial"]["f1"] for k, v in results_agg.items()])
    micro_f1_strict = np.mean([v["strict"]["f1"] for k, v in results_agg.items()])
    print('=================== nervaluate IAA metrics ==================')
    print(f"Macro F1 Partial: {micro_f1_partial}")
    print(f"Macro F1 Strict: {micro_f1_strict}")
    print_ner_scores(inp_dict=results_agg)

    print("\n===== Printing discrepancies between annotations and model prediction =====")

    for instance, predicted_ent, true_ent in zip(list_annotations2[0], labels_spans[0], labels_spans[1]):
        sentence_text = instance["text"]
        if predicted_ent != true_ent:
            print(10 * "=", f"Example with task hash {instance['_task_hash']}", 10 * "=")
            print("REAL LABELS:")
            print(view_all_entities_terminal(inp_text=sentence_text, character_annotations=true_ent))
            print("MODEL PREDICTIONS:")



""""
blob = BlobClient(
    account_url="https://pkpdaiannotationstables.blob.core.windows.net",
    container_name="pkpdaiannotationstables",
    blob_name="table_ner_test_1000_1-output.jsonl",
    credential="?sv=2020-10-02&ss=btqf&srt=sco&st=2022-03-31T10%3A38%3A48Z&se=2025-01-01T11%3A38%3A00Z&sp=rwdxftlacup&sig=nX0fJmK0hXOYzSbZujQgT09FX1%2FUaI3Cm%2FuWk6SuOtA%3D")


#or try smart-open package to download 
this works: wget "https://pkpdaiannotationstables.blob.core.windows.net/annotations/table_ner_test_1000_1-output.jsonl?sv=2020-10-02&ss=btqf&srt=sco&st=2022-03-31T10%3A38%3A48Z&se=2025-01-01T11%3A38%3A00Z&sp=rwdxftlacup&sig=nX0fJmK0hXOYzSbZujQgT09FX1%2FUaI3Cm%2FuWk6SuOtA%3D"

filename = "../data/temp_annotations.jsonl"
with open(filename, "wb") as f:
    f.write(blob.download_blob().readall())
"""
