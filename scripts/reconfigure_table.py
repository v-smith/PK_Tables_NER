"""Reconfigures Cell Entities into tables"""
from table_ner.utils import read_jsonl, clean_instance_span, write_jsonl
import spacy
from itertools import groupby

# starting from model labelled cell entities for tables

data = list(read_jsonl("../data/recontructing/parsed_trial_ner_50_entities_Rel.jsonl"))
model_trained = spacy.load("en_spacy_table_ner")

# get model predictions
predicted_cells = []
for cell in data:
    if cell["row"] == "na" and cell["col"] == "na":
        cell["is_caption"] = True
    else:
        cell["is_caption"] = False
    text_to_predict = model_trained(cell["text"])
    tmp_ents = [dict(start=ent.start_char, end=ent.end_char, label=ent.label_) for ent in text_to_predict.ents]
    cell["ents"] = tmp_ents
    predicted_cells.append(cell)
    a = 1


# group by table_id
def key_func(k):
    return k["table_id"]


grouped_tables = []
for key, value in groupby(predicted_cells, key_func):
    table_cells = list(value)
    grouped_tables.append(table_cells)

write_jsonl("../data/recontructing/grouped_predicted_tables.jsonl", grouped_tables)
