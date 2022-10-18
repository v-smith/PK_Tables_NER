import spacy
from table_ner.utils import read_jsonl, write_jsonl


ner_teach_output = list(read_jsonl("../data/ActiveLearning/ner_teach_output/AL_SampleType.jsonl"))

model_trained = spacy.load("en_spacy_table_ner")

ner_review_input = []
for i in ner_teach_output:
    predict_text = model_trained(i["text"])
    prediction_spans = [dict(start=ent.start_char, end=ent.end_char, label=ent.label_) for ent in predict_text.ents]
    i.update({"spans": prediction_spans})
    ner_review_input.append(i)
    a=1

write_jsonl('../data/ActiveLearning/ner_teach_output/AL_SampleType_model-labelled.jsonl', ner_review_input)
a=1
