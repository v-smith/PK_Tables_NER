"""Script to generate EDA plots for Table NER data"""
from table_ner.utils import read_jsonl
from table_ner.partial_split_tokenizer import create_tokenizer
from table_ner.plots import plot_wordcloud, plot_class_distrib, plot_dataset_sizes, plot_token_lengths, label_stats_perclass, label_stats_perdataset
import spacy

# load in raw annotations
test_annotations = list(read_jsonl('../data/split_data/test.jsonl'))
train_annotations = list(read_jsonl('../data/split_data/train.jsonl'))
dev_annotations = list(read_jsonl('../data/split_data/dev.jsonl'))
data_sets = {"Test": test_annotations, "Train": train_annotations, "Dev": dev_annotations}

# load tokenizer model
nlp = spacy.blank("en")
nlp.tokenizer = create_tokenizer(nlp)

# plot dataset sizes
plot_dataset_sizes(train_annotations, dev_annotations, test_annotations)

# generate other plots
for k, v in data_sets.items():
    plot_class_distrib(nlp, v, k)
    plot_token_lengths(v, nlp, k)
    label_stats_perdataset(v, nlp, k)
    label_stats_perclass(v, nlp, k)
    plot_wordcloud(v, nlp, k)
