"""EDA Plots for Table NER Data"""
import operator
from typing import List, Dict
from wordcloud import WordCloud
import pandas as pd
from table_ner.utils import character_annotations_to_spacy_doc, get_iob_labels, seperate_class_tokens
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib

matplotlib.style.use('ggplot')


def plot_dataset_sizes(train_annotations, dev_annotations, test_annotations):
    """Compares Total Number of Samples in each data set"""
    data = [["train", "dev", "test"], [len(train_annotations), len(dev_annotations), len(test_annotations)]]
    df = pd.DataFrame(data)  # columns=['Data Set', 'Table Cells']
    df_transposed = df.T
    df_transposed.columns = ['Data Set', 'Table Cells']
    df_transposed.plot(kind="bar", x="Data Set", y="Table Cells", legend=False, color=["red", "green", "blue"])
    plt.title("Total Data Set Sizes (Number of Table Cells)")
    plt.tight_layout()
    plt.show()


def plot_class_distrib(tokenizer, data_set, name):
    """Plots Frequency of Class Labels within each data set"""
    spacy_docs = []
    for annot_sentence in data_set:
        doc, _ = character_annotations_to_spacy_doc(inp_annotation=annot_sentence, tokenizer_model=tokenizer)
        spacy_docs.append(doc)
    iob_labels = [get_iob_labels(spacy_doc) for spacy_doc in spacy_docs]
    labels = [[token.ent_type_ if token.ent_type_ else token.ent_iob_ for token in inp_doc] for inp_doc in spacy_docs]
    label_frequencies = Counter([item for sublist in labels for item in sublist])
    iob_frequencies = Counter([item for sublist in iob_labels for item in sublist])
    my_df1 = pd.DataFrame.from_dict(label_frequencies, orient="index", columns=["Frequency"])
    my_df = my_df1.sort_index()
    my_df.drop(labels="O", axis=0, inplace=True)
    plt.bar(my_df.index, my_df["Frequency"], color="blue")
    plt.title(f"Frequencies of Labels in the {name} Set")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_token_lengths(json_list: List[Dict], tokenizer, name):
    """Plots Histogram of Total Token Lengths per Table Cell for each data set"""
    plot_text = [x["text"] for x in json_list]
    tokens = [tokenizer(text) for text in plot_text]
    lens = [len(token) for token in tokens]
    plt.hist(lens)
    plt.xlabel("Token Lengths")
    plt.title(f"Histogram of Total Token Lengths per Table Cell in {name} Set")
    plt.tight_layout()
    plt.show()


def label_stats_perdataset(json_list, tokenizer, name):
    """Plots Token Frequencies for each data set"""
    plot_text = [x["text"] for x in json_list]
    tokens = [str(tokenizer(text)).split() for text in plot_text]
    tokens = [x for sublist in tokens for x in sublist]
    counts = Counter(tokens)
    sorted_counts = dict(sorted(counts.items(), key=operator.itemgetter(1), reverse=True))
    first_100 = {k: sorted_counts[k] for k in list(sorted_counts)[:100]}
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(first_100)), list(first_100.values()), align='center')
    plt.xticks(range(len(first_100)), list(first_100.keys()), rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.title(f"Token frequencies for {name} Set")
    plt.xlabel("tokens")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()


def label_stats_perclass(json_list, tokenizer, name):
    """Plots Frequency of each token per Class in each data set"""
    all_ents = seperate_class_tokens(json_list, tokenizer)
    counts = {k: sorted(Counter(v).items(), reverse=True) for k, v in all_ents.items()}
    for k1, v1 in counts.items():
        sorted_counts = dict(sorted(v1, key=operator.itemgetter(1), reverse=True))
        first_100 = {k: sorted_counts[k] for k in list(sorted_counts)[:20]}
        plt.figure(figsize=(20, 10))
        plt.bar(range(len(first_100)), list(first_100.values()), align='center')
        plt.xticks(range(len(first_100)), list(first_100.keys()), rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title(f"Token frequencies for {k1} Class on {name} Set")
        plt.xlabel("tokens")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.show()


def plot_wordcloud(json_list, tokenizer, name):
    """Plots Frequency as a WordCloud of each token per Class in each data set"""
    all_ents = seperate_class_tokens(json_list, tokenizer)
    counts = {k: Counter(v) for k, v in all_ents.items()}
    for k1, v1 in counts.items():
        keys = v1.keys()
        joined = ' '.join(keys)
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(joined)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"{k1}")
        # wordcloud.to_file("img/first_review.png") #save
        plt.show()
