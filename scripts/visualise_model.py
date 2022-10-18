import spacy_streamlit

# to run use: streamlit run visualise_model.py

models = ["en_spacy_table_ner", "en_spacy_table_ner_AL"]
default_text = "The clearance (CLr) of midazolam (MZ) resulted in CLr=0.4"
spacy_streamlit.visualize(models, default_text, visualizers=["ner"])
