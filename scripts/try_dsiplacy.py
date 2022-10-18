import spacy
from spacy import displacy

nlp = spacy.load("en_spacy_table_ner")

text = """PK Parameters of dADT in plasma and blood following treatment with 100 mg tablets tribendimidine"""

"""intravenous infusion (n = 4)"""
"""AUC0‐∞ (μg·h/mL)"""
"""The AUC(0–t) and Cmax of midazolam (n = 5)"""
"""Cmax (μg/mL)"""


doc = nlp(text)
displacy.serve(doc, style="ent", options={"ents": ["Dose/Route", "Num_Sub", "Num_Value", "PK", "Sample_Type", "Units", "Chem/Drug", "Measure_Type"], "colors": {"Dose/Route": "#00fdc8", "Num_Sub": "#f6cd4c", "Num_Value": "#eeeeee", "PK": "#ab9df2", "Sample_Type": "#78909c", "Units": "#f6cd4c", "Chem/Drug": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}})
a=1
