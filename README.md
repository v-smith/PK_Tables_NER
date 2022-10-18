# PK Tables NER

This project provides scripts and commands to train NER models in spacy to detect Pharmacokinetic Parameter and Covariate Mentions in the text of table cells, for tables from the PubMed Open Access Subset. 

## Resources
* Train SpaCy NER model from scratch: https://spacy.io/usage/training, https://spacy.io/api/cli#train
* Config files: https://github.com/allenai/scispacy/tree/main/configs

Preprocess the data into the appropriate "Doc" format for SpaCy
```angular2html
#using spacy_convert.py (allows custom tokenizer and converts annotations) 
python spacy_convert.py --input-path ../data/split_data/train.jsonl --output-path ../data/spacy/train.spacy/
python spacy_convert.py --input-path ../data/split_data/dev.jsonl --output-path ../data/spacy/dev.spacy/
#using preprocess.py 
python preprocess.py ../data/final_reviewed_out/final_reviewed_ner_trial.jsonl ./train.spacy
```

Run the training
```angular2html
python -m spacy train ./config/config_ner_spacy.cfg --paths.train ./data/spacy/train.spacy --paths.dev ./data/spacy/dev.spacy --output ./data/trained_models --code ./tables_ner/character_tokenizer.py
python -m spacy train ./config/config_ner_spacy.cfg --paths.train ./data/spacy/train.spacy --paths.dev ./data/spacy/dev.spacy --output ./data/trained_models --code ./tables_ner/partial_split_tokenizer.py
python -m spacy train ./config/config_ner_spacy.cfg --paths.train ./data/spacy/train.spacy --paths.dev ./data/spacy/dev.spacy --output ./data/trained_models --code ./tables_ner/whitespace_tokenizer.py

# model_vs_annotator data 
python -m spacy train ./config/config_ner_spacy.cfg --paths.train ./data/spacy2/train2.spacy --paths.dev ./data/spacy2/dev2.spacy --output ./data/trained_models2 --code ./tables_ner/partial_split_tokenizer.py
#model_vs_annotator data w/o Pop and Dem
python -m spacy train ./config/config_ner_spacy_trial.cfg --paths.train ./data/spacy2/train_noPopDem.spacy --paths.dev ./data/spacy2/dev_noPopDem.spacy --output ./data/trained_models2/NoPopDem --code ./table_ner/partial_split_tokenizer.py
#model_vs_annotator data combine Pop and Dem
python -m spacy train ./config/config_ner_spacy_trial.cfg --paths.train ./data/spacy/train_noPopDem.spacy --paths.dev ./data/spacy/dev_noPopDem.spacy --output ./data/trained_models2/NoPopDem --code ./table_ner/partial_split_tokenizer.py
python -m spacy train ./config/config_ner_spacy_earlystop.cfg --paths.train ./data/spacy/train_PopDemCombo.spacy --paths.dev ./data/spacy/dev_PopDemCombo.spacy --output ./data/trained_models2/PopDemCombo --code ./table_ner/partial_split_tokenizer.py --gpu-id 0

#AL
python -m spacy train ./config/config_ner_spacy_earlystop.cfg --paths.train ./data/spacy/trainAL.spacy --paths.dev ./data/spacy/dev.spacy --output ./data/trained_models/AL --code ./table_ner/partial_split_tokenizer.py
```

Eval model and export metrics
```
python -m spacy evaluate ./data/trained_models/model-best ./data/spacy/test.spacy --output data/run_outputs/metrics.json --code tables_ner/partial_split_tokenizer.py
python -m spacy evaluate ./data/trained_models2/model-last ./data/spacy/test.spacy --output data/run_outputs/modelvsannotator_metrics.json --code tables_ner/partial_split_tokenizer.py
```

Training Curve
```angular2html
prodigy train-curve ner hoax_conspiracy en_vectors_web_lg --init-tok2vec ./tok2vec_cd8_model289.bin --eval-split 0.3
```

## Package spacy model as trained model 
```angular2html
python -m spacy package ./data/trained_models2/PopDemCombo/model-best ./spacy_table_ner/packages --name spacy_table_ner --version 0.0.0 --force --code ./table_ner/partial_split_tokenizer.py
python -m spacy package ./data/trained_models/AL/model-best ./spacy_table_ner_AL/packages --name spacy_table_ner_AL --version 0.0.0 --force --code ./table_ner/partial_split_tokenizer.py
#cd to package directory 
python setup.py sdist
cd dist 
pip install en_spacy_table_ner_AL-0.0.0.tar.gz
```
## Spacy Loggers 
https://github.com/explosion/spacy-loggers#wandblogger 
https://spacy.io/usage/training

## Optmizers and Schedulers 
https://thinc.ai/docs/api-schedules
https://thinc.ai/docs/api-optimizers

## Prodigy Ner Teach (Active learning with model in the loop)
N.B. its binary so one label at a time
```angular2html
prodigy ner.teach AL_SampleType en_spacy_table_ner ./data/ActiveLearning/parsed_remaining_ner_relevant_forclass.jsonl --label Sample_Type
prodigy db-out AL_SampleType ./data/ActiveLearning/ner_teach_output
prodigy table-ner data/ActiveLearning/ner_teach_output/AL_SampleType_model-labelled.jsonl AL_SampleType_reviewed -F recipes/table-ner.py
prodigy db-out AL_SampleType_reviewed ./data/ActiveLearning/ner_reviewed_output

#custom -NB lacking tables in some cases, can do one or many labels at a time (NB seems to mainly focus on captions)- interesting to remove these and classify tables alone
prodigy table_teach_ner population_AL en_spacy_table_ner ./data/ActiveLearning/parsed_remaining_ner_relevant_forclass.jsonl -l PK,Units,Measure_Type,Num_Value,Chem/Drug,Dose/Route,Num_Sub,Sample_Type -F recipes/custom_ner_teach.py
```
