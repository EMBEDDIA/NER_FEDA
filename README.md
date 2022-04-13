# Frustratingly Easy Domain Adaptation for NER

This is the code related to the paper [Using a frustratingly easy domain and tagset adaptation for creating slavic named entity recognition systems](https://www.aclweb.org/anthology/2021.bsnlp-1.12/). It is a NER system which applies a Frustrantinglu Easy Domain Adaptation model not only to increase the data available for training the models but also for changing the possible tagset used to annotate.

Each model (see below where to find them) can provide multiple and different tagsets. Do you want need products? or events? or time? Just activate in the configuration annotate_as_dataset and your data will be annotated according to specific tagsets.

# Where to find the models?

You can find all the trained models in [HuggingFace](https://huggingface.co/creat89). The models are NER_FEDA\_*. Although the models were trained mostly using Slavic languages, most of the models are multilingual as they fine-tuned LabSE.

# How to train a model?

You can use this script:

```
python training_NER_multidata.py \
		--experiment  $EXPERIMENT\
		--model_saving_path $OUTPUT_DIR \
		--data_path $DATASET_PATH/$LANGUAGE \
		--special_labels \
		--seed 12 \
		--epochs 20 \
		--early_stop 2 \
		--tags2use "NER_IOBES" \
		--crf \
		--dev_file "dev_*" \
		--train_file "train_*" \
		--no_test \
		--file_extension "txt" \
		--ner_col 2 \
		--bert_model $BERT \
		--separator "\t" \
		--multidata_model "Daume" \
		--train \
		--predict "Dev" \
		--uppercase \
		--uppercase_percentage 0.05 \
		--train_batch_size $BATCH_NORMAL \
		--masking_percentage 0.25
```

# How to run the code and get predictions?

Here is an example of th code:

```
python training_NER_multidata.py \
			--experiment  $EXPERIMENT \
			--model_saving_path $MODEL_PATH \
			--data_path "$DATASET_PATH/annotated/${TOPIC}/${LANGUAGE_SET}" \
			--output_saving_path "$DATASET_PATH/predictions/$EXPERIMENT/$TOPIC/$LANGUAGE_SET" \
			--special_labels \
			--tags2use "NER_IOBES" \
			--crf \
			--no_dev \
			--test_file "*" \
			--file_extension "lacd" \
			--ner_col 2 \
			--separator "\t" \
			--multidata_model "Daume" \
			--predict "Test" \
			--uppercase \
			--annotate_as_dataset 1
```

Where annotate_as_dataset indicates which tagset (based on the model used) should be used. The ID of the dataset starts at 0, and 1 means in all models SlavNER 2021. See the HuggingFace models to see which tagsets are available. Use the uppercase flag if the models was trained with uppercase process.

# How to cite this work?

Please cite this work using the following paper:
```
@inproceedings{cabrera-diego-etal-2021-using,
    title = "Using a Frustratingly Easy Domain and Tagset Adaptation for Creating {S}lavic Named Entity Recognition Systems",
    author = "Cabrera-Diego, Luis Adrián  and
      Moreno, Jose G.  and
      Doucet, Antoine",
    booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bsnlp-1.12",
    pages = "98--104"
}

```
# Have you found a bug

We cleaned the code before making it public, if you find any bug, please, let us know by raising an issue.

# Parent project

This work is is result of the European Union H2020 Project [Embeddia](http://embeddia.eu/). Embeddia is a project that creates NLP tools that focuses on European under-represented languages and that has for objective to improve the accessibility of these tools to the general public and to media enterprises. Visit [Embeddia's Github](https://github.com/orgs/EMBEDDIA/) to discover more NLP tools and models created within this project.
