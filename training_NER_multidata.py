import logging
import os
import random

import numpy as np
import pickle
import torch
from torch import nn
from torch.utils import data
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup

from architectures.BERT_model_multidata import BERT_model_multidata
from architectures.architectures_utils_multidata import train_bert_model_multidata, predict_multidata
from eval_metrics.evaluateNER import EvaluateNER
from utils.BatcherBERT_multidata import BatcherBERT_multidata
from utils.ReadNERMultiData import MultiDataset
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments(parser):
	parser.add_argument("--file_extension", type=str, default="txt")
	parser.add_argument("--experiment", type=str, default="baseline")
	parser.add_argument("--model_saving_path", type=str, default="")
	parser.add_argument("--train_file", type=str, default="train")
	parser.add_argument("--data_path", type=str, default="train")
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--force_size", action="store_true")
	parser.add_argument("--special_labels", action="store_true")
	parser.add_argument("--masking_percentage", type=float, default=0.0)
	parser.add_argument("--seed", type=int, default=12)  # 12 seems to be a good seed
	parser.add_argument("--early_stop", type=int, default=0)
	parser.add_argument("--tags2use", type=str, default="NER_IOB2")
	parser.add_argument("--separator", type=str, default=" ")
	parser.add_argument("--test_file", type=str, default="test")
	parser.add_argument("--dev_file", type=str, default="valid")
	parser.add_argument("--predict", type=str, default="None",
						choices=["None", "Train", "Test", "Dev", "All", "DevTest"])
	parser.add_argument("--crf", action="store_true")
	parser.add_argument("--uppercase", action="store_true")
	parser.add_argument("--lr", type=float, default=2e-5)
	parser.add_argument("--epsilon", type=float, default=1e-8)
	parser.add_argument("--bert_model", type=str, default="bert-base-cased")
	parser.add_argument("--token_col", type=int, default=0)
	parser.add_argument("--ner_col", type=int, default=3)
	parser.add_argument("--evaluate", action="store_true", help="It will be activated if training is activated")
	parser.add_argument("--mask_entities", action="store_true",
						help="If activated, instead of masking based on tokens, it will mask based on entities.")
	parser.add_argument("--train_batch_size", type=int, default=32)
	parser.add_argument("--comment_line", type=str, default="-DOCSTART")
	parser.add_argument("--sequence_size", type=int, default=128)
	parser.add_argument("--multi_gpu", action="store_true")
	parser.add_argument("--fullwords_mask", action="store_false", default=None)
	parser.add_argument("--redundant_uppercase", action="store_false", default=None)
	parser.add_argument("--bert_hidden_size", type=int, default=768)
	parser.add_argument("--no_dev", action="store_true")
	parser.add_argument("--multidata_model", type=str, default="None", choices=["None", "Daume", "All"])
	parser.add_argument("--annotate_as_dataset", type=int, help="Indicate the ID of the dataset that will be used for annoting.", default=-1)
	parser.add_argument("--output_saving_path", type=str, default=None, help="If not set, the path will be the same where the model is saved")
	parser.add_argument("--predict_boundaries", action="store_true")
	parser.add_argument("--uppercase_percentage", type=float, default=0.0)
	parser.add_argument("--no_test", action="store_true")

	args = parser.parse_args()
	for k in args.__dict__:
		print(k + ": " + str(args.__dict__[k]))

	return args


parser = argparse.ArgumentParser()
opt = parse_arguments(parser)

seed = opt.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

extension = opt.file_extension
columns = None
masking = False
mask_entities = False
fullwords_mask = False
do_predictions_on = []
if opt.predict != "None":
	if opt.predict == "All":
		do_predictions_on = ["Dev", "Test", "Train"]
		if opt.extra_test_file:
			do_predictions_on.append("ExtraTest")
	elif opt.predict == "DevTest":
		do_predictions_on = ["Dev", "Test"]
	else:
		do_predictions_on = [opt.predict]

if opt.train:
	opt.evaluate = True

if opt.masking_percentage > 0:
	masking = True
	mask_entities = opt.mask_entities
	fullwords_mask = opt.fullwords_mask

columns = {
	opt.token_col: 'tokens',
	opt.ner_col: 'NER_IOB'
}

experiment_name = opt.experiment
tags2use = opt.tags2use
model_saving_path = opt.model_saving_path
train_file = opt.train_file
data_path = opt.data_path
if opt.separator == "\\t":
	opt.separator = "\t"

if opt.no_dev:
	opt.early_stop = 0

if not os.path.exists(f"{model_saving_path}/{experiment_name}/"):
	os.makedirs(f"{model_saving_path}/{experiment_name}/")

if opt.output_saving_path is not None:
	output_saving_path = f"{opt.output_saving_path}/"
	if not os.path.exists(output_saving_path):
		os.makedirs(output_saving_path)
else:
	output_saving_path = f"{model_saving_path}/{experiment_name}/"

dataset_config = {
	'columns': columns,
	'columnsSeparator': opt.separator,
	'basePath': data_path,
	'dataType': {},
	'commentSymbol': [opt.comment_line],
	'specialLabels': opt.special_labels
}

if not opt.no_test:
	dataset_config["dataType"]["Test"] = {
		'prefix': opt.test_file,
		'extension': extension,
		'labelsAsTraining': False
	}

if not opt.no_dev:
	dataset_config["dataType"]["Dev"] = {
		'prefix': opt.dev_file,
		'extension': extension,
		'labelsAsTraining': False
	}

if opt.train:

	dataset_config["dataType"]["Train"] = {
		'prefix': train_file,
		'extension': extension,
		'labelsAsTraining': True
	}

	bert_base_model = opt.bert_model
	tokenizer = BertTokenizer.from_pretrained(bert_base_model, do_lower_case=False)

	data_processor = MultiDataset(dataset_config, tokenizer=tokenizer)
	mapping = data_processor.getMapping()

	add_tokens = data_processor.getTokensToAdd()

	resize_vocab_size = None

	num_added_tokens = 0
	if len(add_tokens) > 0:
		print(f"Adding {len(add_tokens)} tokens to Tokenizer vocabulary")
		num_added_tokens = tokenizer.add_tokens(add_tokens)
		#We need to add this to the mappings as the tokenizer of Transformers 2.11.0 do not store the added tokens
		mapping["new_tokens"] = add_tokens
	if opt.uppercase:
		special_tokens_dict = {'additional_special_tokens': ['[UP]', '[up]']}

		num_added_tokens += tokenizer.add_special_tokens(special_tokens_dict)
		#We need to add this to the mappings as the tokenizer of Transformers 2.11.0 do not store the added tokens
		mapping["new_special_tokens"] = special_tokens_dict

	if num_added_tokens > 0:
		resize_vocab_size = len(tokenizer)

	tokenizer.save_vocabulary(f"{model_saving_path}/{experiment_name}/")

	with open(f"{model_saving_path}/{experiment_name}/mapping.pkl", 'wb') as mapping_file:
		pickle.dump(mapping, mapping_file, -1)

	dataset = data_processor.getProcessedData()
	del data_processor

	with open(f"{model_saving_path}/{experiment_name}/params-{experiment_name}.txt", "w") as output_file:
		output_file.write(f"seed: {seed}\n")
		output_file.write(f"file_extension: {extension}\n")
		output_file.write(f"experiment_name: {experiment_name}\n")
		output_file.write(f"tags2use: {tags2use}\n")
		output_file.write(f"special_labesl: {opt.special_labels}\n")
		output_file.write(f"force_size: {opt.force_size}\n")
		output_file.write(f"train_file: {train_file}\n")
		output_file.write(f"masking_percentage: {opt.masking_percentage}\n")
		output_file.write(f"seed: {seed}\n")
		output_file.write(f"epochs: {opt.epochs}\n")
		output_file.write(f"early_stop: {opt.early_stop}\n")
		output_file.write(f"separator: {opt.separator}\n")
		output_file.write(f"test_file: {opt.test_file}\n")
		output_file.write(f"dev_file: {opt.dev_file}\n")
		output_file.write(f"crf: {opt.crf}\n")
		output_file.write(f"uppercase: {opt.uppercase}\n")
		output_file.write(f"lr: {opt.lr}\n")
		output_file.write(f"epsilon: {opt.epsilon}\n")
		output_file.write(f"bert_model: {opt.bert_model}\n")
		output_file.write(f"mask_entities: {mask_entities}\n")
		output_file.write(f"train_batch_size: {opt.train_batch_size}\n")
		output_file.write(f"comment_line: {opt.comment_line}\n")
		output_file.write(f"sequence_size: {opt.sequence_size}\n")
		output_file.write(f"multi_gpu: {opt.multi_gpu}\n")
		output_file.write(f"fullwords_mask: {opt.fullwords_mask}\n")
		output_file.write(f"redundant_uppercase: {opt.redundant_uppercase}\n")
		output_file.write(f"bert_hidden_size: {opt.bert_hidden_size}\n")
		output_file.write(f"no_dev: {opt.no_dev}\n")
		output_file.write(f"multidata_model: {opt.multidata_model}\n")
		output_file.write(f"training_datasets: {dataset['Train_files']}\n")
		output_file.write(f"predict_boundaries: {opt.predict_boundaries}\n")
		output_file.write(f"uppercase_percentage: {opt.uppercase_percentage}\n")
else:
	with open(f"{model_saving_path}/{experiment_name}/mapping.pkl", 'rb') as mapping_file:
		mapping = pickle.load(mapping_file)
	data_processor = MultiDataset(dataset_config, predifined_mapping=mapping)
	tokenizer = BertTokenizer.from_pretrained(f"{model_saving_path}/{experiment_name}/", do_lower_case=False)
	#It is mandatory to add the tokens in the same order than before
	if "new_tokens" in mapping:
		tokenizer.add_tokens(mapping["new_tokens"])
	if "new_special_tokens" in mapping:
		tokenizer.add_special_tokens(mapping["new_special_tokens"])

	dataset = data_processor.getProcessedData()
	del data_processor

inverse_tagset = dict(map(reversed, mapping[tags2use].items()))
evaluateNER = EvaluateNER(inverse_tagset)


def Train():
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	type_crf_constraints = None
	if opt.crf:
		if tags2use in ["NER_IOB2", "NER_IOBA"]:
			type_crf_constraints = "BIO"
		elif tags2use in ["NER_IOBES", "NER_IOBESA"]:
			type_crf_constraints = "BIOES"

	md_model = False
	md_da = False
	md_number = 0
	if opt.multidata_model in ["Daume", "All"]:
		md_model = True
		if opt.multidata_model == "Daume":
			md_number = len(dataset["Train"])
			md_da = True

	boundaries_labels = None
	if opt.predict_boundaries:
		if tags2use == "NER_IOB2":
			boundaries_labels = mapping["NER_IOBA"]
		if tags2use == "NER_IOBES":
			boundaries_labels = mapping["NER_IOBESA"]

	labels_to_use = mapping[tags2use]

	bert_config = BertConfig.from_pretrained(bert_base_model, num_labels=len(labels_to_use) + 1,
											 finetuning_task=experiment_name,
											 task_specific_params={"crf": opt.crf,
																   "type_crf_constraints": type_crf_constraints,
																   "predict_masked": masking,
																   "bert_hidden_size": opt.bert_hidden_size,
																   "predict_boundaries": opt.predict_boundaries,
																   "boundaries_labels": boundaries_labels,
											 					   "md_model": md_model,
																   "md_number": md_number},
											 label2id=labels_to_use)

	params_train = {'batch_size': opt.train_batch_size,
					'shuffle': True}


	print("Processing train")
	training_batcher = BatcherBERT_multidata(dataset["Train"], tokenizer,
											 	tags_field=tags2use,
												max_length=opt.sequence_size,
												tagset=mapping[tags2use],
												force_size=opt.force_size,
												special_labels=opt.special_labels,
												mask_percentage=opt.masking_percentage,
												uppercase=opt.uppercase,
												fullwords_mask=fullwords_mask,
												redundant_uppercase=opt.redundant_uppercase,
												boundaries_dict=boundaries_labels,
												predict_boundaries=opt.predict_boundaries,
												as_da=md_da,
											 	uppercase_percentge=opt.uppercase_percentage
											 	)

	training_batcher.createBatches()

	if opt.no_dev:
		dev_generator = None
		dev_aligner = None
	else:
		params_dev = {'batch_size': 8,
					  'shuffle': False}

		print("Processing dev")
		dev_set = BatcherBERT_multidata(dataset["Dev"], tokenizer, tags_field=tags2use,
							  max_length=opt.sequence_size, test=True, tagset=mapping[tags2use], force_size=opt.force_size,
							  special_labels=opt.special_labels, uppercase=opt.uppercase,
							  redundant_uppercase=opt.redundant_uppercase, as_da=md_da)

		dev_generator = data.DataLoader(dev_set, **params_dev, collate_fn=dev_set.collate_fn)

		dev_aligner = dev_set.getAligner()

	model = BERT_model_multidata.from_pretrained(bert_base_model,
												   from_tf=False,
												   config=bert_config)

	if resize_vocab_size is not None:
		model.resize_embedding_and_fc(resize_vocab_size)

	if opt.multi_gpu and torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs")
		model = nn.DataParallel(model)
		#model = DataParallelModel(model)
		opt.multi_gpu = True
	else:
		opt.multi_gpu = False


	num_train_epochs = opt.epochs
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.weight']
	weight_decay = 0.01
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(
			nd in n for nd in no_decay)], 'weight_decay': weight_decay},
		{'params': [p for n, p in param_optimizer if any(
			nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]

	warmup_proportion = 0.1
	gradient_accumulation_steps = 1
	learning_rate = opt.lr
	num_train_optimization_steps = int(
		len(training_batcher) / params_train['batch_size'] / gradient_accumulation_steps) * num_train_epochs
	warmup_steps = int(warmup_proportion * num_train_optimization_steps)
	optimizer = AdamW(optimizer_grouped_parameters,
					  lr=learning_rate, eps=opt.epsilon,
					  correct_bias=True)  # If we would like to replicate BERT, we need to set the compensate_bias as false
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
												num_training_steps=num_train_optimization_steps)

	train_bert_model_multidata(model, experiment_name, num_train_epochs,
						 optimizer, scheduler, training_batcher, params_train, dev_generator, evaluateNER.calculate,
						 model_saving_path, use_gpu=True, masking=masking, early_stop=opt.early_stop, bert_hidden_size=opt.bert_hidden_size,
						 dev_aligner=dev_aligner, multi_gpu=opt.multi_gpu, uppercase_percentage=opt.uppercase_percentage)


def printCommentLines(output_file, lines_array):
	for line in lines_array:
		output_file.write(f"{line}\n")


def testAndPredict(model, data_split, data_id, tagged=False, md_da=False):
	params_test = {'batch_size': 8,
				   'shuffle': False}

	override_da_index = data_id
	if opt.annotate_as_dataset >= 0:
		override_da_index = opt.annotate_as_dataset

	override_da_index += 1

	tagset = mapping[tags2use]

	output_file_name = dataset[f"{data_split}_files"][data_id]

	print(f"Batching {output_file_name}")

	test_set = BatcherBERT_multidata([dataset[data_split][data_id]], tokenizer, tags_field=tags2use,
						   max_length=opt.sequence_size, test=True, tagset=tagset, force_size=opt.force_size,
						   special_labels=opt.special_labels, uppercase=opt.uppercase,
						   redundant_uppercase=opt.redundant_uppercase, as_da=md_da, override_da_index=override_da_index)

	test_generator = data.DataLoader(test_set, **params_test, collate_fn=test_set.collate_fn)

	print(f"Processing model on {output_file_name}")

	if tagged:
		predictions, _, report = predict_multidata(model, test_generator, tagged=True,
														evaluation_function=evaluateNER.calculate, use_gpu=True,
														test_aligner=test_set.getAligner(), multi_gpu=opt.multi_gpu,
														bert_hidden_size=opt.bert_hidden_size)
		with open(f"{output_saving_path}/{output_file_name}_{experiment_name}-results.txt", "w") as output_file:
			output_file.write(report)
			output_file.write("\n")
	else:
		predictions = predict_multidata(model, test_generator, tagged=False,
										evaluation_function=evaluateNER.calculate,
										use_gpu=True,
										test_aligner=test_set.getAligner(),
										multi_gpu=opt.multi_gpu,
										bert_hidden_size=opt.bert_hidden_size)

	with open(f"{output_saving_path}/{output_file_name}_{experiment_name}-predictions.txt", "w") as output_file:
		if -1 in dataset[f"{data_split}_comments"][data_id]:
			printCommentLines(output_file, dataset[f"{data_split}_comments"][data_id][-1])
			output_file.write("\n")

		for sentence_id, sentence in enumerate(dataset[data_split][data_id]):
			assert (len(predictions[sentence_id]) == len(sentence["tokens"]))
			for (before, token, middle, predicted_tag, after) in zip(sentence["colsBefore"], sentence["tokens"],
																	 sentence["colsMiddle"],
																	 predictions[sentence_id],
																	 sentence["colsAfter"]):
				if not tagged:
					if predicted_tag not in inverse_tagset:
						predicted_tag = "O"
					else:
						predicted_tag = inverse_tagset[predicted_tag]

				if before is not None:
					output_file.write(f"{before}{opt.separator}")
				output_file.write(f"{token}{opt.separator}")
				if middle is not None:
					output_file.write(f"{middle}{opt.separator}")
				output_file.write(f"{predicted_tag}")
				if after is not None:
					output_file.write(f"{opt.separator}{after}")
				output_file.write("\n")
			output_file.write(f"\n")
			if sentence_id in dataset[f"{data_split}_comments"][data_id]:
				printCommentLines(output_file, dataset[f"{data_split}_comments"][data_id][sentence_id])
				output_file.write("\n")


if opt.train:
	Train()

if opt.evaluate or len(do_predictions_on) > 0:

	model = BERT_model_multidata.from_pretrained(f"{model_saving_path}/{experiment_name}/", from_tf=False)

	if opt.multi_gpu and torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs")
		model = nn.DataParallel(model)
		opt.multi_gpu = True
	else:
		opt.multi_gpu = False

	md_da = False
	if opt.multidata_model == "Daume":
		md_da = True

	for data_split in do_predictions_on:
		if data_split == "Train":
			continue
		for i in range(len(dataset[data_split])):
			testAndPredict(model, data_split, i, tagged=opt.evaluate, md_da=md_da)
