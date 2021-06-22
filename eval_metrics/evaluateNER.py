import itertools
import re

from seqeval.metrics import classification_report, f1_score as f1_metric

#from eval_metrics.DavidNER import Evaluator


class EvaluateNER:

	def __init__(self, inverse_tagset, use_david=False):
		self.__inverse_tagset = inverse_tagset
		self.__use_david = use_david
		self.__david = None
		if self.__use_david:
			tags = list(self.__inverse_tagset.values())
			types = []
			for tag in tags:
				if tag.startswith("[") or tag == "O":
					continue
				elif tag[2:] not in types:
					types.append(tag[2:])
			#self.__david = Evaluator(types)

	def __revertTags(self, sentences):
		for id_, tags in enumerate(sentences):
			for i, tag in enumerate(tags):
				if tag in self.__inverse_tagset:
					label = self.__inverse_tagset[tag]
					if label.startswith("["):	#Trigger, or SEP or CLS
						label = "O"
					tags[i] = label
				else:
					tags[i] = "O"

		return sentences

	def calculate(self, predictions, gold_standard, convert_predicted_tags=True):
		assert (len(predictions) == len(gold_standard))

		if convert_predicted_tags:
			predictions = self.__revertTags(predictions)
		gold_standard = self.__revertTags(gold_standard)

		y_true = list(itertools.chain(*gold_standard))
		y_pred = list(itertools.chain(*predictions))

		report = classification_report(y_true, y_pred, digits=4)
		report = re.sub("\n\n", "\n", report)
		report = re.sub("^ +", "", report)
		report = re.sub("\n +", "\n", report)
		report = re.sub("  +", "\t", report)
		report = "results\t" + report

		david = None
		#if self.__use_david:
		#	david = self.__david.evaluate(gold_standard, predictions)

		return f1_metric(y_true, y_pred), report, david
