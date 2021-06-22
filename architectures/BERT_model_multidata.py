import torch

from torch import nn
from transformers import BertForTokenClassification, BertModel

from transformers.modeling_bert import BertOnlyMLMHead

from architectures.crf import allowed_transitions, ConditionalRandomField


class BERT_model_multidata(BertForTokenClassification):

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.num_boundaries_labels = 0

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		self.use_crf = False
		self.crf = None
		self.crf_constrains = None
		self.crf_boundaries = None

		self.classifier = None
		self.classifier_boundaries = None

		self.predict_masked = False
		self.predict_boundaries = False

		self.md_model = False
		self.md_number = 0
		self.md_layers = None

		self.biaffine = False

		if hasattr(config, "task_specific_params"):
			other_config = config.task_specific_params
			if other_config["crf"]:
				self.use_crf = True
				self.crf_constrains = other_config["type_crf_constraints"]
				self.crf = self.__crf_model(config.label2id.items())
			if other_config["predict_boundaries"] and not other_config["biaffine"]:
				self.predict_boundaries = True
				self.num_boundaries_labels = len(other_config["boundaries_labels"]) + 1
			if other_config["predict_masked"]:
				self.predict_masked = True
				self.classifier_masked = BertOnlyMLMHead(config)
			if other_config["md_model"]:
				self.md_model = True
				self.md_number = other_config["md_number"]
				if self.md_number > 1:
					self.md_layers = nn.ModuleList()
					self.classifier = nn.ModuleList()
					if self.predict_boundaries:
						self.classifier_boundaries = nn.ModuleList()
					for _ in range(self.md_number+1):
						md_layer, classifier, classifier_boundaries = self.__md_model(config.hidden_size, config.hidden_dropout_prob)
						self.md_layers.append(md_layer)
						self.classifier.append(classifier)
						if self.predict_boundaries:
							self.classifier_boundaries.append(classifier_boundaries)
				else:
					md_layer, classifier, classifier_boundaries = self.__md_model(config.hidden_size, config.hidden_dropout_prob)
					self.md_layers = md_layer
					self.classifier = classifier
					self.classifier_boundaries = classifier_boundaries
			elif other_config["biaffine"]:
				#self.predict_boundaries = True
				self.biaffine = other_config["biaffine"]
				#self.classifier_boundaries = Biaffine(config.hidden_size, 512, second_bias=False)
				#As the arcs go only to the right, the first bias is not used
				if other_config["predict_boundaries"]:
					self.predict_boundaries = True
					self.classifier_boundaries = Biaffine(config.hidden_size, 512, second_bias=False)
				self.classifier = Biaffine(config.hidden_size, 128, output_size=config.num_labels)
			else:
				self.classifier = nn.Linear(config.hidden_size, config.num_labels)
				self.classifier_boundaries = self.__boundaries_model(config.hidden_size)
			if self.predict_boundaries and self.use_crf and not self.biaffine:
				self.crf_boundaries = self.__crf_model(other_config["boundaries_labels"].items())
		self.init_weights()

	def __md_model(self, hidden_size, dropout):
		md_layers = nn.Sequential(nn.Linear(hidden_size, 512),
							  		nn.ReLU(),
							  		nn.Dropout(dropout))
		classifier = nn.Linear(512, self.num_labels)
		classifier_boundaries = self.__boundaries_model(512)
		return md_layers, classifier, classifier_boundaries

	def __crf_model(self, labels):
		crf_constraints = allowed_transitions(self.crf_constrains, dict(map(reversed, labels)))
		return ConditionalRandomField(len(labels)+1, constraints=crf_constraints)

	def __boundaries_model(self, input_size):
		classifier_boundaries = None
		if self.predict_boundaries:
			classifier_boundaries = nn.Linear(input_size, self.num_boundaries_labels)
		return classifier_boundaries

	def __forward__md_daume(self, valid_output_tokens, da_sets, specialized_logits, specialized_boundaries_logits):
		inter_general_logits = self.md_layers[0](valid_output_tokens)
		general_logits = self.classifier[0](inter_general_logits)
		general_boundaries_logits = None
		if self.predict_boundaries and specialized_boundaries_logits is not None:
			general_boundaries_logits = self.classifier_boundaries[0](inter_general_logits)
		del inter_general_logits
		for i, da_set in enumerate(da_sets):
			inter_specialized_logits = self.md_layers[da_set.item()](valid_output_tokens[i])
			specialized_logits[i] = self.classifier[da_set.item()](inter_specialized_logits)
			if self.predict_boundaries and specialized_boundaries_logits is not None:
				specialized_boundaries_logits[i] = self.classifier_boundaries[da_set.item()](inter_specialized_logits)

		del inter_specialized_logits

		logits = torch.stack((general_logits, specialized_logits), 2).sum(2)

		del general_logits
		del specialized_logits

		logits_boundaries = None
		if self.predict_boundaries and specialized_boundaries_logits is not None:
			logits_boundaries = torch.stack((general_boundaries_logits, specialized_boundaries_logits), 2).sum(2)

			del general_boundaries_logits
			del specialized_boundaries_logits

		return logits, logits_boundaries

	def __forward_masked_tokens(self, original_sequence_output, valid_output_predict, lm_mask, lm_labels):
		batch_size, max_len, _ = original_sequence_output.shape
		for i in range(batch_size):
			jj = -1
			for j in range(max_len):
				if lm_mask[i][j].item() == 1:
					jj += 1
					valid_output_predict[i][jj] = original_sequence_output[i][j]
		lm_sequence_output = self.dropout(valid_output_predict)
		del valid_output_predict
		del original_sequence_output

		lm_logits = self.classifier_masked(lm_sequence_output)

		del lm_sequence_output

		loss_lm_fct = nn.CrossEntropyLoss(ignore_index=0)

		active_logits = lm_logits.view(-1, self.config.vocab_size)
		active_labels = lm_labels.view(-1)
		loss_masks = loss_lm_fct(active_logits, active_labels)

		return loss_masks

	def __forward_bert(self, original_sequence_output, tokens_mask, valid_output_tokens):

		if tokens_mask is not None:
			batch_size, max_len, _ = original_sequence_output.shape
			for i in range(batch_size):
				jj = -1
				for j in range(max_len):
					if tokens_mask[i][j].item() == 1:
						jj += 1
						valid_output_tokens[i][jj] = original_sequence_output[i][j]
			valid_output_tokens = self.dropout(valid_output_tokens)
		else:
			valid_output_tokens = self.dropout(original_sequence_output)

		return valid_output_tokens

	def __forward_biaffine(self, valid_output_tokens, labels, labels_boundaries_set, labelling_mask, train=False):
		logits_boundaries = None
		if self.predict_boundaries:
			logits_boundaries = self.classifier_boundaries(valid_output_tokens)
		logits = self.classifier(valid_output_tokens)
		loss_boundaries = 0
		loss_gold_boundary = 0
		loss = 0
		if labels_boundaries_set is not None:

			labels_boundaries = labels_boundaries_set[0]
			labels_boundaries_min = labels_boundaries_set[1]
			labels_boundaries_max = labels_boundaries_set[2]

			loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

			active_tokens = labelling_mask == 1
			labelling_mask_2 = labelling_mask.unsqueeze(1)
			labelling_mask_2 = labelling_mask_2.repeat(1, 128, 1)

			active_tokens_2 = labelling_mask_2 == 0
			if labelling_mask is not None:

				if self.predict_boundaries:

					logits_boundaries[active_tokens_2] = float("-inf")

					# Convert into triangular matrix
					#triangular_mask = (torch.tril(torch.ones_like(logits_boundaries), diagonal=-1) == 1)
					#logits_boundaries[triangular_mask] = float("-inf")

					#Loss of boundaries
					active_logits_boundaries = logits_boundaries[active_tokens]
					active_labels_boundaries = labels_boundaries[active_tokens]
					loss_boundaries = loss_fct(active_logits_boundaries, active_labels_boundaries)

					#Loss with respect to the predicted boundary but actual label
					active_labels_boundaries = torch.argmax(nn.functional.log_softmax(active_logits_boundaries, dim=-1), dim=-1)

					active_labels_boundaries = active_labels_boundaries.unsqueeze(-1).repeat(1, self.num_labels)
					active_labels_boundaries = active_labels_boundaries.unsqueeze(1)
					active_logits = logits[active_tokens]
					active_logits = torch.gather(active_logits, 1, active_labels_boundaries).squeeze(1)
					active_labels = labels[active_tokens]
					loss = loss_fct(active_logits, active_labels)

					#Loss with respect to the correct boundary

					# active_labels_boundaries = active_labels_boundaries.unsqueeze(-1).repeat(1, self.num_labels)
					# active_labels_boundaries = active_labels_boundaries.unsqueeze(1)
					#
					# active_logits = logits[active_tokens]
					# active_logits = torch.gather(active_logits, 1, active_labels_boundaries).squeeze(1)
					# active_labels = labels[active_tokens]
					# loss_gold_boundary = loss_fct(active_logits, active_labels)

					#Loss with respect to the predicted boundary but if wrong prediction, uses O
					# active_predicted_boundaries = torch.argmax(nn.functional.log_softmax(active_logits_boundaries, dim=-1), dim=-1)
					#
					# set_other_label = torch.logical_or(labels_boundaries_min[active_tokens] > active_predicted_boundaries,
					# 									active_predicted_boundaries > labels_boundaries_max[active_tokens])
					#
					# correct_boundary_prediction = active_labels_boundaries == active_predicted_boundaries
					#
					# active_predicted_boundaries = active_predicted_boundaries.unsqueeze(-1).repeat(1, self.num_labels)
					# active_predicted_boundaries = active_predicted_boundaries.unsqueeze(1)
					# active_logits = logits[active_tokens]
					# active_logits = torch.gather(active_logits, 1, active_predicted_boundaries).squeeze(1)
					# active_labels = labels[active_tokens]
					# #One makes reference to the "O"
					# active_labels[set_other_label] = 1
					# loss = loss_fct(active_logits, active_labels)

					#Loss with predicted boundary but predict to tail
					# predicted_boundaries = torch.argmax(nn.functional.log_softmax(logits_boundaries, dim=-1), dim=-1)
					#
					# active_predicted_boundaries = predicted_boundaries[active_tokens]
					# #correct_boundary_prediction = active_labels_boundaries == active_predicted_boundaries
					#
					# labels_for_predicted = torch.gather(labels, 1, predicted_boundaries)
					#
					# active_predicted_boundaries = active_predicted_boundaries.unsqueeze(-1).repeat(1, self.num_labels)
					# active_predicted_boundaries = active_predicted_boundaries.unsqueeze(1)
					# active_logits = logits[active_tokens]
					# active_logits = torch.gather(active_logits, 1, active_predicted_boundaries).squeeze(1)
					#
					# active_labels = labels_for_predicted[active_tokens]
					#
					# loss = loss_fct(active_logits, active_labels)
					#possible_cases = set_other_label.shape[0]
					# #Conditional loss_gold_boundary
					# if torch.sum((set_other_label == True)) > 0.5*possible_cases:
					#
					# 	active_labels_boundaries = active_labels_boundaries.unsqueeze(-1).repeat(1, self.num_labels)
					# 	active_labels_boundaries = active_labels_boundaries.unsqueeze(1)
					#
					# 	active_logits = logits[active_tokens]
					# 	active_logits = torch.gather(active_logits, 1, active_labels_boundaries).squeeze(1)
					# 	active_labels = labels[active_tokens]
					# 	loss_gold_boundary = loss_fct(active_logits, active_labels)

					# Only on wrong loss_gold_boundary
					# active_labels_boundaries = active_labels_boundaries.unsqueeze(-1).repeat(1, self.num_labels)
					# active_labels_boundaries = active_labels_boundaries.unsqueeze(1)
					#
					# active_logits = logits[active_tokens]
					# active_logits = torch.gather(active_logits, 1, active_labels_boundaries).squeeze(1)
					# active_labels = labels[active_tokens]
					#
					# #Loss will disregard the entries that got correctly the boundary prediction
					# active_labels[correct_boundary_prediction] = -1
					# loss_gold_boundary = loss_fct(active_logits, active_labels)
					#
					# loss += loss_gold_boundary
				#If we're not predicting boundaries per se, but through the NE
				else:
					# Convert into triangular matrix
					triangular_mask = (torch.tril(torch.ones_like(labelling_mask_2), diagonal=-1) == 1)
					triangular_mask = triangular_mask[active_tokens]
					# We need to make add the padding to the other dimension
					active_tokens_2 = active_tokens_2[active_tokens]

					active_logits = logits[active_tokens]
					active_labels_boundaries = labels_boundaries[active_tokens]

					best_labels = torch.argmax(nn.functional.log_softmax(active_logits, dim=-1), dim=-1)
					active_boundaries = torch.gather(active_logits, -1, best_labels.unsqueeze(-1)).squeeze()
					active_boundaries[active_tokens_2] = float("-inf")
					active_boundaries[triangular_mask] = float("-inf")

					loss_boundaries = loss_fct(active_boundaries, active_labels_boundaries)

					active_predicted_boundaries = torch.argmax(nn.functional.log_softmax(active_boundaries, dim=-1), dim=-1)

					# Loss with respect to the predicted boundary

					set_other_label = torch.logical_or(
						labels_boundaries_min[active_tokens] > active_predicted_boundaries,
						active_predicted_boundaries > labels_boundaries_max[active_tokens])

					correct_boundary_prediction = active_labels_boundaries == active_predicted_boundaries

					active_predicted_boundaries = active_predicted_boundaries.unsqueeze(-1).repeat(1, self.num_labels)
					active_predicted_boundaries = active_predicted_boundaries.unsqueeze(1)
					active_logits = logits[active_tokens]
					active_logits = torch.gather(active_logits, 1, active_predicted_boundaries).squeeze(1)

					active_labels = labels[active_tokens]
					# One makes reference to the "O"
					active_labels[set_other_label] = 1

					loss = loss_fct(active_logits, active_labels)

					# Loss with respect to the correct boundary
					active_labels_boundaries = active_labels_boundaries.unsqueeze(-1).repeat(1, self.num_labels)
					active_labels_boundaries = active_labels_boundaries.unsqueeze(1)

					active_logits = logits[active_tokens]
					active_logits = torch.gather(active_logits, 1, active_labels_boundaries).squeeze(1)
					active_labels = labels[active_tokens]

					active_labels[correct_boundary_prediction] = -1
					loss_gold_boundary = loss_fct(active_logits, active_labels)

					# As the prediction of boundaries it is deative, we need to sum the loss here
					loss += loss_gold_boundary
					loss += loss_boundaries
					loss_boundaries = 0
			else:
				print("Unsupported")
				exit(1)
				#loss_boundaries = loss_fct(logits_boundaries.view(-1, max_len), labels_boundaries.view(-1))
		return [logits_boundaries, logits], loss_boundaries, loss


	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, labels_boundaries=None, tokens_mask=None,
				labelling_mask=None, lm_mask=None, lm_labels=None, valid_output_tokens=None, valid_output_predict=None,
				specialized_logits=None, specialized_boundaries_logits=None, da_sets=None, train=False):

		logits_boundaries = None
		loss = 0.0
		loss_boundaries = 0.0

		original_sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)

		del pooled_output

		self.__forward_bert(original_sequence_output, tokens_mask, valid_output_tokens)

		if self.md_number == 0:
			if self.biaffine:
				logits, loss_boundaries, loss = self.__forward_biaffine(valid_output_tokens, labels, labels_boundaries, labelling_mask, train=train)
			elif self.md_model:
				inter_logits = self.md_layers(valid_output_tokens)
				logits = self.classifier(inter_logits)
				if self.predict_boundaries and labels_boundaries is not None:
					logits_boundaries = self.classifier_boundaries(inter_logits)
				del inter_logits
			else:
				logits = self.classifier(valid_output_tokens)
				if self.predict_boundaries and labels_boundaries is not None:
					logits_boundaries = self.classifier_boundaries(valid_output_tokens)
		else:
			logits, logits_boundaries = self.__forward__md_daume(valid_output_tokens, da_sets, specialized_logits, specialized_boundaries_logits)

		del valid_output_tokens

		# Masked tokens
		loss_masks = 0.0
		if self.predict_masked and lm_labels is not None:
			loss_masks = self.__forward_masked_tokens(original_sequence_output, valid_output_predict, lm_mask, lm_labels)

		del original_sequence_output

		if labels is not None and not self.biaffine:
			if self.use_crf:
				loss = self.crf(logits, labels, labelling_mask)
				loss = loss.mean()

				if self.predict_boundaries and labels_boundaries is not None:
					loss_boundaries = self.crf_boundaries(logits_boundaries, labels_boundaries, labelling_mask)
					loss_boundaries = loss_boundaries.mean()
			else:
				loss_fct = nn.CrossEntropyLoss(ignore_index=0)
				if labelling_mask is not None:
					active_tokens = labelling_mask.view(-1) == 1
					active_logits = logits.view(-1, self.num_labels)[active_tokens]
					active_labels = labels.view(-1)[active_tokens]
					loss = loss_fct(active_logits, active_labels)

					if self.predict_boundaries and labels_boundaries is not None:
						active_tokens = labelling_mask.view(-1) == 1
						active_logits = logits_boundaries.view(-1, self.num_labels_boundaries)[active_tokens]
						active_labels = labels_boundaries.view(-1)[active_tokens]
						loss_boundaries = loss_fct(active_logits, active_labels)
				else:
					loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
					if self.predict_boundaries:
						loss_boundaries = loss_fct(logits_boundaries.view(-1, self.num_labels_boundaries), labels_boundaries.view(-1))

		if labels is not None:
			if self.predict_masked:
				loss += loss_masks
			if self.predict_boundaries:
				loss += loss_boundaries
			return loss, logits
		else:
			return logits

	def predictMasked(self):
		return self.predict_masked

	def hasCRF(self):
		return self.use_crf

	def getCRFtags(self, logits, labelling_mask):
		if self.use_crf:
			_, logits = self.crf.viterbi_tags(logits, labelling_mask)
			return logits
		else:
			return None

	def isBiaffine(self):
		return self.biaffine

	#From https://github.com/huggingface/transformers/issues/1730#issuecomment-550081307
	def resize_embedding_and_fc(self, new_num_tokens):

		if self.predict_masked:
			# Change the FC
			old_fc = self.classifier_masked.predictions.decoder
			self.classifier_masked.predictions.decoder = self._get_resized_fc(old_fc, new_num_tokens)

			# Change the bias
			old_bias = self.classifier_masked.predictions.bias
			self.classifier_masked.predictions.bias = self._get_resized_bias(old_bias, new_num_tokens)

		# Change the embedding
		self.resize_token_embeddings(new_num_tokens)

	def _get_resized_bias(self, old_bias, new_num_tokens):
		old_num_tokens = old_bias.data.size()[0]
		if old_num_tokens == new_num_tokens:
			return old_bias

		# Create new biases
		new_bias = nn.Parameter(torch.zeros(new_num_tokens))
		new_bias.to(old_bias.device)

		# Copy from the previous weights
		num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
		new_bias.data[:num_tokens_to_copy] = old_bias.data[:num_tokens_to_copy]
		return new_bias

	def _get_resized_fc(self, old_fc, new_num_tokens):

		old_num_tokens, old_embedding_dim = old_fc.weight.size()
		if old_num_tokens == new_num_tokens:
			return old_fc

		# Create new weights
		new_fc = nn.Linear(in_features=old_embedding_dim, out_features=new_num_tokens)
		new_fc.to(old_fc.weight.device)

		# initialize all weights (in particular added tokens)
		self._init_weights(new_fc)

		# Copy from the previous weights
		num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
		new_fc.weight.data[:num_tokens_to_copy, :] = old_fc.weight.data[:num_tokens_to_copy, :]
		return new_fc

