import random

from torch.utils import data
import torch
from tqdm import tqdm


class BatcherBERT_multidata(data.Dataset):
    def __init__(self, dataset, bert_tokenizer, tags_field="NER_IOBES", max_length=None, tagset=None, special_labels=False, force_size=None,
                 test=False, mask_percentage=0.0, uppercase=False, predict_boundaries=False, boundaries_dict=None, create_aligner=True,
                 fullwords_mask=False, redundant_uppercase=False, as_da=False, override_da_index=0, uppercase_percentge=0.0):
        self.__bert_tokenizer = bert_tokenizer
        self.__max_length = max_length
        self.__tags_field = tags_field
        self.__has_triggers = False
        self.__special_labels = special_labels
        self.__tagset = tagset
        self.__mask_percentage = 0.0
        self.__uppercase_percentage = uppercase_percentge
        self.__predict_masked = False
        self.__uppercase = uppercase
        self.__redundant_uppercase = redundant_uppercase
        self.__predict_boundaries = False
        self.__force_size = force_size
        self.__test = test
        self.__create_aligner = False
        self.__aligner = []
        self.__mask_all_subtokens = False
        self.__as_da = as_da
        self.__override_da_index = override_da_index
        if self.__test:
            self.__create_aligner = create_aligner
            self.__data = self.__prepareTokens(dataset)
            self.__size = len(self.__data)
            self.__entries_ids = list(range(self.__size))
        else:
            self.__predict_boundaries = predict_boundaries
            self.__boundaires_convertor = self.__createConvertorBoundaries(boundaries_dict)
            self.__mask_percentage = mask_percentage
            if self.__mask_percentage > 0.0:
                self.__predict_masked = True
                self.__fullwords_mask = fullwords_mask
            self.__dataset = dataset
            self.__data = None
            self.__size = 0
            self.__entries_ids = []

    def __createConvertorBoundaries(self, boundaries_dict):
        if not self.__predict_boundaries:
            return None
        conversion_dict = {}
        for tag in self.__tagset:
            if tag == "O":
                conversion_dict[self.__tagset["O"]] = boundaries_dict["O"]
            elif tag[0] == "B":
                conversion_dict[self.__tagset[tag]] = boundaries_dict["B-A"]
            elif tag[0] == "I":
                conversion_dict[self.__tagset[tag]] = boundaries_dict["I-A"]
            elif tag[0] == "S":
                conversion_dict[self.__tagset[tag]] = boundaries_dict["S-A"]
            elif tag[0] == "E":
                conversion_dict[self.__tagset[tag]] = boundaries_dict["E-A"]
            else:
                conversion_dict[self.__tagset[tag]] = boundaries_dict[tag]
        return conversion_dict

    def getAligner(self):
        if self.__create_aligner:
            return self.__aligner
        else:
            return None

    def createBatches(self):
        self.__data = self.__prepareTokens(self.__dataset)
        self.__size = len(self.__data)
        self.__entries_ids = list(range(self.__size))

    def __len__(self):
        return self.__size

    def __entryGenerator(self, sub_dataset_id):
        bert_tokens_list = []
        new_entry = {}
        # This will be the CLS token
        if not self.__special_labels:
            new_entry["bert_tokens_mask"] = [0]
            new_entry["labelling_mask"] = []
            if self.__tags_field is not None:
                new_entry["bert_tags"] = []
            if self.__predict_masked:
                new_entry["lm_mask"] = []
                new_entry["lm_labels"] = []
        else:
            new_entry["bert_tokens_mask"] = [1]
            #We don't want to predict this token during testing
            if self.__test:
                new_entry["labelling_mask"] = [0]
            else:
                new_entry["labelling_mask"] = [1]
            if self.__tags_field is not None:
                new_entry["bert_tags"] = [self.__tagset["[CLS]"]]   # During testing this will disappear with the labelling mask
            if self.__predict_masked:
                new_entry["lm_mask"] = [0]
                new_entry["lm_labels"] = []
        new_entry["token_type_ids"] = [0]  # This will be the CLS token
        if self.__as_da:
            new_entry["sub_dataset"] = sub_dataset_id
        return bert_tokens_list, new_entry

    def __processEntry(self, bert_tokens_list, new_entry, overflow=-1):
        encoding = self.__bert_tokenizer.encode_plus(bert_tokens_list,
                                                     max_length=self.__max_length,
                                                     pad_to_max_length=True,
                                                     return_attention_masks=True,
                                                     add_special_tokens=True,
                                                     )
        new_entry["bert_tokens"] = encoding["input_ids"]
        new_entry["attention_mask"] = encoding["attention_mask"]

        # This is the SEP
        if self.__special_labels:
            new_entry["bert_tokens_mask"].append(1)

            if overflow == -1:
                overflow = len(new_entry["labelling_mask"])

            if self.__tags_field is not None:
                new_entry["bert_tags"].insert(overflow, self.__tagset["[SEP]"])

            if self.__test:
                new_entry["labelling_mask"].insert(overflow, 0)
            else:
                new_entry["labelling_mask"].insert(overflow, 1)

        if self.__predict_boundaries:
            new_entry["boundaries_tags"] = []
            for tag in new_entry["bert_tags"]:
                if tag == 0:
                    new_entry["boundaries_tags"].append(0)
                else:
                    new_entry["boundaries_tags"].append(self.__boundaires_convertor[tag])

        if self.__tags_field is not None and self.__max_length is not None:
            for i in range(len(new_entry["bert_tags"]), self.__max_length):
                new_entry["bert_tags"].append(0)
            assert (len(new_entry["bert_tags"]) == self.__max_length)

        if self.__max_length is not None:
            for i in range(len(new_entry["bert_tokens_mask"]), self.__max_length):
                new_entry["bert_tokens_mask"].append(0)
        assert (len(new_entry["bert_tokens_mask"]) == self.__max_length)

        if self.__max_length is not None:
            for i in range(len(new_entry["labelling_mask"]), self.__max_length):
                new_entry["labelling_mask"].append(0)
        assert (len(new_entry["labelling_mask"]) == self.__max_length)

        assert (len(new_entry["labelling_mask"]) == len(new_entry["bert_tokens_mask"]))

        if self.__max_length is not None:
            for i in range(len(new_entry["token_type_ids"]), self.__max_length):
                new_entry["token_type_ids"].append(0)

        if self.__predict_masked and self.__max_length is not None:
            for i in range(len(new_entry["lm_mask"]), self.__max_length):
                new_entry["lm_mask"].append(0)
            assert (len(new_entry["lm_mask"]) == self.__max_length)
            for i in range(len(new_entry["lm_labels"]), self.__max_length):
                new_entry["lm_labels"].append(0)
            assert (len(new_entry["lm_labels"]) == self.__max_length)

        if self.__predict_boundaries:
            for i in range(len(new_entry["boundaries_tags"]), self.__max_length):
                new_entry["boundaries_tags"].append(0)
            assert (len(new_entry["boundaries_tags"]) == self.__max_length)

        assert (len(new_entry["token_type_ids"]) == self.__max_length)

        assert (len(new_entry["token_type_ids"]) == len(new_entry["bert_tokens_mask"]))

        return new_entry

    def __prepareTokens(self, dataset):
        modified_dataset = []
        for (sub_dataset_id, sub_dataset) in enumerate(dataset):
            if self.__override_da_index:
                sub_dataset_id = self.__override_da_index
            else:
                sub_dataset_id += 1

            entries_to_upper = []
            if self.__uppercase_percentage > 0.0:
                entries_to_upper = random.sample(list(range(len(sub_dataset))),
                                                 int(round(len(sub_dataset) * self.__uppercase_percentage)))
                entries_to_upper.sort()

            for (entry_id, entry) in enumerate(tqdm(sub_dataset, total=len(sub_dataset))):
                split_in_more = 0
                entities_to_mask = []

                if self.__create_aligner:
                    self.__aligner.append(len(entry["tokens"]))

                if self.__mask_percentage > 0.0 and not self.__test:
                    tokens_sequence_length = len(entry["tokens"])
                    if tokens_sequence_length > 3:
                        entities_to_mask = random.sample(list(range(tokens_sequence_length)),
                                                         int(round(tokens_sequence_length * self.__mask_percentage)))
                        entities_to_mask.sort()

                bert_tokens_list, new_entry = self.__entryGenerator(sub_dataset_id)
                if self.__tags_field is not None:
                    if entry_id in entries_to_upper:
                        upper_tokens = []
                        for token in entry["tokens"]:
                            upper_tokens.append(token.upper())
                        zipped_info = zip(upper_tokens, entry[self.__tags_field])
                    else:
                        zipped_info = zip(entry["tokens"], entry[self.__tags_field])
                    flag = 0
                else:
                    zipped_info = entry["tokens"]
                    flag = 1

                tag = None
                sentence_limit = False
                overflow = -1
                stop_process = False
                for info_id, info in enumerate(zipped_info):
                    if flag == 0:
                        token, tag = info
                    else:
                        token = info

                    if len(entities_to_mask) > 0 and info_id in entities_to_mask:
                        bert_tokens = self.__bert_tokenizer.tokenize(token)
                        if len(bert_tokens) == 0:  # If for whatever reason bert_tokens is empty, then, we need to consider that this characters is unknown
                            bert_tokens = ["[UNK]"]
                        bert_tokens_ids = self.__bert_tokenizer.convert_tokens_to_ids(bert_tokens)
                        if not self.__fullwords_mask:
                            token_to_mask = 0
                            if len(bert_tokens_ids) > 1:
                                token_to_mask = random.randint(0, len(bert_tokens_ids)-1)
                            bert_tokens[token_to_mask] = "[MASK]"
                            new_entry["lm_labels"].append(bert_tokens_ids[token_to_mask])
                        else:
                            # Deliverable and ECIR 2021
                            bert_tokens = []
                            for ids in bert_tokens_ids:
                                new_entry["lm_labels"].append(ids)
                                bert_tokens.append("[MASK]")
                    else:
                        if self.__uppercase and token.isupper():
                            if not self.__redundant_uppercase:
                                token_title = token.title()
                                if token_title != token_title:
                                    token = f"[UP] {token} {token.title()} {token.lower()} [up]"
                                else:
                                    token = f"[UP] {token} {token.lower()} [up]"
                            else:
                                token = f"[UP] {token} {token.title()} {token.lower()} [up]"
                        bert_tokens = self.__bert_tokenizer.tokenize(token)

                    if len(bert_tokens) == 0:  # If for whatever reason bert_tokens is empty, then, we need to consider that this characters is unknown
                        bert_tokens = ["[UNK]"]
                    # The reson for self.__max_length-1 is because, we have an extra token at the end for BERT, thus, a string can be only 126 bert tokens
                    if len(new_entry["token_type_ids"]) + len(bert_tokens) > self.__max_length-1:
                        split_in_more += 1
                        if sentence_limit:
                            if len(new_entry["labelling_mask"]) + 1 > self.__max_length-1:
                                print(f"Warning, dev/test sentence representation overflowing, trunking to the maximum number of real tokens allowed: {self.__max_length-1}")
                                stop_process = True
                            else:
                                self.__processOverloadTokensToEntry(new_entry, bert_tokens, tag)
                        elif self.__force_size:
                            difference = self.__max_length-1 - len(new_entry["token_type_ids"])
                            if difference < 0:
                                difference = 0
                            elif difference > 0:
                                bert_tokens_in = bert_tokens[:difference]
                                bert_tokens_list.extend(bert_tokens_in)
                                self.__processTokensToEntry(new_entry, bert_tokens_in, tag)
                            if self.__test:
                                overflow = len(new_entry["labelling_mask"])
                                self.__processOverloadTokensToEntry(new_entry, bert_tokens[difference:], tag)
                            else:
                                print(f"Warning, training sentence representation overflowing, trunking to {self.__max_length}")
                                stop_process = True
                            sentence_limit = True
                        else:
                            new_entry = self.__processEntry(bert_tokens_list, new_entry)
                            modified_dataset.append(new_entry)
                            bert_tokens_list, new_entry = self.__entryGenerator(sub_dataset_id)
                    if not sentence_limit:
                        bert_tokens_list.extend(bert_tokens)
                        self.__processTokensToEntry(new_entry, bert_tokens, tag)
                    if stop_process:
                        break

                new_entry = self.__processEntry(bert_tokens_list, new_entry, overflow=overflow)

                modified_dataset.append(new_entry)

        return modified_dataset

    def __processOverloadTokensToEntry(self, new_entry, bert_tokens, tag):
        if len(bert_tokens) > 0:
            new_entry["labelling_mask"].append(1)
            if self.__tags_field is not None:
                new_entry["bert_tags"].append(tag)

    def __processTokensToEntry(self, new_entry, bert_tokens, tag):
        for i in range(len(bert_tokens)):
            new_entry["token_type_ids"].append(0)
            if i == 0:
                new_entry["bert_tokens_mask"].append(1)
                new_entry["labelling_mask"].append(1)
                if self.__tags_field is not None:
                    new_entry["bert_tags"].append(tag)
            else:
                new_entry["bert_tokens_mask"].append(0)
            if self.__predict_masked:
                if bert_tokens[i] == "[MASK]":
                    new_entry["lm_mask"].append(1)
                else:
                    new_entry["lm_mask"].append(0)

    def __getitem__(self, index):
        id_ = self.__entries_ids[index]
        return self.__data[id_]

    def collate_fn(self, batch):
        temp_batch = {}
        for entry in batch:
            for field_name in entry:
                if field_name not in temp_batch:
                    temp_batch[field_name] = []
                temp_batch[field_name].append(entry[field_name])
        batch = temp_batch
        del temp_batch
        bert_tokens = torch.tensor([f for f in batch['bert_tokens']], dtype=torch.long)
        attention_mask = torch.tensor([f for f in batch["attention_mask"]], dtype=torch.long)
        token_type_ids = torch.tensor([f for f in batch["token_type_ids"]], dtype=torch.long)
        bert_tags = None
        labels_boundaries = None
        if self.__tags_field is not None:
            bert_tags = torch.tensor([f for f in batch["bert_tags"]], dtype=torch.long)
            if self.__predict_boundaries:
                labels_boundaries = torch.tensor([f for f in batch["boundaries_tags"]], dtype=torch.long)
        bert_tokens_mask = torch.tensor([f for f in batch["bert_tokens_mask"]], dtype=torch.long)
        labelling_mask = torch.tensor([f for f in batch["labelling_mask"]], dtype=torch.long)
        lm_mask = None
        lm_labels = None
        if self.__predict_masked:
            lm_mask = torch.tensor([f for f in batch["lm_mask"]], dtype=torch.long)
            lm_labels = torch.tensor([f for f in batch["lm_labels"]], dtype=torch.long)
        da_sets = None
        if self.__as_da:
            da_sets = torch.tensor([f for f in batch["sub_dataset"]], dtype=torch.long)

        if self.__test:
            return bert_tokens, attention_mask, token_type_ids, bert_tags, bert_tokens_mask, labelling_mask, labels_boundaries, da_sets
        else:
            return bert_tokens, attention_mask, token_type_ids, bert_tags, bert_tokens_mask, labelling_mask, lm_mask, lm_labels, labels_boundaries, da_sets



