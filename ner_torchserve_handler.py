import logging
import os
import json
import numpy as np
import torch
import torch.nn as nn
import importlib
import inspect
from transformers import BertTokenizer, BertConfig


logger = logging.getLogger(__name__)


def list_classes_from_module(module, parent_class=None):
    """
    Parse user defined module to get all the model service classes in it.
    Args:
        module
        parent_class
    Returns:
        list: list of model service class definitions
    """
    # parsing the module to get all defined classes
    classes = [
        cls[1]
        for cls in inspect.getmembers(
            module,
            lambda member: inspect.isclass(member) and member.__module__ == module.__name__
        )
    ]
    # filter classes that is subclass of parent_class
    if parent_class is not None:
        return [c for c in classes if issubclass(c, parent_class)]


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()

        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return None


class InputExample(object):
    """
    A single training/test example.
    """
    def __init__(self, guid, words=None, labels=None, sentence=None):
        """Contructs a InputExample object.
        Args:
            guid (TYPE): unique id for the example
            words (TYPE): the words of the sequence
            labels (TYPE): the labels for each work of the sentence
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.sentence = sentence

        if self.words is None and self.sentence:
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            # split sentence on whitepsace so that different tokens may be attributed to their original positions
            for c in self.sentence:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            self.words = doc_tokens
            if self.labels is None:
                self.labels = ["O"]*len(self.words)


class InputFeatures(object):
    """
    A sigle set of input features for an example.
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_ids=None, token_to_orig_index=None, orig_to_token_index=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.token_to_orig_index = token_to_orig_index
        self.orig_to_token_index = orig_to_token_index


class NERTorchServeHandler:
    def __init__(self):
        self.model = None
        self.label2idx = None
        self.device = None
        self.initialized = False
        self.manifest = None

    def initialized(self, ctx):
        self.manifest = manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_availabel() else "cpu")
        self.device = torch.device("cpu")

        # model serialized file (model weights file)
        serialized_file = self.manifest['model']['serializedFile']

        # model def file and other model related files
        model_file = self.manifest['manifest']['modelFile']
        model_def_path = os.path.join(model_dir, model_file)
        model_vocab_path = os.path.join(model_dir, 'vocab.txt')
        model_bert_config_path = os.path.join(model_dir, "bert_config.json")
        model_config_path = os.path.join(model_dir, "bert_for_token_classification.json")
        labels_file_path = os.path.join(model_dir, "labels_file.txt")

        # loading model config file
        if os.path.isfile(model_config_path):
            with open(model_config_path, "r") as reader:
                text = reader.read()
            self.model_config_dict = json.loads(text)
            self.max_seq_length = self.model_config_dict['max_seq_length']
            self.num_special_tokens = self.model_config_dict['num_special_tokens']
        else:
            logger.debug("model_config_path doesnt exists.")

        # loading labels
        if os.path.isfile(labels_file_path):
            self.labels = get_labels(labels_file_path)
            self.label2idx = {l: i for i, l in enumerate(self.labels)}
            self.idx2label = {i: l for i, l in enumerate(self.labels)}
        else:
            logger.debug("labels_file_path doesnt exists.")

        # loading bert config file
        if os.path.isfile(model_bert_config_path):
            self.bert_config = BertConfig.from_json_file(
                model_bert_config_path
            )
        else:
            logger.debug("bert config path doesnt exists.")

        # loading bert tokenizer
        if os.path.isfile(model_vocab_path):
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                model_vocab_path, config=self.bert_config,
                do_lower_case=True # I used do_lower_case=True during training so here also True
            )
        else:
            logger.debug("vocab path doesnt exists.")

        # loading model weigths into definitions
        if os.path.isfile(model_def_path):
            module = importlib.import_module(model_file.split(",")[0])
            model_class_definitions = list_classes_from_module(module)
            if len(model_class_definitions) != 1:
                raise ValueError("Expected only one class as model definition. {}".format(model_class_definitions))

            model_class = model_class_definitions[0]
            self.model = model_class.from_pretrained(
                serialized_file,
                config=self.bert_config,
                num_labels=len(self.labels),
                classification_layer_sizes=self.model_config_dict["classification_layer_sizes"]
            )
        else:
            logger.debug("No model class found")

        self.model.to(self.device)
        self.model.eval()
        logger.debug("Model successfully loaded.")

        self.initialized = True

    def convert_sentence_to_example(self, sentence):
        example = InputExample(
                guid=0, words=None, labels=None, sentence=sentence
        )
        return example

    def convert_example_to_feature(self, example):
        tokens = []
        token_to_orig_index = []
        orig_to_token_index = []
        for word_idx, word in enumerate(example.words):
            orig_to_token_index.append(len(tokens))
            word_tokens = self.bert_tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
            for tok in word_tokens:
                token_to_orig_index.append(word_idx)

        if len(tokens) > self.max_seq_length - self.special_tokens_count:
            tokens = tokens[:(self.max_seq_length - self.special_tokens_count)]

        tokens += [self.bert_tokenizer.sep_token]
        segment_ids = [0]*len(tokens)

        tokens = [self.bert_tokenizer.cls_token] + tokens
        segment_ids = [0] + segment_ids

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        # Zero pad up to the sequence length
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [self.bert_tokenizer.pad_token_id] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=None,
            token_to_orig_index=token_to_orig_index,
            orig_to_token_index=orig_to_token_index
        )
        return feature




        self.model_config_path = model_config_path
        self.labels_file = labels_file
        self.device = device
        if os.path.exists(self.model_config_path):
            with open(self.model_config_path, "r", encoding="utf-8") as reader:
                text = reader.read()
            self.model_config_dict = json.loads(text)
        else:
            print("model_config_path doesn't exist.")
            sys.exit()

        if os.path.exists(self.model_config_dict["final_model_saving_dir"]):
            self.model_file = self.model_config_dict["final_model_saving_dir"] + "pytorch_model.bin"
            self.config_file = self.model_config_dict["final_model_saving_dir"] + "bert_config.json"
            self.vocab_file = self.model_config_dict["final_model_saving_dir"] + "vocab.txt"
        else:
            print("model_saving_dir doesn't exist.")
            sys.exit()
        if os.path.exists(self.labels_file):
            print("Labels file exist")
        else:
            print("labels_file doesn't exist.")
            sys.exit()

        self.bert_config = BertConfig.from_json_file(self.config_file)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.vocab_file,
            config=self.bert_config,
            do_lower_case=self.model_config_dict["tokenizer_do_lower_case"]
        )
        self.labels = get_labels(self.labels_file) + ["<PAD>"]
        self.label2idx = {l: i for i, l in enumerate(self.labels)}

        self.model = BertForTokenClassification.from_pretrained(
            self.model_file,
            config=self.bert_config,
            num_labels=len(self.labels),
            classification_layer_sizes=self.model_config_dict["classification_layer_sizes"]
        )
        self.model.to(self.device)
        print("Model loaded successfully from the config provided.")

    def tag_sentences(self, sentence_list, logger, batch_size):
        dataset, examples, features = load_and_cache_examples(
            max_seq_length=self.model_config_dict["max_seq_length"],
            tokenizer=self.bert_tokenizer,
            label_map=self.label2idx,
            pad_token_label_id=self.label2idx["<PAD>"],
            mode="inference", data_dir=None,
            logger=logger, sentence_list=sentence_list,
            return_features_and_examples=True
        )

        label_predictions = predictions_from_model(
            model=self.model, tokenizer=self.bert_tokenizer,
            dataset=dataset, batch_size=batch_size,
            label2idx=self.label2idx, device=self.device
        )
        # restructure test_label_predictions with real labels
        aligned_predicted_labels, _ = align_predicted_labels_with_original_sentence_tokens(
            label_predictions, examples, features,
            max_seq_length=self.model_config_dict["max_seq_length"],
            num_special_tokens=self.model_config_dict["num_special_tokens"]
        )
        results = []
        for label_tags, example in zip(aligned_predicted_labels, examples):
            results.append(
                convert_to_ents(example.words, label_tags)
            )
        return results
