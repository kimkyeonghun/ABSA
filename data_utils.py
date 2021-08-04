import os
import json

from pytorch_pretrained_bert.tokenization import BertTokenizer

class AETokenizer(BertTokenizer):
    def subword_tokenize(self, tokens, labels): # for AE
        split_tokens, split_labels = [], []
        idx_map = []
        for idx, token in enumerate(tokens):
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            for jdx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[idx]=="B" and jdx > 0:
                    split_labels.append("I")
                else:
                    split_labels.append(labels[idx])
                idx_map.append(idx)
        return split_tokens, split_labels, idx_map

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data set"""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
        
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

class AeProcessor(DataProcessor):
    """Processor for the SemEval Aspect Extraction"""

    def get_train_examples(self, data_dir, fn='train.json'):
        return self._create_examples(self._read_json(data_dir),"train")

    def get_dev_examples(self, data_dir, fn='dev.json'):
        return self._create_examples(self._read_json(data_dir),"dev")

    def get_test_examples(self, data_dir, fn='test.json'):
        return self._create_examples(self._read_json(data_dir),"test")

    def get_labels(self):
        return ["O","B","I"]
    
    def _create_examples(self,lines,set_type):
        """Create examples for the training and dev sets."""
        examples = []
        for (_,ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids)
            text_a = lines[ids]['sentence']
            label = lines[ids]['label']
            examples.append(InputExample(guid=guid,text_a=text_a,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        if mode != "AE":
            pass
        else:
            tokens_a, labels_a, example.idx_map = tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label)
        
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            _truncate_seq_pair(tokens_a,tokens_b,max_seq_length-3)
        else:
            if len(tokens_a) > max_seq_length -2:
                tokens_a = tokens_b[0:(max_seq_length-2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        #zero padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode != "AE":
            label_id = label_map[example.label]
        else:
            label_id = [-1]*len(input_ids) # -1 is ignore token
            lb = [label_map[label] for label in labels_a]
            if len(lb) > max_seq_length -2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb)+1] = lb
    
        features.append(InputFeatures(
            input_ids = input_ids,
            input_mask = input_mask,
            segment_ids = segment_ids,
            label_id = label_id
        ))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()