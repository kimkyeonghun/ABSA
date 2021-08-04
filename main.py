import os
import argparse
import logging

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
from transformers.optimization import AdamW
from AEBERT import BertForSequenceClassification
from transformers.models.bert.configuration_bert import BertConfig

from data_utils import AETokenizer, AeProcessor
import data_utils
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-6, help= "Learning Rate")
parser.add_argument("--data", type=str, default='Restaurants',choices=['Restaurants',"Laptops","Twitter"])
parser.add_argument("--year", type=int, default=2014, help="Choose semEval year in [2014, 2015, 2016] except Twitter")
parser.add_argument("--n_epoch",type=int, default=100, help="# of Model Epoch")
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=4)
args=parser.parse_args()

logger,log_dir = utils.get_logger(os.path.join('./logs'))

DEVICE = torch.device("cuda:0")


def load_dataset():
    logger.info(f"**************************Load SemEval_{args.data}_{args.year} Dataset")
    processor = AeProcessor()
    tokenizer = AETokenizer.from_pretrained("bert-base-uncased")
    label_list = processor.get_labels()
    train_examples = processor.get_train_examples(os.path.join('data',"AE_data",f"{args.data}_{args.year}_Train_AE.json"))
    test_examples = processor.get_test_examples(os.path.join('data',"AE_data",f"{args.data}_{args.year}_Test_AE.json"))

    train_feature = data_utils.convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, "AE")

    all_input_ids = torch.tensor([f.input_ids for f in train_feature], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_feature], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_feature], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in train_feature], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_id)
    train_sampler = RandomSampler(train_data)
    trainLoader = DataLoader(train_data, sampler = train_sampler, batch_size=args.train_batch_size)

    test_feature = data_utils.convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer, "AE")

    all_input_ids = torch.tensor([f.input_ids for f in test_feature], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_feature], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_feature], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in test_feature], dtype=torch.long)

    test_data = TensorDataset(all_input_ids,all_segment_ids,all_input_mask,all_label_id)
    test_sampler = RandomSampler(test_data)
    testLoader = DataLoader(test_data, sampler = test_sampler, batch_size=args.test_batch_size)

    return trainLoader, testLoader, processor

def prepare_for_training(processor):
    model_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(processor.get_labels()), config = model_config).to(DEVICE)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
             "params" : [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    return model, optimizer

def train_epoch(model, trainLoader, optimizer):
    train_loss = 0.
    model.train()
    for _, batch in enumerate(tqdm(trainLoader,desc="Iteration")):
        batch = tuple(t.cuda(DEVICE) for t in batch)
        input_ids, segment_ids, input_mask, label_id = batch
        loss = model(input_ids, segment_ids, input_mask, label_id)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    return train_loss/len(trainLoader)

def test_epoch(model, testLoader):
    test_loss = 0.
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testLoader, desc="Iteration")):
            batch = tuple(t.cuda(DEVICE) for t in batch)
            input_ids, segment_ids, input_mask, label_id = batch
            loss = model(input_ids, segment_ids, input_mask, label_id)
            test_loss += loss.item()

    return test_loss/len(testLoader)

def train(model, optimizer, trainLoader, testLoader):
    logger.info("============================Training Start=====================================")
    logger.info(f" Training Exampels : {len(trainLoader)}")
    logger.info(f" Batch Size : {args.train_batch_size}")

    model_save_path = utils.make_date_dir('./model_save')
    logger.info(f"Model save path: {model_save_path}")
    patience = 0
    best_epoch = 0
    best_loss = float('inf')
    for epoch in range(int(args.n_epoch)):
        patience +=1

        logger.info("====================================Train====================================")
        train_loss = train_epoch(model, trainLoader, optimizer)
        logger.info(f"[Epoch {epoch + 1}] train_loss : {train_loss}")
        logger.info("====================================Test====================================")
        test_loss = test_epoch(model,testLoader)
        logger.info(f"[Epoch {epoch + 1}] train_loss : {test_loss}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch+1
            patience = 0
        else:
            patience += 1

    logger.info(f"[Best Epoch {best_epoch}] best_loss : {best_loss}")

def main():
    trainLoader, testLoader, processor = load_dataset()
    model, optimizer = prepare_for_training(processor)
    train(model, optimizer, trainLoader, testLoader)


if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()