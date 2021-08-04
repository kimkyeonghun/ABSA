import os
import json
import glob
import argparse
from collections import defaultdict

from allennlp.predictors.predictor import Predictor
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm 

DATA_DIR = './data/raw_data'
model_path = os.path.join('./data','pretrained-models',"biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",type=str, default=model_path, help='Path to biaffine dependency parser.')
    parser.add_argument("--data_path",type=str, default=DATA_DIR, help='Directory of where semEval or twitter data held.')
    parser.add_argument('--mode', type=str, default = 'ABSA')

    return parser.parse_args()

def json2docs(file_path, predictor):
    with open(file_path, 'r') as f:
        data = json.load(f)
    docs = defaultdict(dict)
    print("Predicting dependency information...")
    for key in tqdm(data.keys()):
        docs[key] = predictor.predict(sentence=data[key]['text'])
    return docs

def dependencies2format(doc):
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    sentence = {}
    sentence['tokens'] = doc['words']
    sentence['tags'] = doc['pos']
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc['predicted_dependencies']
    predicted_heads = doc['predicted_heads']
    sentence['predicted_dependencies'] = doc['predicted_dependencies']
    sentence['predicted_heads'] = doc['predicted_heads']
    sentence['dependencies'] = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentence['dependencies'].append([dep_tag, frm, to])

    return sentence


def get_dependency(file_path, predictor):
    docs = json2docs(file_path,predictor)
    sentences = defaultdict(dict)
    for key in docs.keys():
        sentences[key] = dependencies2format(docs[key])
    return sentences

def syntaxInfo2json(sentences, originData, data_path):
    json_data = []
    tk = TreebankWordTokenizer()
    with open(originData, 'r') as f:
        real_count = 0
        data = json.load(f)
        for key in data.keys():
            example = dict()
            if data[key].get('aspect') is None:
                continue
            example['sentence'] = data[key]['text']
            example['tokens'] = sentences[key]['tokens']
            example['tags'] = sentences[key]['tags']
            example['predicted_dependencies'] = sentences[key]['predicted_dependencies']
            example['predicted_heads'] = sentences[key]['predicted_heads']
            example['dependencies'] = sentences[key]['dependencies']
            example["aspect_sentiment"] = []
            example['from_to'] = []
            for c in data[key]['aspect']:
                real_count += 1
                example["aspect_sentiment"].append((c[0],c[1]))
                left_idx, right_idx = map(int,c[-1])
                left_word_offset = len(tk.tokenize(example['sentence'][:left_idx]))
                to_word_offset = len(tk.tokenize(example['sentence'][:right_idx]))

                example['from_to'].append((left_word_offset,to_word_offset))
            if len(example['aspect_sentiment']) == 0:
                continue
            json_data.append(example)
    biaffine_dependency_filepath = originData.replace('raw_data','biaffine_data')
    biaffine_dependency_filename = biaffine_dependency_filepath.replace('.json','_biaffine_dep.json')
    with open(biaffine_dependency_filename, 'w') as f:
        json.dump(json_data, f)
    print("DONE!", biaffine_dependency_filename ,real_count)

def json2AE(file_path, predictor):
    tk = TreebankWordTokenizer()
    with open(file_path,'r') as f:
        data = json.load(f)
    count = 0
    AE_data = {}
    label = -1
    for key in tqdm(data.keys(),desc='Iteration'):
        label+=1
        parsed = predictor.predict(sentence=data[key]['text'])
        AE_data[str(label)] = {"sentence" : parsed['words']}
        if data[key].get('aspect'):
            labels = ['O']*len(parsed['words'])
            temp = []
            for c in data[key]['aspect']:
                left_idx, right_idx = map(int,c[-1])
                left_word_offset = len(tk.tokenize(data[key]['text'][:left_idx]))
                to_word_offset = len(tk.tokenize(data[key]['text'][:right_idx]))
                temp.append((left_word_offset,to_word_offset))
            for t in temp:
                first = True
                try:
                    for i in range(t[0],t[1]):
                        if first:
                            labels[i] = 'B'
                            first = False
                        else:
                            labels[i] = 'I'
                    AE_data[str(label)]['label'] = labels
                except:
                    del AE_data[str(label)]
                    label -=1                
        else:
            AE_data[str(label)]['label'] = ["O"]*len(parsed['words'])
    AE_filepath = file_path.replace('raw_data','AE_data')
    AE_filename = AE_filepath.replace('.json','_AE.json')
    with open(AE_filename, 'w') as f:
        json.dump(AE_data, f)
    print("DONE",file_path)
    

def main():
    args = parse_args()
    predictor = Predictor.from_path(args.model_path)
    if args.mode == "ABSA":
        datas = glob.glob(os.path.join(args.data_path,'*.json'))
        for data in datas:
            sentences = get_dependency(data,predictor)
            syntaxInfo2json(sentences, data, os.path.join(args.data_path))
    elif args.mode == 'AE':
        datas = glob.glob(os.path.join(args.data_path,'*.json'))
        for data in datas:
            json2AE(data, predictor)

if __name__ == "__main__":
    main()