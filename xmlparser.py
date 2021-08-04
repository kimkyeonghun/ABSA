from abc import ABC
import os
import json
import argparse
from glob import glob
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()

parser.add_argument('--year', type = str, default = '2014', choices = ['2014','2015','2016'])

args = parser.parse_args()

DATA_PATH = './data'



def matchSentiment(polarity):
    if polarity == "positive":
        return 1
    elif polarity == "neutral":
        return 0
    elif polarity == "negative":
        return -1
    else:
        return float('inf')

def parse2014():
    Laptop_PATH = os.path.join(DATA_PATH,args.year,"Laptops")
    Laptops = glob(os.path.join(Laptop_PATH,'*.xml'))
    Restaurant_PATH = os.path.join(DATA_PATH,args.year,"Restaurants")
    Restaurants = glob(os.path.join(Restaurant_PATH,'*.xml'))
    xml_files = [Laptops,Restaurants]
    for xml_file in xml_files:
        for xml_f in xml_file:
            name,type_ = xml_f.split("-")
            name = name.split("/")[-1]
            type__ = type_.split(".")[0]
            doc = ET.parse(xml_f)
            root = doc.getroot()
            sentences = root.findall("sentence")
            dataset = dict()
            for sentence in sentences:
                sentence_id = sentence.attrib['id']
                if dataset.get(sentence_id):
                    raise Exception("Duplicate Id")
                else:
                    dataset[sentence_id] = {"text": sentence.findtext("text")}
                    aspectTerms = sentence.find("aspectTerms")
                    if aspectTerms is None:
                        continue
                    for aspectTerm in aspectTerms:
                        aspect = aspectTerm.attrib
                        aspect_po = matchSentiment(aspect['polarity'])
                        if aspect_po == float('inf'):
                            continue
                        if dataset[sentence_id].get("aspect"):
                            dataset[sentence_id]["aspect"].append([aspect["term"],aspect_po,(aspect['from'],aspect['to'])])
                        else:
                            dataset[sentence_id]["aspect"] = [[aspect["term"],aspect_po,(aspect['from'],aspect['to'])]]
            with open(os.path.join(DATA_PATH,'raw_data',f"{name}_{args.year}_{type__}.json"),"w") as f:
                json.dump(dataset,f)
            print("FIN",xml_f)

def parse2015():
    Restaurant_PATH = os.path.join(DATA_PATH,args.year,"Restaurants")
    xml_files = glob(os.path.join(Restaurant_PATH,'*.xml'))
    for xml_file in xml_files:
        no_aspect = 0
        real_count = 0
        sentence_count = 0
        name,type_ = xml_file.split("-")
        name = name.split("/")[-1]
        type__ = type_.split(".")[0]
        doc = ET.parse(xml_file)
        root = doc.getroot()
        reviews = root.findall("Review")
        dataset = dict()
        for review in reviews:
            sentences = review.findall('sentences')
            for sentence in sentences:
                sen = sentence.findall('sentence')
                sentence_count += len(sen)
                for s in sen:
                    sentence_id = s.attrib['id']
                    if dataset.get(sentence_id):
                        raise Exception("Duplicate Id")
                    else:
                        text = s.findtext("text")
                        dataset[sentence_id] = {"text": text}
                        aspectTerms = s.find("Opinions")
                        if aspectTerms is None:
                            no_aspect+=1
                            continue
                        aspectTerms = aspectTerms.findall('Opinion')
                        for aspectTerm in aspectTerms:
                            real_count +=1
                            aspect = aspectTerm.attrib
                            aspect_po = matchSentiment(aspect['polarity'])
                            if aspect_po == float('inf'):
                                real_count -=1
                                continue
                            if aspect['target'] != "NULL":
                                if dataset[sentence_id].get("aspect"):
                                    if dataset[sentence_id]["aspect"][0][0]==aspect['target']:
                                        real_count-=1
                                        continue
                                    dataset[sentence_id]["aspect"].append([aspect["target"],aspect_po,(aspect['from'],aspect['to'])])
                                else:
                                    dataset[sentence_id]["aspect"] = [[aspect["target"],aspect_po,(aspect['from'],aspect['to'])]]
                            else:
                                real_count -=1
        with open(os.path.join(DATA_PATH,'raw_data',f"{name}_{args.year}_{type__}.json"),"w") as f:
                    json.dump(dataset,f)    
        print(xml_file,real_count)    

def parse2016():
    Restaurant_PATH = os.path.join(DATA_PATH,args.year,"Restaurants")
    xml_files = glob(os.path.join(Restaurant_PATH,'*.xml'))
    for xml_file in xml_files:
        name,type_ = xml_file.split("-")
        name = name.split("/")[-1]
        type__ = type_.split(".")[0]
        doc = ET.parse(xml_file)
        root = doc.getroot()
        reviews = root.findall("Review")
        no_aspect = 0
        real_count = 0
        dataset = dict()
        for review in reviews:
            sentences = review.findall('sentences')
            for sentence in sentences:
                sen = sentence.findall('sentence')
                for s in sen:
                    sentence_id = s.attrib['id']
                    if dataset.get(sentence_id):
                        raise Exception("Duplicate Id")
                    else:
                        text = s.findtext("text")
                        dataset[sentence_id] = {"text": text}
                        aspectTerms = s.find("Opinions")
                        if aspectTerms is None:
                            no_aspect+=1
                            continue
                        aspectTerms = aspectTerms.findall('Opinion')
                        targets = []
                        for aspectTerm in aspectTerms:
                            aspect = aspectTerm.attrib
                            if aspect['target']=='NULL':
                                continue
                            real_count +=1
                            aspect_po = matchSentiment(aspect['polarity'])
                            if aspect_po == float('inf'):
                                real_count -=1
                                continue
                            if dataset[sentence_id].get("aspect"):
                                if aspect['target'] in targets:
                                    real_count-=1
                                    continue
                                dataset[sentence_id]["aspect"].append([aspect["target"],aspect_po,(aspect['from'],aspect['to'])])
                                targets.append(aspect['target'])
                            else:
                                dataset[sentence_id]["aspect"] = [[aspect["target"],aspect_po,(aspect['from'],aspect['to'])]]
                                targets.append(aspect['target'])
        with open(os.path.join(DATA_PATH,'raw_data',f"{name}_{args.year}_{type__}.json"),"w") as f:
                    json.dump(dataset,f)
        print(xml_file,real_count)


if __name__ == "__main__":
    if args.year == '2014':
        parseData = parse2014
    elif args.year == '2015':
        parseData = parse2015
    elif args.year == '2016':
        parseData = parse2016
    parseData()
    