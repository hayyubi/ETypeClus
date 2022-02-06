import argparse
import pickle as pk
import spacy
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from extract_svo import findSVOs

def parse(line):

    save_dict_data = {}

    if IS_SENTENCE == 1:
        doc = NLP(line)
        token_list = [token.text for token in doc]
        raw_sent = " ".join(token_list)
        svos = findSVOs(doc)

        save_dict_data = {
            "raw_sentence": raw_sent,
            "token_list": token_list,
            "svos": svos,
        }
    else:
        doc = NLP(line)
        for sent in doc.sents:
            new_doc = NLP(sent.text)

            token_list = [token.text for token in new_doc]
            raw_sent = " ".join(token_list)
            svos = findSVOs(new_doc)

            save_dict_data = {
                "raw_sentence": raw_sent,
                "token_list": token_list,
                "svos": svos,
            }

    return save_dict_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", help="input corpus file")
    parser.add_argument("--save_path", help="output file with subject-verb-object triplets extracted")
    parser.add_argument("--is_sentence", default=1, help="1 if each line in input_file is a sentence, 0 if each line is a document")
    parser.add_argument("--spacy_model", default="en_core_web_lg", help="Spacy model for POS tagging")
    parser.add_argument("--num_threads", type=int, help="Number of threads used to download articles") 
    args = parser.parse_args()
    print(vars(args))

    lines = []
    with open(args.input_file, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                lines.append(line)
    
    IS_SENTENCE = args.is_sentence

    print("loading Spacy model")
    NLP = spacy.load(args.spacy_model)

    parsed_data = []
    with Pool(args.num_threads) as p:
        for result in tqdm(p.imap_unordered(parse, lines), total=len(lines)):
            parsed_data.append(result)

    with open(args.save_path, "wb") as f:
        pk.dump(parsed_data, f)