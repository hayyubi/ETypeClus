"""
Please run parse_corpus_and_extract_svo.py first to obtain the .pk sentence files before running this code.
"""
import argparse
import json
import pickle as pk
from collections import defaultdict
from tqdm import tqdm
import spacy
from spacy.tokens import Doc
from nltk.corpus import stopwords
import string
from multiprocessing import Pool


stop_words = set(stopwords.words('english'))
for c in string.ascii_letters:
    stop_words.add(c)
for c in string.digits:
    stop_words.add(c)
for c in string.punctuation:
    stop_words.add(c)


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def get_salience(item2local_freq, item2global_freq):
    import math
    N_sent_bkg = 100000000
    item2salience = {}
    for item, local_freq in item2local_freq.items():
        if item not in item2global_freq:
            item2salience[item] = -1
        else:
            salience = (1+math.log(local_freq, 10)**2)*math.log(N_sent_bkg/item2global_freq[item] , 10)
            item2salience[item] = salience
    return item2salience
    

def get_salient_frequent_verb_lemmas(verb2local_freq, verb2global_freq, top_ratio=0.8, min_freq=5):
    verb2salience = get_salience(verb2local_freq, verb2global_freq)    
    stopword_verbs = stop_words | {
        "could", "can", "may", "might", "will", "would", "should", "shall", "be",
        "'d'", ",", '’', "take", "use", "make", "have", "go", "come", "get", "do",
        "give", "put", "set", "argue", "say", "claim", "suggest", "tell", "must", 
        "become", "happen"
    } 

    V = int(len(verb2salience) * top_ratio)
    salient_verbs = {}
    for ele in sorted(verb2salience.items(), key=lambda x:-x[1]):
        if ele[0].lower() not in stopword_verbs:
            salient_verbs[ele[0]] = ele[1]
        if len(salient_verbs) == V:
            break
            
    print(f"select {len(salient_verbs)} salient verbs")

    frequent_salient_verbs = {}
    for verb, saliency in salient_verbs.items():
        if verb2local_freq[verb] >= min_freq:
            frequent_salient_verbs[verb] = saliency
        
    print(f"select {len(frequent_salient_verbs)} frequent and salient verbs")
    return frequent_salient_verbs


def get_salient_frequent_object_heads(oh2local_freq, oh2global_freq, top_ratio=0.8, min_freq=3):
    oh2salience = get_salience(oh2local_freq, oh2global_freq)    

    stopword_nouns = stop_words | {""}
    V = int(len(oh2salience) * top_ratio)
    salient_oh = {}
    for ele in sorted(oh2salience.items(), key=lambda x:-x[1]):
        if ele[0] not in stopword_nouns:
            salient_oh[ele[0]] = ele[1]
        if len(salient_oh) == V:
            break
            
    print(f"select {len(salient_oh)} salient object heads")

    frequent_salient_ohs = {}
    for verb, saliency in salient_oh.items():
        if oh2local_freq[verb] >= min_freq:
            frequent_salient_ohs[verb] = saliency
        
    print(f"select {len(frequent_salient_ohs)} frequent and salient object heads")
    return frequent_salient_ohs

def main(corpus):
    verb2local_freq = defaultdict(int)
    obj2local_freq = defaultdict(int)
    for doc in corpus:
        for em in doc['svos']:
            verb = em[1][0]
            if verb.startswith("!"):
                verb = verb[1:]
            verb2local_freq[verb] += 1
            if em[2] is not None:
                obj = em[2][0]
                obj2local_freq[obj] += 1
   
    verb2local_freq = dict(verb2local_freq)
    obj2local_freq = dict(obj2local_freq)

    return verb2local_freq, obj2local_freq

def get_obj_info(inp):
    obj, local_freq = inp
    if " ".join(obj.split()) != obj:
        return
    parsed_obj = nlp(obj)
    for i, tok in enumerate(parsed_obj):
        if tok.dep_ == "ROOT":
            obj_head_lemma = tok.lemma_
            obj_head_relative_index = i
            break
    obj_info = {
        "obj_head_lemma": obj_head_lemma,
        "obj_head_relative_index": obj_head_relative_index,
    }
    return ((obj, obj_info), (obj_head_lemma, local_freq))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_w_svo_pickle", help="input svo triplet extracted corpus file")
    parser.add_argument("--verb_freq_file", default="./resources/verb_freq.json")
    parser.add_argument("--all_lemma_freq_file", default="./resources/all_lemma_freq.json")
    parser.add_argument("--spacy_model", default="en_core_web_lg", help="Spacy model for POS tagging")
    parser.add_argument("--min_verb_freq", default=5, type=int, help="minimum verb lemma frequency")
    parser.add_argument("--top_verb_ratio", default=0.8, type=float, help="top percentage of selected salient verb lemma")
    parser.add_argument("--min_obj_freq", default=5, type=int, help="minimum object head frequency")
    parser.add_argument("--top_obj_ratio", default=0.8, type=float, help="top percentage of selected salient object head")
    parser.add_argument("--num_threads", type=int, help="Number of threads used to download articles") 
    args = parser.parse_args()
    print(vars(args))

    print("loading Spacy model")
    nlp = spacy.load(args.spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    print("loading corpus")
    with open(args.corpus_w_svo_pickle, "rb") as f:
        corpus = pk.load(f)

    print("loading term frequency files")
    with open(args.verb_freq_file, "r") as f:
        verb2global_freq = json.load(f)

    with open(args.all_lemma_freq_file, "rb") as f:
        lemma2global_freq = json.load(f)

    verb2local_freq, obj2local_freq = main(corpus) 

    obj2local_freq_list = list(obj2local_freq.items())
    parsed_data = []
    with Pool(args.num_threads) as p:
        for result in tqdm(p.imap_unordered(get_obj_info, obj2local_freq_list), total=len(obj2local_freq_list)):
            if result is not None:
                parsed_data.append(result)

    obj_info_items, obj_freq_items = zip(*parsed_data)
    obj2obj_head_info = dict(obj_info_items)
    obj_head2local_freq = defaultdict(int)
    for obj, freq in obj_freq_items:
        obj_head2local_freq[obj] += freq
    obj_head2local_freq = dict(obj_head2local_freq)

    frequent_salient_verbs = get_salient_frequent_verb_lemmas(
        verb2local_freq, verb2global_freq, top_ratio=args.top_verb_ratio, min_freq=args.min_verb_freq)
    
    frequent_salient_object_heads = get_salient_frequent_object_heads(
        obj_head2local_freq, lemma2global_freq, top_ratio=args.top_obj_ratio, min_freq=args.min_obj_freq)

    print("saving selected salient terms")
    with open(f"{args.corpus_w_svo_pickle[:-3]}_salient_verbs.pk", "wb") as f:
        pk.dump(frequent_salient_verbs, f)
    with open(f"{args.corpus_w_svo_pickle[:-3]}_salient_obj_heads.pk", "wb") as f:
        pk.dump(frequent_salient_object_heads, f)
    with open(f"{args.corpus_w_svo_pickle[:-3]}_obj2obj_head_info.pk", "wb") as f:
        pk.dump(obj2obj_head_info, f)