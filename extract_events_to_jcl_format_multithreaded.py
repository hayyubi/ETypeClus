import pickle
import json
import argparse
import spacy
from extract_svo import findSVOs
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Process
import os
import glob
import shutil
import numpy as np

SPACY_MODEL = 'en_core_web_lg'

def extract_events(inp, salient_verbs, nlp, save_file_prefix):
    output = []
    file_no = 0
    for k, v in tqdm(inp):
        sen_and_svos_list = []
        article = v['article']
        article = k + '. ' + article

        # extracting events
        lines = article.split('\n')
        for line in lines:
            if not line:
                continue
            doc = nlp(line)
            for sent in doc.sents:
                new_doc = nlp(sent.text)

                token_list = [token.text for token in new_doc]
                raw_sent = " ".join(token_list)
                svos = findSVOs(new_doc)
                verb_set = set()
                for svo in svos:
                    verb = tuple(svo[1])
                    if verb in verb_set:
                        continue
                    verb_set.add(verb)
                    pred = svo[1][0]
                    pred_token_id = svo[1][1]

                    # Select only salient verbs
                    if pred not in salient_verbs:
                        continue

                    event_char_id = pred_token_id + sum([len(token_list[i]) for i in range(pred_token_id)])
                    event_mention = raw_sent[event_char_id:].split(' ')[0]
                    event = {'sent': raw_sent, 'event_char_id': event_char_id, 'event_mention': event_mention}
                    sen_and_svos_list.append(event)

        # Formatting event event relations for predicting their relations
        possible_event_event_relations = []
        for i in range(len(sen_and_svos_list)):
            for j in range(i+1, len(sen_and_svos_list)):
                possible_event_event_relations.append({
                    'sent_1': sen_and_svos_list[i]['sent'],
                    'e1_start_char': sen_and_svos_list[i]['event_char_id'],
                    'e1_mention': sen_and_svos_list[i]['event_mention'],
                    'sent_2': sen_and_svos_list[j]['sent'],
                    'e2_start_char': sen_and_svos_list[j]['event_char_id'],
                    'e2_mention': sen_and_svos_list[j]['event_mention'],
                })

        output.append((k, possible_event_event_relations))
        if len(output) % 1000 == 0:
            with open(save_file_prefix + '_{}.json'.format(file_no), 'w') as f:
                json.dump(dict(output), f)
            file_no += 1
            output = []

    with open(save_file_prefix + '_{}.json'.format(file_no), 'w') as f:
        json.dump(dict(output), f)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='outputs/m2e2/m2e2_test_100_wcaptions.json', help="input corpus file")
    parser.add_argument("--save_file", default='outputs/m2e2_extracted_events_check2.json', help="output file events in joint constrained learning format")
    parser.add_argument("--salient_verb_file", default='../data/extracted_events/svos_salient_verbs.pk', help="verbs to choose from")
    parser.add_argument("--num_threads", default = 96, type=int, help="Number of threads used to download articles") 
    args = parser.parse_args()

    nlp = spacy.load(SPACY_MODEL)
    with open(args.input_file) as f:
        input_data = json.load(f)

    with open(args.salient_verb_file, 'rb') as f:
        salient_verbs = pickle.load(f)

    already_extracted = {}
    if os.path.exists(args.save_file):
        with open(args.save_file) as f:
            already_extracted = json.load(f)

    for key in already_extracted:
        del(input_data[key])

    input_data = list(input_data.items())

    temp_dir = args.save_file.split('.json')[0] + '_temp'
    os.makedirs(temp_dir, exist_ok=True)

    vid_split_idxs = np.linspace(0, len(input_data), num=args.num_threads, endpoint=False, dtype=int)
    vid_split_idxs = np.unique(vid_split_idxs)

    vid_splits = []
    i=0
    for i in range(1, vid_split_idxs.shape[0]):
        vid_splits.append(input_data[vid_split_idxs[i-1]:vid_split_idxs[i]])

    vid_splits.append(input_data[vid_split_idxs[i]:])

    processV = []
    for i, vid_split in enumerate(vid_splits):
        save_file_prefix = os.path.join(temp_dir, '{}'.format(i))
        processV.append(Process(target=extract_events, args = (vid_split, salient_verbs, nlp, save_file_prefix)))

    for i in range(len(vid_splits)):
        processV[i].start()

    for i in range(len(vid_splits)):
        processV[i].join()

    print('Joining temporary files')
    e2e_joined = {}
    if os.path.exists(args.save_file):
        with open(args.save_file) as f:
            e2e_joined = json.load(f)

    for file in glob.glob(os.path.join(temp_dir, '*.json')):
        with open(file) as f:
            data = json.load(f)

        e2e_joined.update(data)

    print('Joined temporary files. Removing temp files.')
    shutil.rmtree(temp_dir)
    with open(args.save_file, 'w') as f:
        json.dump(e2e_joined, f)