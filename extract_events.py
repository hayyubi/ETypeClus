import pickle
import json
import argparse
import spacy
from extract_svo import findSVOs
from tqdm import tqdm
from multiprocessing import Pool

SPACY_MODEL = 'en_core_web_lg'

def extract_events(k):
    # k = k.split('.mp4')[0]
    v = input_data[k]
    sen_and_svos_list = []
    article = v['article']
    article = k + '. ' + article

    # extracting events
    lines = article.split('\n')
    sent_id = 0
    for line in lines:
        if not line:
            continue
        doc = nlp(line)
        for sent in doc.sents:
            sent_id += 1
            new_doc = nlp(sent.text)

            token_list = [token.text for token in new_doc]
            raw_sent = " ".join(token_list)
            svos = findSVOs(new_doc)
            verb_set = set()
            num_event_marked = 0
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

                # # Select verbs which have objects
                # if svo[2] is None:
                #     continue

                # # Select from only salient objects
                # obj = svo[2][0]
                # parsed_obj = nlp(obj)
                # for i, tok in enumerate(parsed_obj):
                #     if tok.dep_ == "ROOT":
                #         obj_head_lemma = tok.lemma_
                # if obj_head_lemma not in salient_obj_heads:
                #     continue

                event_char_id = pred_token_id + sum([len(token_list[i]) for i in range(pred_token_id)])
                if mark_event:
                    mod_char_id = event_char_id + num_event_marked * 3
                    raw_sent = raw_sent[:mod_char_id] + '[[[' + raw_sent[mod_char_id:]
                    num_event_marked += 1
                else:
                    event_mention = raw_sent[event_char_id:].split(' ')[0]
                    event = {'sent': raw_sent, 'event_char_id': event_char_id, 'event_mention': event_mention}
                    sen_and_svos_list.append(event)

            if mark_event:
                sen_and_svos_list.append(raw_sent)

    if mark_event:
        output = (k, ''.join(sen_and_svos_list))
    else:
        output = (k, sen_and_svos_list)
        
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='outputs/m2e2/m2e2_test_100_wcaptions.json', help="input corpus file")
    parser.add_argument("--video_list", default=None, help="List containing names of videos to be extracted events from")
    parser.add_argument("--save_file", default='outputs/m2e2_extracted_events.json', help="output file events in joint constrained learning format")
    parser.add_argument("--salient_verb_file", default='../data/extracted_events/svos_salient_verbs.pk', help="verbs to choose from")
    parser.add_argument("--salient_obj2obj_head_info", default='../data/extracted_events/svos_obj2obj_head_info.pk', help="object2 object heads info")
    parser.add_argument("--salient_obj_heads", default='../data/extracted_events/svos_salient_obj_heads.pk', help="object heads to choose from")
    parser.add_argument("--num_threads", dest="num_threads", type=int, default=1, help="Number of threads used to download articles")
    parser.add_argument("--mark_events", action='store_true')
    args = parser.parse_args()

    mark_event = args.mark_events
    nlp = spacy.load(SPACY_MODEL)
    with open(args.input_file) as f:
        input_data = json.load(f)

    if args.video_list:
        with open(args.video_list, 'rb') as f:
            video_list = pickle.load(f)
    else:
        video_list = list(input_data.keys())

    with open(args.salient_verb_file, 'rb') as f:
        salient_verbs = list(pickle.load(f).keys())

    # with open(args.salient_obj2obj_head_info, 'rb') as f:
    #     salient_obj2obj_head_info = pickle.load(f)

    # with open(args.salient_obj_heads, 'rb') as f:
    #     salient_obj_heads = pickle.load(f)

    extracted_events = []
    with Pool(args.num_threads) as p:
        for result in tqdm(p.imap_unordered(extract_events, video_list), total=len(video_list)):
            extracted_events.append(result)

    with open(args.save_file, 'w') as f:
        json.dump(dict(extracted_events), f)
