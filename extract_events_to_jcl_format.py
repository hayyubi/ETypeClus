import pickle
import json
import argparse
import spacy
from extract_svo import findSVOs
from tqdm import tqdm

SPACY_MODEL = 'en_core_web_lg'

def extract_events(inp, salient_verbs, salient_obj2obj_head_info, salient_obj_heads):
    nlp = spacy.load(SPACY_MODEL)
    output = {}
    for k, v in tqdm(inp.items()):
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
        
        output[k] = possible_event_event_relations

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='outputs/m2e2/m2e2_test_100_wcaptions.json', help="input corpus file")
    parser.add_argument("--save_file", default='outputs/m2e2_extracted_events_check2.json', help="output file events in joint constrained learning format")
    parser.add_argument("--salient_verb_file", default='../data/extracted_events/svos_salient_verbs.pk', help="verbs to choose from")
    parser.add_argument("--salient_obj2obj_head_info", default='../data/extracted_events/svos_obj2obj_head_info.pk', help="object2 object heads info")
    parser.add_argument("--salient_obj_heads", default='../data/extracted_events/svos_salient_obj_heads.pk', help="object heads to choose from")
    args = parser.parse_args()

    with open(args.input_file) as f:
        input_data = json.load(f)

    with open(args.salient_verb_file, 'rb') as f:
        salient_verbs = pickle.load(f)

    with open(args.salient_obj2obj_head_info, 'rb') as f:
        salient_obj2obj_head_info = pickle.load(f)

    with open(args.salient_obj_heads, 'rb') as f:
        salient_obj_heads = pickle.load(f)

    extracted_events = extract_events(input_data, list(salient_verbs.keys()), salient_obj2obj_head_info, salient_obj_heads)

    with open(args.save_file, 'w') as f:
        json.dump(extracted_events, f)
