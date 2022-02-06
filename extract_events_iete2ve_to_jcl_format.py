import pickle
import json
import argparse
import spacy
from extract_svo import findSVOs
from tqdm import tqdm

SPACY_MODEL = 'en_core_web_lg'

def extract_events(inp, salient_verbs):
    nlp = spacy.load(SPACY_MODEL)
    output = {}
    for k, v in tqdm(inp.items()):
        sen_and_svos_list_with_event_position_relative_to_article = []
        events_in_eval_format = []
        article = v['article']
        article = k + '\n' + article

        # extracting events
        lines = article.split('\n')
        sent_id = 0
        cur_sent_char_id = 0
        first_line = True
        for line in lines:
            if first_line:
                first_line = False
            else:
                cur_sent_char_id += 1
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

                    event_begin_id = cur_sent_char_id + sum([len(new_doc[i].text_with_ws) for i in range(pred_token_id)])
                    event_end_id = event_begin_id + len(token_list[pred_token_id])
                    events_in_eval_format.append((event_begin_id, event_end_id, event_mention))

                    event = {'sent': raw_sent, 'event_char_id_relative_to_sen': event_char_id, 
                             'event_char_begin_id_relative_to_article': event_begin_id,
                             'event_char_end_id_relative_to_article': event_end_id,
                             'event_mention': event_mention}
                    sen_and_svos_list_with_event_position_relative_to_article.append(event)

                cur_sent_char_id += sum([len(token.text_with_ws) for token in sent])

        # Form all possible event-event relations
        possible_event_event_relations = []
        for i in range(len(sen_and_svos_list_with_event_position_relative_to_article)):
            for j in range(i+1, len(sen_and_svos_list_with_event_position_relative_to_article)):
                possible_event_event_relations.append({
                    'sent_1': sen_and_svos_list_with_event_position_relative_to_article[i]['sent'],
                    'e1_start_char': sen_and_svos_list_with_event_position_relative_to_article[i]['event_char_id_relative_to_sen'],
                    'e1_mention': sen_and_svos_list_with_event_position_relative_to_article[i]['event_mention'],
                    'e1_char_begin_id_relative_to_article': sen_and_svos_list_with_event_position_relative_to_article[i]['event_char_begin_id_relative_to_article'],
                    'e1_char_end_id_relative_to_article': sen_and_svos_list_with_event_position_relative_to_article[i]['event_char_end_id_relative_to_article'],
                    'sent_2': sen_and_svos_list_with_event_position_relative_to_article[j]['sent'],
                    'e2_start_char': sen_and_svos_list_with_event_position_relative_to_article[j]['event_char_id_relative_to_sen'],
                    'e2_mention': sen_and_svos_list_with_event_position_relative_to_article[j]['event_mention'],
                    'e2_char_begin_id_relative_to_article': sen_and_svos_list_with_event_position_relative_to_article[j]['event_char_begin_id_relative_to_article'],
                    'e2_char_end_id_relative_to_article': sen_and_svos_list_with_event_position_relative_to_article[j]['event_char_end_id_relative_to_article'],
                })

        output[k] = possible_event_event_relations
        
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='outputs/m2e2/m2e2_test_100_wcaptions.json', help="input corpus file")
    parser.add_argument("--save_file", default='outputs/m2e2_extracted_events_checkx.json', help="output file events in joint constrained learning format")
    parser.add_argument("--salient_verb_file", default='../data/extracted_events/svos_salient_verbs.pk', help="verbs to choose from")
    args = parser.parse_args()

    with open(args.input_file) as f:
        input_data = json.load(f)

    with open(args.salient_verb_file, 'rb') as f:
        salient_verbs = pickle.load(f)

    extracted_events = extract_events(input_data, list(salient_verbs.keys()))

    with open(args.save_file, 'w') as f:
        json.dump(extracted_events, f)
