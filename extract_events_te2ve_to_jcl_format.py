import pickle
import json
import argparse
import spacy
from extract_svo import findSVOs
from tqdm import tqdm
import re

SPACY_MODEL = 'en_core_web_lg'

def extract_events(annots):
    nlp = spacy.load(SPACY_MODEL)
    output = {}
    for vid_name, annots in tqdm(annots.items()):
        article = vid_name + '\n' + test_data[vid_name]['article']
        unique_events = set()
        for annot in annots:
            event = (annot['text_start'], annot['text_end'])
            unique_events.add(event)

        all_events_with_sents = []
        sen_beg_end_pattern = re.compile(r'[\r\n\.]')
        for event in unique_events:
            event_start_in_article = event[0]
            event_end_in_article = event[1]

            event_sent_start_match = sen_beg_end_pattern.search(article[event_start_in_article::-1])
            if event_sent_start_match:
                event_start_char = event_sent_start_match.start() - 1
            else:
                event_start_char = event_start_in_article
            event_sent_start = event_start_in_article - event_start_char
            event_sent_end_match = sen_beg_end_pattern.search(article, event_end_in_article)
            if event_sent_end_match:
                event_sent_end = event_sent_end_match.start()
            else:
                event_sent_end = len(article)
            event_sent = article[event_sent_start: event_sent_end + 1]
            if event_sent[event_start_char-1] != ' ':
                event_sent = event_sent[:event_start_char] + ' ' + event_sent[event_start_char:]
                event_start_char = event_start_char + 1
            event_mention = event_sent[event_start_char:].split(' ')[0]

            event_info = {
                'sent': event_sent,
                'event_char_id_relative_to_sen': event_start_char,
                'event_char_begin_id_relative_to_article': event_start_in_article,
                'event_char_end_id_relative_to_article': event_end_in_article,
                'event_mention': event_mention
            }
            all_events_with_sents.append(event_info)

        possible_event_event_relations = []
        for i in range(len(all_events_with_sents)):
            for j in range(i+1, len(all_events_with_sents)):
                possible_event_event_relations.append({
                    'sent_1': all_events_with_sents[i]['sent'],
                    'e1_start_char': all_events_with_sents[i]['event_char_id_relative_to_sen'],
                    'e1_mention': all_events_with_sents[i]['event_mention'],
                    'e1_char_begin_id_relative_to_article': all_events_with_sents[i]['event_char_begin_id_relative_to_article'],
                    'e1_char_end_id_relative_to_article': all_events_with_sents[i]['event_char_end_id_relative_to_article'],
                    'sent_2': all_events_with_sents[j]['sent'],
                    'e2_start_char': all_events_with_sents[j]['event_char_id_relative_to_sen'],
                    'e2_mention': all_events_with_sents[j]['event_mention'],
                    'e2_char_begin_id_relative_to_article': all_events_with_sents[j]['event_char_begin_id_relative_to_article'],
                    'e2_char_end_id_relative_to_article': all_events_with_sents[j]['event_char_end_id_relative_to_article'],
                })

        output[vid_name] = possible_event_event_relations

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--annots_file", default='../data/annotations/validation.json', help="input corpus file")
    parser.add_argument("--test_file", default='../data/test.json', help="output file events in joint constrained learning format")
    parser.add_argument("--save_file", default='../data/extracted_events_test/val_split/val_events_annots_converted_to_jcl_format_debugged.json', help="output file events in joint constrained learning format")
    args = parser.parse_args()

    with open(args.annots_file) as f:
        annots_data = json.load(f)

    with open(args.test_file) as f:
        test_data = json.load(f)

    extracted_events = extract_events(annots_data)

    with open(args.save_file, 'w') as f:
        json.dump(extracted_events, f)
