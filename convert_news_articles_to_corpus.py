import json
import argparse

def convert_to_corpus(inp):
    articles_list = []
    for k, v in inp.items():
        article = k + '. ' + v['article']
        articles_list.append(article)

    corpus = '\n'.join(articles_list)
    return corpus
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='outputs/m2e2_test_100_wcaptions.json', help="input corpus file")
    parser.add_argument("--save_file", default='outputs/m2e2/m2e2_100_corpus.json', help="output file events in joint constrained learning format")
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    corpus = convert_to_corpus(data)

    with open(args.save_file, 'w') as f:
        f.write(corpus)

