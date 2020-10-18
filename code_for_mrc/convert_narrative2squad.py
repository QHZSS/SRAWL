import os
import csv
import json
import argparse

import numpy as np

from tqdm import tqdm
from hotpot_util import find_span
from collections import Counter, defaultdict
import string
from nltk.tokenize import sent_tokenize
from rouge import Rouge
from IPython import embed



data_dir = "/home/sewon/data/narrativeqa"
title_s = "<title>"
title_e = "</title>"

def save(data, data_type):
    file_path = os.path.join(data_dir, '{}.json'.format(data_type))
    with open(file_path, 'w') as f:
        print ("Saving {}".format(file_path))
        json.dump({'data': data}, f)

def split_sentence(context, threshold=200):
    sentences = sent_tokenize(context)
    sent_lengths = [len(sentence.split(' ')) for sentence in sentences]
    contexts = [[]]
    context_lengths = [[]]
    for (sentence, l) in zip(sentences, sent_lengths):
        if sum(context_lengths[-1]) + l <= threshold:
            contexts[-1].append(sentence)
            context_lengths[-1].append(l)
        elif context_lengths[-1][-1] + l <= threshold:
            contexts.append([contexts[-1][-1], sentence])
            context_lengths.append([context_lengths[-1][-1], l])
        else:
            contexts.append([sentence])
            context_lengths.append([l])
    return [" ".join(context) for context in contexts]

def match_context_answer(data):
    for i, article in enumerate(data):
        for j, para in enumerate(article['paragraphs']):
            contexts = para['context']
            for k, qa in enumerate(para['qas']):
                answers = []
                for context in contexts:
                    answers.append([])
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        if answer_text in context:
                            offset = 0
                            while True:
                                if answer_text not in context[offset:]:
                                    break
                                index = context[offset:].index(answer_text) + offset
                                answers[-1].append({'text': answer_text, 'answer_start': index})
                                offset = index+1
                data[i]['paragraphs'][j]['qas'][k]['answers'] = answers
    return data


def main():
    np.random.seed(1995)
    orig_data = []
    with open(os.path.join(data_dir, 'qaps.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            orig_data.append(row)

    data = defaultdict(list)
    for d in orig_data[1:]:
        data[d[1]].append({
                'document_id': d[0], 'question': d[2], 'answer1': d[3], 'answer2': d[4],
                'question_tokenized': d[5], 'answer1_tokenized': d[6], 'answer2_tokenized': d[7]
            })
    doc_orig_data = []
    with open(os.path.join(data_dir, 'third_party/wikipedia/summaries.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            doc_orig_data.append(row)

    doc_data = defaultdict(dict)
    for d in doc_orig_data[1:]:
        doc_data[d[1]][d[0]] = {'summary': d[2], 'summary_tokenized': d[3]}

    # This is for saving
    for data_type in ['train']: #data:
        print ("*** {} set: {} questions, {} stories ***".format(data_type, len(data[data_type]), len(doc_data[data_type])))
        data_list = process(data[data_type], doc_data[data_type], is_training=data_type=='train')
        save(data_list, data_type+"3")

def process(data, doc_data, is_training):
    rouge = Rouge()

    n_ques_tokens, n_para_tokens, n_answer_tokens = [], [], []
    is_span = []
    max_rouge = []

    punkt = string.punctuation
    def is_text(text):
        return np.any([c not in punkt for c in text])

    for doc_id, doc in tqdm(doc_data.items()):
        candidates = []
        tokens = doc['summary_tokenized'].lower().split(' ')
        for start in range(len(tokens)):
            for end in range(start+1, len(tokens)):
                if end-start>40: break
                candidates.append(' '.join(tokens[start:end]))
        doc_data[doc_id]['candidates'] = [c for c in candidates if is_text(c)]

    data_list = []
    n_candidates = []
    for i, d in tqdm(enumerate(data)):
        paragraph = doc_data[d['document_id']]['summary_tokenized'].lower()
        question = d['question_tokenized'].lower()
        answer1 = d['answer1_tokenized'].lower()
        answer2 = d['answer2_tokenized'].lower()

        n_ques_tokens.append(len(question.split(' ')))
        n_para_tokens.append(len(paragraph.split(' ')))
        for answer in [answer1, answer2]:
            n_answer_tokens.append(len(answer.split(' ')))

        ### computing maximum rouge-l score

        def compute_maximum_rouge(candidates):
            scores1 = [score['rouge-l']['f'] for score in rouge.get_scores(candidates, [answer1]*len(candidates))]
            scores2 = [score['rouge-l']['f'] for score in rouge.get_scores(candidates, [answer2]*len(candidates))]
            scores = [max(s1, s2) for s1, s2 in zip(scores1, scores2)]
            if max(scores)<0.5 and is_training:
                return [], 0
            max_score = max(0.5, sorted(scores, reverse=True)[4]) #np.max(scores)
            max_candidates = [candidate for candidate, score in zip(candidates, scores) \
                                    if score>=max_score]
            return max_candidates, max(scores)

        max_candidates, max_score = compute_maximum_rouge(doc_data[d['document_id']]['candidates'])
        max_rouge.append(max_score)
        n_candidates.append(len(max_candidates))

        if is_training and max_score==0:
            continue

        answers = []
        paragraphs = split_sentence(paragraph)

        for paragraph in paragraphs:
            answers.append([])
            for candidate in max_candidates:
                offset = 0
                while True:
                    if candidate not in paragraph[offset:]:
                        break
                    index = paragraph[offset:].index(candidate) + offset
                    answers[-1].append({'text': candidate, 'answer_start': index})
                    offset = index+1

        for (paragraph, answer) in zip(paragraphs, answers):
            for a in answer:
                assert a['text'] == paragraph[a['answer_start']:a['answer_start']+len(a['text'])]

        paragraph = {
                'context': paragraphs,
                'qas': [{
                    'final_answers': max_candidates,
                    'question': question,
                    'answers': answers,
                    'id': i,
                }]
            }
        data_list.append({'title': '', 'paragraphs': [paragraph]})

    def print_statistic(name, l):
        print ("# token %s: Avg %.2f Med %.2f Max %d Min %d" % (
            name, np.mean(l), np.median(l), np.max(l), np.min(l)))

    print_statistic('paragraph', n_para_tokens)
    print_statistic('question', n_ques_tokens)
    print_statistic('answer', n_answer_tokens)
    print_statistic('n_candidates (|Z|)', n_candidates)

    print ("%.2f%% have answer in span" % (100.0*np.mean(is_span)))
    print ("Max Rouge-L: %.2f%% " % (100.0*np.mean(max_rouge)))

    return data_list


if __name__ == '__main__':
    main()
