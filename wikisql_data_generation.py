import os
import re
import string
import json
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from IPython import embed

import spacy
nlp = spacy.load('en_core_web_sm')
from argparse import ArgumentParser

from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))
agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<']
syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']

N = 282

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_type', type=str, default="dev", help='source file for the prediction')
    args = parser.parse_args()

    with open('data_and_model/train_tok.jsonl', 'r') as f:
        lines = f.readlines()
    process(lines, load_table_schema(args.data_type))

def process(lines, id2table):

    data = []
    three_fields = Counter()
    num_conds = Counter()
    conds_three_fields = Counter()
    for line in lines:
        data.append(json.loads(line))

    type_counter = defaultdict(list)
    constant_counter = Counter()
    num_constants = []
    n_ds_candidates = Counter()

    queries_covered = Counter()
    num_queries = []
    all_queries = []
    agg_counter, condi_op_counter, sel_counter = Counter(), Counter(), Counter()
    error_cases = Counter()

    np.random.seed(1995)
    indices = np.random.permutation(range(len(data)))
    data = [data[index] for index in indices]

    for data_index, d in tqdm(enumerate(data)):
        three_fields[set(d['sql'].keys()) == set(['agg', 'conds', 'sel'])]+=1
        agg_counter[d['sql']['agg']]+=1
        sel_counter[d['sql']['sel']]+=1
        num_conds[len(d['sql']['conds'])]+=1
        for cond in d['sql']['conds']:
            conds_three_fields[len(cond)]+=1
            condi_op_counter[cond[1]]+=1

        header = id2table[d['table_id']]['header']
        header2set = id2table[d['table_id']]['header2set']
        #question_tokens = d['question'].lower().split(' ')
        question_tokens, question_entities = extract_ents(d['question'])

        possible_queries = {'agg': list(range(6)), 'sel': list(range(len(header)))}
        '''
        if question_tokens[0] in ['what', "what's", 'which']:
            if question_tokens[0] in ['what', "what's"] and len(set(['total', 'sum'])&set(question_tokens[1:]))>0:
                type_ = (0, 'Sum', d['sql']['agg'])
                matches = ['tot' in h or 'sum' in h for h in header]
                if not any(matches):
                    possible_queries['agg'] = [3, 4]
            elif question_tokens[0] in ['what', "what's"] and len(set(['average', 'avg'])&set(question_tokens[1:]))>0:
                type_ = (0, 'Avg', d['sql']['agg'])
                matches = ['average' in h or 'avg' in h for h in header]
                if not any(matches):
                    possible_queries['agg'] = [5]
            else:
                type_ = (0, 'select')
                possible_queries['agg'] = [0, 1, 2, 3]

        elif question_tokens[:2] == ['how', 'many']:
            matches = [question_tokens[1].startswith(h) or h.startswith(question_tokens[1]) for h in header]
            if sum(matches)==0:
                type_ = (1, 'Fail')
            else:
                type_ = (1, sum(matches), matches[d['sql']['sel']], d['sql']['agg']==3)
                possible_queries['agg'] = [3]
        else:
            type_ = (-1, 'etc')
        type_counter[type_].append(d)
        '''
        # ------ compute all queries ----- #
        header2set_inquestion = defaultdict(set)
        for i, entities in header2set.items():
            for entity in entities:
                add = False
                for s in [str(entity), str(entity).replace(',', '')]:
                    if s in d['question'].lower():
                        add = True
                    if len(s)>15 and s.replace(' ','') in d['question'].lower().replace(' ', ''):
                        add = True
                if add:
                    header2set_inquestion[i].add(entity)
        all_numbers = set()
        types = id2table[d['table_id']]['types']
        if 'real' in types:
            for token in question_tokens:
                token = token.replace(',','')
                try:
                    token = int(token)
                except Exception:
                    try:
                        token = float(token)
                    except Exception:
                        continue
                assert type(token) != str
                all_numbers.add(token)
            #for (ent, ty) in question_entities:
            #    if ty=='CARDINAL': all_numbers.add(ent)
            #    elif ty=='PERCENT': all_numbers.add(ent.split('%')[0].strip())

        _all_queries = []
        covered=False
        LIMIT = 1000000 #100000

        for agg in possible_queries['agg']:
            for sel in possible_queries['sel']:
                conditions = []
                for i, h in enumerate(header):
                    if i==sel: continue
                    if types[i] == 'text':
                        for ent in header2set_inquestion[i]:
                            conditions.append([i, 0, ent])
                    elif types[i] == 'real':
                        for number in all_numbers:
                            if min(header2set[i]) <= number <= max(header2set[i]):
                                conditions.append([i, 0, number])
                                conditions.append([i, 1, number])
                                conditions.append([i, 2, number])
                    else:
                        raise NotImplementedError("Does not support type [{}]".format(types[i]))
                all_conditions, prev_new_conditions = [[]], [[]]
                n_possible_conditions = len([s for s in header2set_inquestion.values() if len(s)>0]) + types.count('real')

                while len(all_conditions[-1]) < min(3, n_possible_conditions):
                    new_conditions = []
                    for i, condi1 in enumerate(prev_new_conditions):
                        #covered_columns = set([c[0] for c in condi1])
                        #covered_const = set([c[2] for c in condi1 if type(c)!=str])
                        for j, condi2 in enumerate(conditions):
                            if len(condi1)==0 or condi2[0] >= condi1[-1][0]:
                            #if condi2[0] not in covered_columns: # and condi2[2] not in covered_const:
                                new_conditions.append(condi1.copy() + [condi2])
                    if len(new_conditions) == 0:
                        break
                    all_conditions += new_conditions
                    prev_new_conditions = new_conditions

                _all_queries += [{'query': {'agg': agg, 'sel': sel, 'conds': c}} for c in all_conditions]
                if len(_all_queries) > LIMIT: break

                if agg==d['sql']['agg'] and sel==d['sql']['sel']:
                    conds = map_conds_to_str(d['sql']['conds'])
                    if conds in [map_conds_to_str(c) for c in all_conditions]:
                        covered=True
                    '''
                    elif len(conds)<=3:
                        print (conds)
                        print (d['sql'])
                        print (d['question'])
                        embed()'''

        if len(_all_queries) > LIMIT:
            _all_queries = []
            covered=False
            error = "Too many queries"
        elif d['sql']['agg'] not in possible_queries['agg']:
            assert not covered
            error = "Miss agg"
            '''print ()
            print (d['question'])
            print (type_)
            print (d['sql'])
            embed()'''
        elif d['sql']['sel'] not in possible_queries['sel']:
            assert not covered
            error = "Miss sel"
        elif not covered and len(d['sql']['conds'])>3:
            error = "Too many conditions"
        elif not covered:
            error = "Miss condition"

        if not covered:
            error_cases[error] += 1

        queries_covered[covered]+=1

        all_queries.append(_all_queries)
        num_queries.append(len(_all_queries))


        if data_index % 200 == 0:
            print (np.mean(num_queries), np.median(num_queries))
            print ("%.2f%% covered" % (100.0*queries_covered[True]/sum(queries_covered.values())))
            print (error_cases)

        # ------------------- save candidiates -----------------
        # train data 56355. Save into 940 files with 60 datapoints
        # Or 282 * 200

    print (np.mean(num_queries))
    print ("%.2f%% covered" % (100.0 * queries_covered[True]/sum(queries_covered.values())))
    return data, all_queries

def map_conds_to_str(conds):
    for i, cond in enumerate(conds):
        conds[i][2]=str(cond[2]).replace(',', '').replace(' ', '').lower()
    return sorted(conds, key=lambda x: x[0])

def extract_ents(text):

    def _extract_ents(text, doc=None):
        if doc is None:
            doc = nlp(text)
        tokens = [(t.text, t.tag_) for t in doc]
        entities = [(ent.text, ent.start, ent.end, ent.label_) for ent in doc.ents]

        tags = [i for i, (token, tag) in enumerate(tokens) if tag in ["``", "''"]]
        for i1, i in enumerate(tags):
            for j in tags[i1+1:]:
                if j-i<=5:
                    entities.append((" ".join([e[0] for e in tokens[i+1:j]]), i, j, 'TEXT'))


        i = 0
        while i<len(tokens):
            entity = []
            while i<len(tokens) and tokens[i][0][0].isupper():
                entity.append(tokens[i][0])
                i+=1
            if len(entity)>1:
                entities.append((" ".join(entity), i-len(entity), i, 'CAP'))
            i+=1

        new_entities = []
        for (text, start, end, label) in sorted(entities, key=lambda x: x[2]-x[1], reverse=True):
            put = True
            for (text1, start1, end1, label1) in new_entities:
                if start1<=start<=end<=end1:
                    put = False
                    break
            if put:
                new_entities.append((text, start, end, label))

        non_punct = [t.text for t in doc if t.pos_!='PUNCT']
        if len(tokens)>len(non_punct):
            new_entities += _extract_ents(" ".join(non_punct))

        return new_entities

    doc1 = nlp(text)
    entities = _extract_ents(text, doc1)
    entities = list(set([(e[0], e[3]) for e in entities if len(e[0])>0]))

    new_entities = []
    for i, (text, label) in enumerate(entities):
        others = entities[:i] + entities[i+1:]
        if text.islower() and any([t.lower()==text and l==label and not t.islower() for t, l in others]):
            continue
        new_entities.append((text, label))

    return [t.text.lower()  for t in doc1], new_entities


def normalize_string(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def load_table_schema(data_type):
    with open('data_and_model/train.tables.jsonl', 'r') as f:
        lines = f.readlines()
    id2table = {}
    for line in lines:
        d = json.loads(line)
        header = [t.lower() for t in d['header']]
        rows = d['rows']
        assert all([len(r)==len(header) for r in rows])
        header2set = defaultdict(set)
        for row in rows:
            for i, col in enumerate(row):
                if type(col)==str:
                    if d['types'][i] == 'text':
                        col=col.lower()
                    else:
                        col = col.replace(',', '')
                        try:
                            col = int(col)
                        except Exception:
                            col = float(col)
                header2set[i].add(col)
        id2table[d['id']] = {'header': header,
                             'header2set': header2set,
                             'types': d['types']}
    return id2table



if __name__ == '__main__':
    main()
