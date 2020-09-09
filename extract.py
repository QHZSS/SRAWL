from tqdm import tqdm
import json
import sys
import os
import re
from IPython import embed

def read_tables(filepath):
    tables=[]
    with open(filepath,'r') as f:
        lines=f.readlines()
        for line in lines:
            tables.append(json.loads(line))
    table_dic={}
    for table in tables:
        table_dic[table['id']]=json.dumps(table)
    return table_dic

def read_queries(filepath):
    queries=[]
    with open(filepath,'r') as f:
        lines=f.readlines()
        for line in lines:
            queries.append(json.loads(line))
    return queries

def read_questions(filepath):
    questions=[]
    with open(filepath,'r') as f:
        lines=f.readlines()
        for line in lines:
            questions.append(json.loads(line)['question'])
    return questions

def read_gold_sqls(filepath):
    g_sqls=[]
    with open(filepath,'r') as f:
        lines=f.readlines()
        for line in lines:
            g_sqls.append(json.loads(line)['sql'])
    return g_sqls

def read_table_headers(filepath):
    tables=[]
    with open(filepath,'r') as f:
        lines=f.readlines()
        for line in lines:
            tables.append(json.loads(line))
    headers={}
    for table in tables:
        headers[table['id']]=table['header']
    return headers


def read_gold_answers(filepath):
    answers=[]
    with open(filepath,'r') as f:
        lines=f.readlines()
        for line in lines:
            answers.append(eval(line))
    return answers

def read_potential_sqls(filepath):
    p_sqlss=[]
    with open(filepath,'r') as f:
        lines=f.readlines()
        for line in tqdm(lines):
            query_list=json.loads(line)
            query_dist=[{'query' : query} for query in query_list]
            p_sqlss.append(query_dist)
    return p_sqlss

if __name__ == '__main__':

    root='/mnt/sda/qhz/sqlova'
    query_path=os.path.join(root,'data_and_model','train_tok_process_9081006.jsonl')
    queries=read_queries(query_path)
    #answer_path='./syn.txt'
    #answers=read_gold_answers(answer_path)
    with open('./data_and_model/train_tok_9081006.jsonl','w') as f:
        for i,query in enumerate(tqdm(queries)):
            sqls=query['sql']
            question=query['question']
            question_tok=query['question_tok']
            wvis_corenlp=[]
            tok2idx=[]
            question_lower=re.sub('[ ,\u00a0]','',question).lower()
            beg=0
            for tok in question_tok:
                tok_pro=re.sub('[ ,\u00a0]','',tok).lower()
                pair=[tok_pro,question_lower.find(tok_pro,beg)]
                if len(pair[0])==0:
                    pair[1]=tok2idx[-1][1]
                tok2idx.append(pair)
                beg=tok2idx[-1][1]+len(tok2idx[-1][0])
            tok2idx.append(['',sys.maxsize])
            for sql in sqls:
                wvs=[cond[2] for cond in sql['conds']]
                wvi_corenlp=[]
                for wv in wvs:
                    wv=re.sub('[ ,\u00a0]','',str(wv).lower())
                    start=question_lower.find(wv)
                    if start==-1:
                        if wv=='0.667' and i==10527:
                            wvi_corenlp.append([4,4])
                        elif wv=='100000000' and i== 54372:
                            wvi_corenlp.append([11,11])
                        elif wv=='100000' and i== 54375:
                            wvi_corenlp.append([12,12])
                        else:
                            print(i)
                            print(question_lower)
                            print(wv)
                            wvi_corenlp.append([-1,-1])
                        continue
                    end=start+len(wv)
                    wvi_corenlp_i=[]
                    for j,t2i in enumerate(tok2idx):
                        tok_start=t2i[1]
                        if start == tok_start and len(wvi_corenlp_i)==0:
                            wvi_corenlp_i.append(j)
                        if end <= tok_start and len(wvi_corenlp_i)==1:
                            tmp=j-2 if tok2idx[j-1][0]=='' else j-1
                            wvi_corenlp_i.append(tmp)
                            break
                    if len(wvi_corenlp_i)==0:#when the tokens may find in another words
                        tok_pros=[x[0] for x in tok2idx]
                        if wv not in tok_pros:
                            """ print(i)
                            print(wv)
                            print(question_lower)
                            print('\n') """
                            wvi_corenlp_i=[-1,-1]
                        else: 
                            for j,t2i in enumerate(tok2idx):
                                tok_start=j
                                if wv == t2i[0]:
                                    wvi_corenlp_i=[j,j]
                                    break
                    wvi_corenlp.append(wvi_corenlp_i)
                assert len(wvs)==len(wvi_corenlp)
                wvis_corenlp.append(wvi_corenlp)
            assert len(wvis_corenlp)==len(sqls)
            query['wvis_corenlp']=wvis_corenlp
            if [-1,-1] in query['wvis_corenlp'][0]:
                flag=False
                for k,wvis in enumerate(query['wvis_corenlp']):
                    if [-1,-1] not in wvis:
                        flag=True
                        query['wvis_corenlp'][0]=query['wvis_corenlp'][k]
                        query['sql'][0]=query['sql'][k]
                        break
                if not flag:
                    continue
                for g,wvis in enumerate(query['wvis_corenlp']):
                    if [-1,-1] in wvis:
                        query['wvis_corenlp'][g]=query['wvis_corenlp'][0]
                        query['sql'][g]=query['sql'][0]

            else:
                for g,wvis in enumerate(query['wvis_corenlp']):
                    if [-1,-1] in wvis:
                        query['wvis_corenlp'][g]=query['wvis_corenlp'][0]
                        query['sql'][g]=query['sql'][0]
            g_wvi=query['wvi_corenlp']
            if g_wvi:
                for wvi1 in wvis_corenlp:
                    if any([idx[0] >= len(question_tok) for idx in wvi1]) or any([idx[1] >= len(question_tok) for idx in wvi1]):
                        embed()
            f.write(json.dumps(query))
            f.write('\n')
                    









            """ if(len(conds_g)==2):
                count+=1
                conds_r_wv=[re.sub('[ ,]','',str(cond[2])).lower() for cond in conds_r]
                conds_g_wv=[re.sub('[ ,]','',str(cond[2])).lower() for cond in conds_g]
                if conds_r_wv[0]==conds_g_wv[0] and conds_r_wv[1]==conds_g_wv[1]:
                    count_eq+=1
                elif conds_r_wv[0]==conds_g_wv[1] and conds_r_wv[0]==conds_g_wv[1]:
                    query['wvi_corenlp'].reverse()
                else:
                    query['ignore']=1
                    #print(f"ranked:{conds_r_wv}")
                    #print(f"gold:{conds_g_wv}")
            if(len(conds_g)==3):
                count_3+=1
                conds_r_wv=[re.sub('[ ,]','',str(cond[2])).lower() for cond in conds_r]
                conds_g_wv=[re.sub('[ ,]','',str(cond[2])).lower() for cond in conds_g]
                conds_r_wv.sort()
                conds_g_wv.sort()
                if(conds_r_wv==conds_g_wv):
                    wv2wvi={}
                    for i,item in enumerate(conds_g_wv):
                        wv2wvi[item]=query['wvi_corenlp'][i]
                    new_wvi_corenlp=[wv2wvi[item] for item in conds_r_wv]
                    query['wvi_corenlp']=new_wvi_corenlp
                else:
                    query['ignore']=1
            if not query.__contains__('ignore'):
                f.write(json.dumps(query))
                f.write('\n') """
    
