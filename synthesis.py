import extract
from tqdm import tqdm
import json
import sys,os
from wikisql.lib.dbengine import DBEngine
from wikisql.lib.query import Query
from sqlnet.dbengine import DBEngine as DBEngine_s
from copy import deepcopy
import eventlet
from IPython import embed

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<']


def generate_span(q,length):
    spans=[]
    for i in range(length):
        span=[]
        for j in range(len(q)-i):
            span.append(q[j:j+i+1])
        spans.append(span)
    return spans

def get_answer(sql,table_id,engine):
    
    qg=Query.from_dict(sql, ordered=False)
    gold = engine.execute_query(table_id, qg, lower=True)
    return gold

def append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id,is_final=False):
    span_len=len(spans)
    
    for con_col in range(cols):
        con2_col=con_col
        #print(f"con2_col:{con2_col}")
        for op_i,_ in enumerate(cond_ops):
            con2_op=op_i
            #print(f"con2_op:{con2_op}")
            for i in range(span_len):
                for span in spans[i]:
                    _conds=deepcopy(conds)
                    con2=' '.join(span)
                    try:
                        con2 = float(con2)
                    except ValueError:
                        if con2_op >0:
                            continue
                    _conds.append([con2_col,con2_op,con2])
                    sql={'sel': sel,'conds':_conds, 'agg': agg}
                    qg=Query.from_dict(sql, ordered=True)
                    res = engine.execute_query(table_id, qg, lower=True)
                    if res == gold:
                        #print("res is gold")
                        sqls.append(sql)
                    """ if not is_final:
                        if res and not res[0] is None:
                            if agg==0 and set(gold).issubset(set(res)): # agg=0
                                print(f"agg:{agg},con3")
                                sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id,True)        
                            elif agg==1: # agg=MAX
                                try:
                                    gold_ans = float(gold[0])
                                except ValueError:
                                    continue
                                try:
                                    res_ans = float(res[0])
                                except ValueError:
                                    continue
                                if(gold_ans <= res_ans):
                                    print(f"agg={agg},cond3,gold_ans:{gold_ans},res_ans:{res_ans}")
                                    sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id,True)
                            elif agg==2: # agg=MIN
                                try:
                                    gold_ans = float(gold[0])
                                except ValueError:
                                    continue
                                try:
                                    res_ans = float(res[0])
                                except ValueError:
                                    continue
                                if(gold_ans >= res_ans):
                                    print(f"agg:{agg},con3")
                                    sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id,True)
                            elif agg==3: # agg=COUNT
                                try:
                                    gold_ans = float(gold[0])
                                except ValueError:
                                    continue
                                try:
                                    res_ans = float(res[0])
                                except ValueError:
                                    continue
                                if(gold_ans <= res_ans):
                                    print(f"agg:{agg},con3")
                                    sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id,True)
                            elif agg==4: # agg=SUM
                                try:
                                    gold_ans = float(gold[0])
                                except ValueError:
                                    continue
                                try:
                                    res_ans = float(res[0])
                                except ValueError:
                                    continue
                                if(gold_ans <= res_ans):
                                    print(f"agg:{agg},con3")
                                    sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id,True)
                            elif agg==5: # agg=AVG
                                try:
                                    gold_ans = float(gold[0])
                                except ValueError:
                                    continue
                                try:
                                    res_ans = float(res[0])
                                except ValueError:
                                    continue
                                print(f"agg:{agg},con3")
                                sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id,True) """
    return sqls

def generate_sqls(idx,query,engine):
    #generate all possible sqls for the query and answer
    
    table_id=query['table_id']
    table=engine.show_table(table_id)
    gold_sql=query['sql']
    gold=get_answer(gold_sql,table_id,engine)
    cols=len(table[0])
    sqls=[]
    question=query['question'].strip(" ?").split()
    #print(question)
    #print(gold_sql)
    #print(gold)
    span_len=len(question) if len(question)<10 else 10
    spans=generate_span(question,span_len)
    for col in tqdm(range(cols)):
        sel=col
        #print(f"sel:{sel}")
        for agg_i, _ in enumerate(agg_ops):
            agg = agg_i
            #at most 3 conditions
            #print(f"agg:{agg}")
            for con_col in range(cols):# condition 1
                con1_col=con_col
                #print(f"con1_col:{con1_col}")
                for op_i,_ in enumerate(cond_ops):
                    con1_op=op_i
                    #print(f"con1_op:{con1_op}")
                    for i in range(span_len):
                        for span in spans[i]:
                            con1=' '.join(span)
                            try:
                                con1 = float(con1)
                            except ValueError:
                                if con1_op >0:
                                    continue
                            conds=[[con1_col,con1_op,con1]]
                            sql={'sel': sel,'conds':conds, 'agg': agg}
                            qg=Query.from_dict(sql, ordered=True)
                            res = engine.execute_query(table_id, qg, lower=True)
                            if res == gold:
                                sqls.append(sql)
                                #print("res is gold")
                            if res and not res[0] is None :     
                                if agg==0 and set(gold).issubset(set(res)): # agg=0
                                    #print(f"agg={agg},cond2")
                                    sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id)
                                    
                                elif agg==1: # agg=MAX
                                    
                                    try:
                                        gold_ans = float(gold[0])
                                    except ValueError:
                                        continue
                                    try:
                                        res_ans = float(res[0])
                                    except ValueError:
                                        continue
                                    if(gold_ans <= res_ans):
                                        #print(res)
                                        #print(f"agg={agg},cond2,gold_ans:{gold_ans},res_ans:{res_ans}")
                                        sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id)
                                elif agg==2: # agg=MIN
                                    
                                    try:
                                        gold_ans = float(gold[0])
                                    except ValueError:
                                        continue
                                    try:
                                        res_ans = float(res[0])
                                    except ValueError:
                                        continue
                                    if(gold_ans >= res_ans):
                                        #print(f"agg={agg},cond2")
                                        sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id)
                                elif agg==3: # agg=COUNT
                                    try:
                                        gold_ans = float(gold[0])
                                    except ValueError:
                                        continue
                                    try:
                                        res_ans = float(res[0])
                                    except ValueError:
                                        continue
                                    if(gold_ans <= res_ans):
                                        #print(f"agg={agg},cond2")
                                        sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id)
                                elif agg==4: # agg=COUNT
                                    try:
                                        gold_ans = float(gold[0])
                                    except ValueError:
                                        continue
                                    try:
                                        res_ans = float(res[0])
                                    except ValueError:
                                        continue
                                    if(gold_ans <= res_ans):
                                        #print(f"agg={agg},cond2")
                                        sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id)
                                elif agg==5: # agg=AVG
                                    try:
                                        gold_ans = float(gold[0])
                                    except ValueError:
                                        continue
                                    try:
                                        res_ans = float(res[0])
                                    except ValueError:
                                        continue
                                    #print(f"agg={agg},cond2")
                                    sqls=append_cond(sel,cols,agg,spans,conds,sqls,engine,gold,table_id)
    gen_sqls={}
    gen_sqls['id']=idx
    gen_sqls['sqls']=json.dumps(sqls)
    if len(sqls)==0:
        print(idx)
    return gen_sqls


if __name__ == "__main__":
    root='/mnt/sda/qhz/sqlova'
    query_path=os.path.join(root,'data_and_model','train_tok_origin.jsonl')
    table_path=os.path.join(root,'data_and_model','train.tables.jsonl')
    p_sqls_path=os.path.join(root,'data/distant_data','train_distant.jsonl')
    queries=extract.read_queries(query_path)
    p_sqlss=extract.read_potential_sqls(p_sqls_path)
    answer_path='./syn.txt'
    g_answers=extract.read_gold_answers(answer_path)
    print(len(g_answers))
    engine=DBEngine_s('./data_and_model/train.db')
    rr_p_sqlss=[]
    for i,p_sqls in enumerate(tqdm(p_sqlss)):
        rr_p_sqls=[]
        if(len(p_sqls))<3:
            rr_p_sqls=[query['query'] for query in p_sqls]
        else:
            for p_sql in p_sqls:
                qg=Query.from_dict(p_sql['query'], ordered = True)
                res=engine.execute_query(queries[i]['table_id'],qg,lower=True)
                if res == g_answers[i]:
                    rr_p_sqls.append(p_sql['query'])
        if len(rr_p_sqls) == 0:
            print(f"{i}\n")
        rr_p_sqlss.append(rr_p_sqls)
    with open('rr_p.jsonl','w') as f:
        for sqls in rr_p_sqlss:
            f.write(json.dumps(sqls))
            f.write('\n')
    """ engine=DBEngine_s('./data_and_model/train.db')
    sql={"sel": 0, "conds": [[3, 0, "468-473 (6)"]], "agg": 3}
    qg=Query.from_dict(sql, ordered=True)
    res = engine.execute_query("1-10007452-3", qg, lower=True)
    print(res) """

    