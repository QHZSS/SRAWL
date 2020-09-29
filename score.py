from tqdm import tqdm
import json
import sys
import os
import extract
import re
import numpy as np

from wikisql.lib.query import Query
from sqlnet.dbengine import DBEngine as DBEngine_s
from IPython import embed

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def coverage_score(question,sql,header,iter):
    #the more element in sql appear in question , the higher the score is
    score=0
    sel=sql['sel']
    elements=[]
    conds_sels=[cond[0] for cond in sql['conds']]
    conds_values=[cond[2] for cond in sql['conds']]
    header_sel=str(header[sel]).lower()
    header_sel=re.sub("[, ]","",header_sel)
    header_cs=[re.sub("[, ]","",str(header[cs]).lower()) for cs in conds_sels]
    elements.append(header_sel)
    elements+=header_cs
    elements+=[re.sub("[, ]",'',str(cv).lower()) for cv in conds_values]
    question_tok=re.sub("[?,\u00a3]",'',question.lower()).split()
    length=len(question_tok)
    question=re.sub("[? ,\u00a3]",'',question.lower())
    hit=[]
    dup=0
    if header_sel in question:
        hit.append(header_sel)

    for cs in header_cs:
        if cs in question:
            for item in hit:
                if item in cs or cs in item:
                    dup+=1
            hit.append(cs)

    for i,cv in enumerate(conds_values):
        tmp=re.sub("[, ]",'',str(cv).lower())
        if is_number(tmp) and str(tmp) in question_tok:
            for item in hit:
                if item in str(tmp) or str(tmp) in item:
                    dup+=1
            hit.append(tmp)
        elif not is_number(tmp) and  ((len(tmp) <=2 and tmp in question_tok) or (len(tmp) >2 and tmp in question )):
            for item in hit:
                if item in tmp or tmp in item:
                    dup+=1
            hit.append(tmp)
        else:
            if re.sub("[, ]","",header_cs[i]) in question:
                dup+=1
    
    hit=set(hit)
    score=len(hit)-dup

    return score/length * 100

def simplicity_score(sql):
    conds=sql['conds']
    conds_count=len(conds)
    if(conds_count ==0):
        return 100
    elif(conds_count ==1):
        return 75
    elif(conds_count ==2):
        return 50
    elif(conds_count==3):
        return 25
    else:
        return 0

def natrual_score(question,sql,header):
    agg1=['highest','maximum','most','greatest','largest','latest','biggest','best','max']
    agg2=['first','min','minimum','smallest','worst','lowest','least','earliest']
    agg3=['howmany','wahtisthetot','whatisthenumber','whatistot','amountof','thenumberof','totalrank']
    agg4=['sum','total']
    agg5=['average','frequency']
    question=re.sub("[? ,]",'',question.lower())

    sel=sql['sel']
    conds_sels=[cond[0] for cond in sql['conds']]
    header_sel=str(header[sel]).lower()
    header_cs=[str(header[cs]).lower() for cs in conds_sels]

    score=0

    if any([word in question for word in agg1]) and sql['agg']==1:
        score=1
    elif any([word in question for word in agg2]) and sql['agg']==2:
        score=1
    elif any([word in question for word in agg3]) and sql['agg']==3:
        score=1
    elif any([word in question for word in agg4]) and sql['agg']==4:
        score=1
    elif any([word in question for word in agg5]) and sql['agg']==5:
        score=1
    elif sql['agg']==0 and (any([word in header_sel for word in agg1+agg2+agg3+agg4+agg5]) or any([ word in ' '.join(header_cs) for word in agg1+agg2+agg3+agg4+agg5])):
        score=2
    else:
        score=0
    return score *50

def op_score(question,sql,iter):
    op_1=['higher','more','greater','larger','later','bigger','better','longer']
    op_2=['earlier','less','fewer','smaller','worse','lower','under']
    question=re.sub("[? ,]",'',question.lower())
    conds_ops=[cond[1] for cond in sql['conds']]
    gtl=[ word in question for word in op_1]
    ltl=[ word in question for word in op_2]
    gt=0
    lt=0
    for i in gtl:
        if i:
            gt+=1
    for i in ltl:
        if i:
            lt+=1
    
    eq=len(conds_ops)-gt-lt

    op0=0
    op1=0
    op2=0

    op0l=[op == 0 for op in conds_ops]
    op1l=[op == 1 for op in conds_ops]
    op2l=[op == 2 for op in conds_ops]

    for i in op0l:
        if i:
            op0+=1
    for i in op1l:
        if i:
            op1+=1
    for i in op2l:
        if i:
            op2+=1
    return (int(gt==op1) + int(lt==op2) + int(eq ==op0))/3.0 *100.0
    
    

def same_cond(r,g):
    if r[0] != g[0]:
        return False
    if r[1] != g[1]:
        return False
    if re.sub("[, ]",'',str(r[2]).lower()) != re.sub("[, ]",'',str(g[2]).lower()):
        return False
    return True

def correct(r_sql,g_sql):
    r_agg=r_sql['agg']
    g_agg=g_sql['agg']
    if g_agg != r_agg:
        return False
    r_sel=r_sql['sel']
    g_sel=g_sql['sel']
    if r_agg!=3 and g_sel != r_sel:
        return False
    
    r_conds=r_sql['conds']
    g_conds=g_sql['conds']
    if len(r_conds) != len(g_conds):
        return False
    for r in r_conds:
        count=0
        for g in g_conds:
            if same_cond(r,g):
                count+=1
        if count==0:
            return False    
    return True

if __name__ == '__main__':
    root='/mnt/sda/qhz/sqlova'
    query_path=os.path.join(root,'data_and_model','train_tok_origin.jsonl')
    table_path=os.path.join(root,'data_and_model','train.tables.jsonl')
    p_sqls_path=os.path.join(root,'','rr_p.jsonl')
    queries=extract.read_queries(query_path)
    headers=extract.read_table_headers(table_path)
    p_sqlss=extract.read_potential_sqls(p_sqls_path)
    ranked_sql=[]
    no_psql=0
    for i, p_sqls in enumerate(tqdm(p_sqlss)):
        question=queries[i]['question']
        header=headers[queries[i]['table_id']]
        g_sql=queries[i]['sql']
        flag=False
        for p_sql in p_sqls:
            if(correct(p_sql['query'],g_sql)):
                flag=True
        if not flag:
            p_sqls.append({'query':g_sql})
        for p_sql in p_sqls:
            
            p_sql['c_score']=coverage_score(question,p_sql['query'],header,i)
            p_sql['s_score']=simplicity_score(p_sql['query'])
            p_sql['n_score']=natrual_score(question,p_sql['query'],header)
            p_sql['o_score']=op_score(question,p_sql['query'],i)
        
        p_sqls=sorted(p_sqls,key=lambda x: (x['c_score']+0.005*x['s_score']+0.01*x['o_score']+0.01*x['n_score']) ,reverse=True)
        if p_sqls:
            tops=5 if len(p_sqls) >=5 else len(p_sqls)
            pad=[p_sqls[0]] * (5-tops)
            ranked_sql.append(pad+p_sqls[0:tops])
        else:
            no_psql+=1
            ranked_sql.append([{'query':{'agg': -1, 'conds':[],'sel':0}, 'c_score':0 , 's_score':0 ,'n_score':0 ,'o_score':0}])
    correct_num=0
    correct_count=[0,0,0,0,0]
    with open('score_res_test.txt','w') as f_out:
        for i,r_sql in enumerate(tqdm(ranked_sql)):
            g_sql=queries[i]['sql']
            correct_list=[correct(r_sql_['query'],g_sql) for r_sql_ in r_sql]
            correct_num+=max(correct_list)
            if(max(correct_list)):
                correct_count[correct_list.index(True)]+=1
            for r_sql_i in r_sql: 
                
                if not max([correct(r_sql_['query'],g_sql) for r_sql_ in r_sql]):
                    
                    f_out.write(queries[i]['question'])
                    f_out.write('   ')
                    f_out.write(json.dumps(g_sql))
                    f_out.write('   ')
                    f_out.write(json.dumps(r_sql_i['query']))
                    f_out.write('   ')
                    f_out.write(str(correct(r_sql_i['query'],g_sql)))
                    f_out.write('   ')
                    f_out.write(str(r_sql_i['c_score']))
                    f_out.write('   ')
                    f_out.write(str(r_sql_i['s_score']))
                    f_out.write('   ')
                    f_out.write(str(r_sql_i['o_score']))
                    f_out.write('   ')
                    f_out.write(str(r_sql_i['n_score']))
                    f_out.write('\n')

    print(correct_num)
    print(correct_num/56355)
    print(no_psql)
    print(correct_count)
    with open('./data_and_model/train_tok_sumscore.jsonl','w') as f:
        for i,query in enumerate(tqdm(queries)):
            if ranked_sql[i][0]['query']['agg'] != -1:
                query['sql']=[sql["query"] for sql in ranked_sql[i]]
                f.write(json.dumps(query))
                f.write('\n')
                

        
