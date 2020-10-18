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

root='/mnt/sda/qhz/sqlova'
query_path=os.path.join(root,'data_and_model','train_tok_origin.jsonl')
table_path=os.path.join(root,'data_and_model','train.tables.jsonl')
p_sqls_path=os.path.join(root,'','rr_p.jsonl')
p_sqlss=extract.read_potential_sqls(p_sqls_path)
ranked_sql=[]
no_psql=0
leng=[]
for i, p_sqls in enumerate(tqdm(p_sqlss)):
    leng.append(len(p_sqls))

print(np.average(leng),np.median(leng))