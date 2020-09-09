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

root='/mnt/sda/qhz/sqlova'
query_path=os.path.join(root,'data_and_model','train_tok_origin.jsonl')
table_path=os.path.join(root,'data_and_model','train.tables.jsonl')
queries=extract.read_queries(query_path)
engine=DBEngine_s('./data_and_model/train.db')
engine1=DBEngine('./data_and_model/train.db')
sql={"sel": 4, "conds": [[5, 0, "Sydney Football Stadium , Sydney (11)"]], "agg": 3}
qg=Query.from_dict(sql, ordered = True)
res=engine.execute_query(queries[716]['table_id'],qg,lower=True)
print(res)