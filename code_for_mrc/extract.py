import pandas as pd
import json
from evaluation_script import normalize_answer, f1_score, exact_match_score ,rougel_score
import numpy as np
from IPython import embed

nqfile="../narrativeqa/qaps.csv"
inputfile="out/NarrativeQA9062233-first-only/test_predictions.json"
df=pd.read_csv(nqfile)
test_df=df[df['set']=='test']

test_prd=json.load(open(inputfile,'r'))

n_paragraphs=[10,15,20]
f1s, ems ,rougels = [[] for _ in n_paragraphs], [[] for _ in n_paragraphs], [[] for _ in n_paragraphs]
for j,predictions in enumerate(test_prd.values()):
    groundtruth = [test_df.iloc[j,3],test_df.iloc[j,4]]
    predictions = predictions[:-1]
    if len(groundtruth)==0:
        for i in range(len(n_paragraphs)):
            f1s[i].append(0)
            ems[i].append(0)
            rougels[i].append(0)
        continue
    for i, prediction in enumerate(predictions):
        f1s[i].append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))
        ems[i].append(max([exact_match_score(prediction, gt) for gt in groundtruth]))
        rougels[i].append(max([rougel_score(prediction, gt) for gt in groundtruth]))
for n, f1s_, ems_, rougels_ in zip(n_paragraphs, f1s, ems,rougels):
    print("n=%d\tF1 %.2f\tEM %.2f\tR-L %.2f"%(n, np.mean(f1s_)*100, np.mean(ems_)*100,np.mean(rougels_)*100))          