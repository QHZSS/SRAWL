from tqdm import tqdm
import json
from IPython import embed
import tokenization
from itertools import groupby
import gensim

def read_train_dataset(file_path):
    tokenizer=tokenization.FullTokenizer(vocab_file="uncased_L-12_H-768_A-12/vocab.txt", do_lower_case=True)
    examples=[]
    with open(file_path,'r') as f:
        lines=f.readlines()
        for line in tqdm(lines):
            example=json.loads(line)
            example['question_split']=example['question'].split()
            example['question_tok']=tokenizer.tokenize(example['question'])
            example['context_sentences']=[]
            context_sentences_len=[]
            for i,context in enumerate(example['context']):
                sentences=[list(g) for k, g in groupby(context, lambda x:x=='.' or x=='?' or x==';') if not k]
                example['context_sentences'].append(sentences)
                sentence_len=[len(sent)+1 for i,sent in enumerate(sentences)]
                for i,val in enumerate(sentence_len):
                    if i>0:
                        sentence_len[i]+=sentence_len[i-1]
                context_sentences_len.append(sentence_len)
            for i,answers in enumerate(example['answers']):
                if len(answers)!=0:
                    for j,answer in enumerate(answers):
                        example['answers'][i][j]['text_tok']=example['context'][i][answer['word_start']:answer['word_end']+1]
                        start=answer['word_start']
                        for k,length in enumerate(context_sentences_len[i]):
                            if start<length:
                                example['answers'][i][j]['answer_sentence']=example['context_sentences'][i][k]
                                break
                        if 'answer_sentence' not in example['answers'][i][j]:
                            embed()
            examples.append(example)
    return examples

def coverage_score(question,sentence):

    score=0
    already=[] #words which has already been counted
    for word in question:
        if word !='?' and word in sentence and word not in already:
            score+=1
            already.append(word)
    length=len(question)
    return score/length*100

def simplicity_score(answer):
    return 100-len(answer)

def natural_score(question,sentence,wv):
    score=0
    # remove words that has counted in coverage score
    already=[] #words which has already been counted
    article=['the','a','an']
    aux_verb=['do','does','done','did','is','are','were','was']

    for word in question:
        if word !='?' and word in sentence and word not in already:
            already.append(word)

    for word in question:
        if word !='?' and word not in already and word not in article and word not in aux_verb:
            similarities=[wv.similarity(word,tok) for tok in sentence]
            if any([score>=0.4 for score in similarities]):
                score+=1
    length=len(question)
    return score/length*100


if __name__=="__main__":
    print("loading fasttext pre-trained vectors")
    wv=gensim.models.fasttext.load_facebook_vectors('/mnt/sdb/qhz/qa-hard-em/fasttext/cc.en.300.bin')
    print("loading finished")
    for k in range(0,4):
        examples=read_train_dataset(f"NarrativeQA/train{k}.jsonl")
        with open(f'NarrativeQA/train{k}_n+s.jsonl','w') as f:
            for example in tqdm(examples):
                for i,answers in enumerate(example['answers']): #i for paragraph index
                    if len(answers) !=0: #ignore empty answers paragraph
                        for j,answer in enumerate(answers):
                            example['answers'][i][j]['c_score']=coverage_score(example['question_split'],answer['answer_sentence'])
                            example['answers'][i][j]['s_score']=simplicity_score(answer['text_tok'])
                            example['answers'][i][j]['n_score']=natural_score(example['question_split'],answer['answer_sentence'],wv)
                        example['answers'][i]=sorted(example['answers'][i],key=lambda x:(0.5*x['n_score']+0.1*x['s_score']),reverse=True)
                f.write(json.dumps(example))
                f.write('\n')
                
    