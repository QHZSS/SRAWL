# SRAWL

### Source code
#### Requirements
- `python3.6` or higher.
- `PyTorch 0.4.0` or higher.
- `CUDA 9.0`
- Python libraries: `babel, matplotlib, defusedxml, tqdm`
- Example
    - Install [minicoda](https://conda.io/miniconda.html)
    - `conda install pytorch torchvision -c pytorch`
    - `conda install -c conda-forge records==0.5.2`
    - `conda install babel` 
    - `conda install matplotlib`
    - `conda install defusedxml`
    - `conda install tqdm`
- The code has been tested on GeForce RTX2080Ti running on Ubuntu 18.04.4 LTS.

Our method is consist of two parts: Score ranking and weighted loss update.

#### Generate potential logical forms for QA pairs
- Generate all possible logical forms according to the natural language questions and answers.
- Based on the fact that the logical forms of WIKISQL dataset are fixed:
    - SELECT _ FROM _ WHERE () AND () AND ()
- See wikisql_data_generation.py for details

#### Score ranking
- See score.py for details
- Calculate coverage score, natural score, simplicity score for each candidate logical form for a QA pair.
  - Coverage score: The logical form with more words mentioned in question and table headers will have a higher score.
  - Natural score: If the relationship between specific words and aggregating operators/where condition operators is closer (e.g. "total number" in natrual question and "SUM" in aggregating operators;"more than" in natural question and ">" in where condition operator), the natural score is higher.
  - Simplicity score: The less Where Conditions the logical form has, the higher the score.
  - All candidate logical forms for a QA pair are sorted by multiple scores according to thier order above.
  - Select top5 logical forms for training.

#### Weighted loss
- Use top5 logical forms for training, calculate losses seprately and update losses with weighted sum of the 5 losses. 
- See [sqlova/model/nl2sql/wikisql_models.py#L907-L946](https://github.com/QHZSS/SRAWL/blob/master/sqlova/model/nl2sql/wikisql_models.py#L907-L936) for details


#### Data
- Data needed for training will be upload later

#### Running code
- Type `python3 train.py --seed 1 --bS 4 --accumulate_gradients 8 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 222` on terminal.
    - `--seed 1`: Set the seed of random generator. The accuracies changes by few percent depending on `seed`.
    - `--bS 16`: Set the batch size by 16.
    - `--accumulate_gradients 2`: Make the effective batch size be `16 * 2 = 32`.
    - `--bert_type_abb uS`: Uncased-Base BERT model is used. Use `uL` to use Uncased-Large BERT.
    - `--fine_tune`: Train BERT. Without this, only the sequence-to-SQL module is trained.
    - `--lr 0.001`: Set the learning rate of the sequence-to-SQL module as 0.001. 
    - `--lr_bert 0.00001`: Set the learning rate of BERT module as 0.00001.
    - `--max_seq_leng 222`: Set the maximum number of input token lengths of BERT.     
- Add `--EG` argument while running `train.py` to use execution guided decoding. 
- Whenever higher logical form accuracy calculated on the dev set, following three files are saved on current folder:
    - `model_best.pt`: the checkpoint of the the sequence-to-SQL module.
    - `model_bert_best.pt`: the checkpoint of the BERT module.
    - `results_dev.jsonl`: json file for official evaluation.
- `Shallow-Layer` and `Decoder-Layer` models can be trained similarly (`train_shallow_layer.py`, `train_decoder_layer.py`). 

#### Evaluation on WikiSQL DEV set
- To calculate logical form and execution accuracies on `dev` set using official evaluation script,
    - Download original [WikiSQL dataset](https://github.com/salesforce/WikiSQL).
    - tar xvf data.tar.bz2
    - Move them under `$HOME/data/WikiSQL-1.1/data`
    - Set path on `evaluation_ws.py`. This is the file where the path information has added on original `evaluation.py` script. Or you can use original [`evaluation.py`](https://github.com/salesforce/WikiSQL) by setting the path to the files by yourself.
    - Type `python3 evaluation_ws.py` on terminal.

#### Evaluation on WikiSQL TEST set
- Use --do_test flag
- Save the output of `test(...)` with `save_for_evaluation(...)` function.
- Evaluate with `evaluatoin_ws.py` as before.


#### Code base 
- Pretrained BERT models were downloaded from [official repository](https://github.com/google-research/bert). 
- BERT code is from [huggingface-pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
- The sequence-to-SQL model is started from the source code of [SQLNet](https://github.com/xiaojunxu/SQLNet) and significantly re-written while maintaining the basic column-attention and sequence-to-set structure of the SQLNet.