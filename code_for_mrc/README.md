# Score ranking and weighted loss on machine reading comprehension

## Run on NarrativeQA

```
python 3.5 or higher
PyTorch 1.1.0 or higher
```

Download Data and BERT, and unzip them in the current directory.

- [BERT][bert-model-link]: BERT Base Uncased in PyTorch
- Data: The data used is a preprocessed version of NarrativeQA，we use same method mentioned in [qa-hard-em](https://github.com/shmsw25/qa-hard-em), see [link]() for preprocess detail. The original NarrativeQA data is available at [link](https://github.com/deepmind/narrativeqa).

Then, you can do
```
# NQ
./run.sh nq first-only
./run.sh nq mml
./run.sh nq hard-em 8000
./run.sh nq top3
```
to test different method. The 'top3' flag uses our score ranking and weighted losses method.


## Score ranking
- See [score.py](score.py) for detail.
- Coverage: Whether the sentence which contains the answer covers enough exact match tokens in the natural question.
- Relatedness: Use fastText pretrained word vector to calculate the similarity between words in question and answer sentence.
- Simplicity: Reduce the noise causing by the generation method using ROUGE-L score.

## Weighted loss
- Use [task uncertainty](https://arxiv.org/abs/1705.07115v3) to weigh loss of different top3 solution3.

## Details about the model
The model architecture is exactly same as [Min et al 2019][acl-paper]'s model.

Training flags:

- `--train_batch_size`: batch size for training; experiments reported in the paper use batch size of 192
- `--predict_batch_size`: batch size for evaluating
- `--loss_type`: learning method, one of
            (i) `first-only` which only considers the first answer span,
            (ii) `mml` which uses maximum marginal likelihood objective, and
            (iii) `hard-em` which uses hard em objective (our main objective)
- `--tau`: hyperparameters for hard-em objective; only matters when `loss_type` is `hard-em`; experiments reported in the paper use 4000 for TriviaQA-unfiltered and 8000 for NaturalQuestions
- `--init_checkpoint`: model checkpoint to load; for training, it should be BERT checkpoint; for evaluating, it should be trained model
- `--output_dir`: directory to store trained model and predictions
- `--debug`: running experiment with only first 50 examples; useful for making sure the code is running
- `--eval_period`: interval to evaluate the model on the dev data
- `--n_paragraphs`: number of paragraphs per a question for evaluation; you can specify multiple numbers (`"10,20,40,80"`) to see scores on different number of paragraphs
- `--prefix`: prefix when storing predictions during evaluation
- `--verbose`: specify to see progress bar for loading data, training and evaluating


[bert-model-link]: https://drive.google.com/file/d/1XaMX-u5ZkWGH3f0gPrDtrBK1lKDU-QFk/view?usp=sharing
[preprocessed-data-link]: https://drive.google.com/file/d/1FqTr6NzZf0CQ3FmA2dxF9R-2X0--CmBf/view?usp=sharing
[acl-paper]: https://arxiv.org/pdf/1906.02900.pdf



