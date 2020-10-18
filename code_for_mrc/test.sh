python3.6 main.py --do_predict --output_dir "out/NQA-FULL-top5" \
          --predict_file "NarrativeQA/test.jsonl" \
          --init_checkpoint "out/NQA-FULL-top5/best-model.pt" \
          --predict_batch_size 100 --n_paragraphs "10,15,20" --prefix test_