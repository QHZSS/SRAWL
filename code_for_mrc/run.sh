data=$1
loss=$2
bs=20
pbs=100
data_dir="NarrativeQA"

if [ ${loss} = "hard-em" ]
then
    tau=$3
    output_dir="out/${data}-${loss}-${tau}"
else
    output_dir="out/${data}-${loss}"
    tau=0
fi

train_file="${data_dir}/train0.jsonl"
for index in 1 2 3 ; do
    train_file="${train_file},${data_dir}/train${index}.jsonl"
done
dev_file="${data_dir}/valid.jsonl"
test_file="${data_dir}/test.jsonl"

python3.6 main.py --do_train --output_dir ${output_dir} \
          --train_file ${train_file} --predict_file ${dev_file} \
          --train_batch_size ${bs} --predict_batch_size ${pbs} --verbose --loss_type ${loss} --tau ${tau}
python3.6 main.py --do_predict --output_dir ${output_dir} \
          --predict_file ${dev_file} \
          --init_checkpoint ${output_dir}/best-model.pt \
          --predict_batch_size ${pbs} --n_paragraphs "10,15,20" --prefix dev_
python3.6 main.py --do_predict --output_dir ${output_dir} \
          --predict_file ${test_file} \
          --init_checkpoint ${output_dir}/best-model.pt \
          --predict_batch_size ${pbs} --n_paragraphs "10,15,20" --prefix test_