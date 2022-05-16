python train.py --exp_name supMT_eten --dump_path $model_save_dir --reload_mt_and_enc_emb "$mt_model_path,$bert_model_path" --data_path $data_url --lgs 'et-en' --mt_steps 'et-en' --encoder_only false --emb_dim 512 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 4096 --bptt 256 --optimizer adam_inverse_sqrt,lr=0.0001,beta1=0.9,beta2=0.999,eps=0.00000001 --batch_size 1 --epoch_size 1000 --eval_bleu true --beam_size 5 --stopping_criterion 'valid_et-en_mt_bleu,12' --validation_metrics 'valid_et-en_mt_bleu' --use_lang_emb false --amp 1 --fp16 true --max_len 128