ROOT_DIR=/nas/shared/NLP_A100/feyuan/model/tokenizer_exps/tokenizer_scaling_only_new/baselines_llama2/

MODEL_PATHS=(
${ROOT_DIR}/fully_tuning/ro
${ROOT_DIR}/fully_tuning/el
${ROOT_DIR}/fully_tuning/bn
${ROOT_DIR}/extend_vocab_sft/vocab_100/ro
${ROOT_DIR}/extend_vocab_sft/vocab_800/ro
${ROOT_DIR}/extend_vocab_sft/vocab_1600/ro
${ROOT_DIR}/extend_vocab_sft/vocab_6400/ro
${ROOT_DIR}/extend_vocab_sft/vocab_12800/ro
${ROOT_DIR}/extend_vocab_sft/vocab_25600/ro
${ROOT_DIR}/extend_vocab_sft/vocab_51200/ro
${ROOT_DIR}/extend_vocab_sft/vocab_100/el
${ROOT_DIR}/extend_vocab_sft/vocab_800/el
${ROOT_DIR}/extend_vocab_sft/vocab_1600/el
${ROOT_DIR}/extend_vocab_sft/vocab_6400/el
${ROOT_DIR}/extend_vocab_sft/vocab_12800/el
${ROOT_DIR}/extend_vocab_sft/vocab_25600/el
${ROOT_DIR}/extend_vocab_sft/vocab_51200/el
${ROOT_DIR}/extend_vocab_sft/vocab_100/bn
${ROOT_DIR}/extend_vocab_sft/vocab_800/bn
${ROOT_DIR}/extend_vocab_sft/vocab_1600/bn
${ROOT_DIR}/extend_vocab_sft/vocab_6400/bn
${ROOT_DIR}/extend_vocab_sft/vocab_12800/bn
${ROOT_DIR}/extend_vocab_sft/vocab_25600/bn
${ROOT_DIR}/extend_vocab_sft/vocab_51200/bn
)


for MPATH in "${MODEL_PATHS[@]}"; do
  command="bash /cpfs01/user/yuanfei/code/xllama/analysis/single_script.sh ${MPATH}"
  bash  /cpfs01/user/yuanfei/code/tokenizer_exp/local_submit_jobs/scripts/dlc_submit.sh "${command}" fertility 0
  sleep 1
done


#for MPATH in "${MODEL_PATHS[@]}"; do
#  /cpfs01/user/yuanfei/envs/tokenizer_env/bin/python  /cpfs01/user/yuanfei/code/xllama/analysis/inference_embedding.py  ${MPATH}
#  /cpfs01/user/yuanfei/envs/tokenizer_env/bin/python  /cpfs01/user/yuanfei/code/xllama/analysis/embedding_quality.py  ${MAPTH}
#done