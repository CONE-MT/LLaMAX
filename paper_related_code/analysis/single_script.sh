MPATH=$1

/cpfs01/user/yuanfei/envs/tokenizer_env/bin/python  /cpfs01/user/yuanfei/code/xllama/analysis/inference_embedding.py  ${MPATH}
/cpfs01/user/yuanfei/envs/tokenizer_env/bin/python  /cpfs01/user/yuanfei/code/xllama/analysis/fertility.py ${MPATH}
/cpfs01/user/yuanfei/envs/tokenizer_env/bin/python  /cpfs01/user/yuanfei/code/xllama/analysis/embedding_quality.py  ${MPATH}