
BASEFOLDER="/Users/abdulkadir/workspace/noname"
PYTHON="/opt/anaconda3/envs/pivem/bin/python"
SCRIPT="${BASEFOLDER}/run.py"
export PYTHONPATH="${PYTHONPATH}:/${BASEFOLDER}"

DATASETS=( ia-contacts_hypertext2009 )
#DATASETS=( ia-contact )
#DATASETS=( fb-forum )
#DATASETS=( soc-sign-bitcoinalpha_new )
#DATASETS=( soc-wiki-elec_new )
#DATASETS=( ia-contacts_hypertext2009 ia-contact fb-forum soc-sign-bitcoinalpha_new soc-wiki-elec_new )

BIN=100
K=10
LAMBDA_LIST=( 1e0 1e1 1e2 1e3 1e3 1e5 1e6 )
DIM=2
EPOCH=500
SPE=10
BATCH_SIZE=100
LR=0.1
SEED=0

for DATASET in ${DATASETS[@]}
do

LEN=${#LAMBDA_LIST[@]}
for (( IDX=0; IDX<${LEN}; IDX++ ));
do

echo ${DATASET} ${LAMBDA_LIST[${IDX}]}
MODELNAME=inc_${DATASET}_B=${BIN}_K=${K}_lambda=${LAMBDA}_dim=${DIM}_epoch=${EPOCH}_spe=${SPE}_bs=${BATCH_SIZE}_lr=${LR}_seed=${SEED}
INPUT=${BASEFOLDER}/noname/experiments/samples/seed-19_r-0.001_trainRatio-0.95_splitRatio-0.9/${DATASET}/
OUTPUT=${BASEFOLDER}/noname/experiments/models/${MODELNAME}.model
LOG=${BASEFOLDER}/noname/experiments/logs/${MODELNAME}.txt

${PYTHON} ${SCRIPT} --dataset ${INPUT} --model_path ${OUTPUT} --log ${LOG} --dim ${DIM} --bins_num ${BIN} --k ${K} --prior_lambda ${LAMBDA} --epoch_num ${EPOCH} --lr ${LR} --seed ${SEED} --spe ${SPE} --batch_size ${BATCH_SIZE} --verbose 1

done
done


