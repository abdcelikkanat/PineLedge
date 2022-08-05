### General options
### -- specify queue --
#BSUB -q compute
### -- set the job Name --
#BSUB -J CompPred
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 8GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u abdcelikkanat+dtuexperiment@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o CompPred_%J.out
#BSUB -e CompPred_%J.err

BASEFOLDER="/Users/abdulkadir/workspace/noname"
PYTHON="/opt/anaconda3/envs/pivem/bin/python"
PREDICTION_SCRIPT="${BASEFOLDER}/experiments/completion/generate_samples.py"
PREDICTION_SCRIPT="${BASEFOLDER}/experiments/prediction/generate_samples.py"
SCRIPT="${BASEFOLDER}/experiments/construct_samples.py"
export PYTHONPATH="${PYTHONPATH}:/${BASEFOLDER}"

#DATASETS=( ia-contact )
DATASETS=( ia-contacts_hypertext2009 )
#DATASETS=( fb-forum )
#DATASETS=( soc-sign-bitcoinalpha_new )
#DATASETS=( soc-wiki-elec_new )
#DATASETS=( ia-contact ia-contacts_hypertext2009 fb-forum soc-sign-bitcoinalpha_new soc-wiki-elec_new )
#DATASETS=( 25_clusters_beta=0.25 )

R=0.001
THREADS=16
RADIUS=0.001
SAMPLESET_LIST=( valid test )
TRAINRATIO=0.6
SPLITRATIO=0.95


#LAMBDA_LIST=( 1e0 1e1 1e2 1e3 1e4 1e5 )

#BIN=100
#K=10
#DIM=2
#EPOCH=500
#SPE=10
#BATCH_SIZE=100
#LR=0.1
#SEED=1

for DATASET in ${DATASETS[@]}
do

INPUT=${BASEFOLDER}/datasets/real/${DATASET}/
PRED_OUTPUT=${BASEFOLDER}/experiments/samples/${DATASET}/seed-19_r-${R}_predTrainRatio-${PRED_TRAINRATIO}


#${PYTHON} ${COMPLETION_SCRIPT} --dataset_folder ${INPUT} --output_folder ${PRED_OUTPUT} --radius ${RADIUS} --train_ratio ${PRED_TRAINRATIO} --threads ${THREADS} --seed 19
#${PYTHON} ${PREDICTION_SCRIPT} --dataset_folder ${INPUT} --output_folder ${PRED_OUTPUT} --radius ${RADIUS} --train_ratio ${PRED_TRAINRATIO} --threads ${THREADS} --seed 19
${PYTHON} ${SCRIPT} --dataset_folder ${INPUT} --output_folder ${PRED_OUTPUT} --radius ${RADIUS} --split_ratio ${SPLITRATIO} --train_ratio ${TRAINRATIO} --threads ${THREADS} --seed 19



#${PYTHON} ${PREDICTION_SCRIPT} --dataset_folder ${INPUT} --output_folder ${PRED_OUTPUT} --radius ${RADIUS} --train_ratio ${PRED_TRAINRATIO} --threads ${THREADS} --seed 19

#SAMPLES=${BASEFOLDER}/noname/experiments/completion/samples/${DATASET}/seed-19_trainRatio-${TRAINRATIO}_r-${R}
#MODELPATH=${BASEFOLDER}/noname/experiments/models/completion_${DATASET}_B=${BIN}_K=${K}_lambda=${LAMBDA}_dim=${DIM}_epoch=${EPOCH}_spe=${SPE}_bs=${BATCH_SIZE}_lr=${LR}_seed=${SEED}.model
#OUTPUT=${BASEFOLDER}/noname/experiments/completion/${DATASET}_trainRatio=${TRAINRATIO}_${SAMPLESET}_lambda=${LAMBDA}.result
#
#${PYTHON} ${SCRIPT} --samples_folder ${SAMPLES} --samples_set ${SAMPLESET} --model_path ${MODELPATH} --output_path ${OUTPUT}

done

