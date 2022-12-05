#!/bin/bash
#SBATCH -J TSS_3_32
#SBATCH -o MViTv2_Charades_TEST_3_32_1G.out
#SBATCH -e MViTv2_Charades_TEST_3_32_1G.err

#SBATCH --mail-user=3190105604@zju.edu.cn
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr
##SBATCH --qos=sched_level_2
##SBATCH --exclusive
###SBATCH -p sched_system_all

## User python environment
#HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=slowfast
#CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
#source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
#ulimit -s unlimited


## Number of total processes
echo " "
echo "Job name" $SLURM_JOB_NAME
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

export MPORT=4096
export MASTER_PORT=4095


PWD=`pwd`
DATA_ROOT="/gpfs/u/home/NSVR/NSVRbowu/scratch/"
#ENV_VAR="export PYTHONPATH=/nobackup/users/bowu/code/STAR_code/STAR_Action/code/slowfast/slowfast:$PYTHONPATH"
#echo $ENV_V
cd /gpfs/u/home/NSVR/NSVRbowu/scratch/code/STAR_Action/code/slowfast 
  python tools/run_net.py \
  --cfg configs/STAR/MVITv2_B_32x3.yaml \
  --opts NUM_SHARDS 1  NUM_GPUS 1 \
  DATA.PATH_TO_DATA_DIR /gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/STAR_split/ \
  TEST.CHECKPOINT_FILE_PATH ../../exp/MViTv2_finetune_K400_D/checkpoints/checkpoint_epoch_00400.pyth \
  DATA.PATH_PREFIX /gpfs/u/home/NSVR/NSVRbowu/scratch/data/Charades/Charades_v1_rgb/ \
  TEST.SAVE_RESULTS_PATH Charades_test_3_32_sw.json \
  OUTPUT_DIR ../../exp/MViTv2_Charades_TEST_OC \
  MODEL.NUM_CLASSES 157 \
  TRAIN.ENABLE False \
  TEST.BATCH_SIZE 16 \
  DATA.SAMPLING_RATE 3 \
  DATA.NUM_FRAMES 32 \
  TEST.NUM_ENSEMBLE_VIEWS 20 \
  TEST.SLIDING_WINDOWS True \
  TEST.ENABLE True 
echo "Basic commend:"$CMD
echo "Run completed at:- "
date

