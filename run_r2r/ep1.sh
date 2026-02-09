# export GLOG_minloglevel=2
# export MAGNUM_LOG=quiet
# export LD_PRELOAD=/home/zju/anaconda3/envs/opennav/lib/libstdc++.so.6.0.29
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export PYTHONPATH=$PYTHONPATH:/data1/lsh/CA-Nav/gradslam

# flag=" --exp_name exp_improve_3
#       --run-type eval
#       --exp-config vlnce_baselines/config/exp_lsh.yaml
#       --nprocesses 12
#       NUM_ENVIRONMENTS 1
#       TRAINER_NAME ZS-Evaluator-mp
#       TORCH_GPU_IDS [0,1,2,3,4,5]
#       SIMULATOR_GPU_IDS [0,1,2,3,4,5]
#       "
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python run_mp.py $flag

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export LD_PRELOAD=/home/zju/anaconda3/envs/opennav/lib/libstdc++.so.6.0.29
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:/data1/lsh/CA-Nav/gradslam

flag=" --exp_name exp_improve_2
      --run-type eval
      --exp-config vlnce_baselines/config/exp_lsh.yaml
      --nprocesses 1
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0]
      SIMULATOR_GPU_IDS [0]
      VIDEO_OPTION ['disk']
      VIDEO_DIR 'data/logs/video/'
      "
CUDA_VISIBLE_DEVICES=0 python run_mp.py $flag