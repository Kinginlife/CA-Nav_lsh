export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export LD_PRELOAD=/home/zju/anaconda3/envs/opennav/lib/libstdc++.so.6.0.29
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

flag=" --exp_name exp_lsh
      --run-type eval
      --exp-config vlnce_baselines/config/exp_lsh.yaml
      --nprocesses 1
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0]
      SIMULATOR_GPU_IDS [0]
      "
CUDA_VISIBLE_DEVICES=0 python run_mp.py $flag