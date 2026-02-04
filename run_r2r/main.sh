export GLOG_minloglevel=2 #降低 Google glog 的输出量（只输出 warning/error 级别附近）
export MAGNUM_LOG=quiet  #HabitatSim / Magnum 的日志更安静（少打渲染/引擎日志）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

flag=" --exp_name exp_1
      --run-type eval
      --exp-config vlnce_baselines/config/exp1.yaml
      --nprocesses 16@
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0,1,2,3,4,5,6,7]
      SIMULATOR_GPU_IDS [0,1,2,3,4,5,6,7]
      "
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_mp.py $flag

#用 8 张 GPU，开 16 个 Python 进程做评测，每个进程绑定到某一张 GPU（按 i % num_gpus 轮转）