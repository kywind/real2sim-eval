CUDA_VISIBLE_DEVICES=0 python experiments/eval_policy.py \
gs=T \
env=xarm_pusher \
physics.ckpt_path=log/phystwin/T \
physics.case_name=T_0001 \
policy.inference_cfg_path=policy/configs/inference/pusht.json \
policy.checkpoint_path=log/policy_checkpoints/dp-pusht/checkpoints/007000
