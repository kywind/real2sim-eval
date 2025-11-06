CUDA_VISIBLE_DEVICES=0 python experiments/replay.py \
gs=rope \
env=xarm_gripper \
physics.ckpt_path=log/phystwin/rope \
physics.case_name=rope_0001 \
gt_dir=log/policy_rollouts/rope_act_7000
