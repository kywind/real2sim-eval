CUDA_VISIBLE_DEVICES=0 python experiments/eval_policy.py \
gs=rope \
env=xarm_gripper \
physics.ckpt_path=log/phystwin/rope \
physics.case_name=rope_0001 \
policy.inference_cfg_path=policy/configs/inference/insert_rope.json \
policy.checkpoint_path=log/policy_checkpoints/svla-insert-rope/checkpoints/020000
