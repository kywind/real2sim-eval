CUDA_VISIBLE_DEVICES=0 python experiments/eval_policy.py \
gs=sloth \
env=xarm_gripper \
env.sim.duration=15 \
physics.ckpt_path=log/phystwin/sloth \
physics.case_name=sloth_0001 \
policy.inference_cfg_path=policy/configs/inference/pack_sloth.json \
policy.checkpoint_path=log/policy_checkpoints/svla-pack-sloth/checkpoints/020000
