Every time, first activate the env:
'''
conda activate legged_gym
cd ~/Desktop/legged_gym/legged_gym
'''

conda activate legged_gym
cd ~/Desktop/legged_gym/legged_gym
python legged_gym/scripts/play.py --task=go2_flat

python legged_gym/scripts/train.py --task=go2_flat --headless --num_envs 4096 --max_iterations 1500
python legged_gym/scripts/play.py --task=go2_flat

