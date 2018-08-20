# A2C-Pytorch

Minimal implementation of A2C, running in Mujoco env, using Gym-mujoco. It is based on the code [openai/baselines](https://github.com/openai/baselines).  

Now it is a Pytorch version. The training process can run. But when I train a agent on Ant-v2 task, the learning cruve is not stable. I had checked this repo for a long time, but I did not figure it out. I tried the origin code of [baselines](https://github.com/openai/baselines/tree/master/baselines/a2c) on this task, that also lead to a poor result. And I tried someother repos, but could not get a better result. Maybe some bugs are in it maybe there are someother reasons, I am not sure.  

# Requirement

Python3+  
Pytorch 0.4  
Mujoco  
Gym, Mujoco_py  

# Train

Using following command to train a model, more args can be set in 'main.py'.

    git clone git@github.com:/qqadssp/A2C-Pytorch
    cd A2C-Pytorch
    python3 main.py --env Ant-v2
