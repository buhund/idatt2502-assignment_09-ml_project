# Super Mario Bros PPO

**University project assignment, IDATT2502 Applied Machine Learning, BIDATA (Computer Engineer B.Sc., NTNU) 2024.**

Implementation of the Proximal Policy Optimization (PPO) algorithm for training a machine learning in Super Mario Bros.

The primary goal of the implementation is to compare it to other ML algorithms, specifically DQN and DDQN in order to determine which has the better performance in learning to play Super Mario Bros.

All implementations being compared uses the same environment, `SuperMarioBros-1-1-v0` and `SimpleMovements`.



### Installation and requirements



Create a new environment with Python 3.10.15 (Conda):

```
conda create -n mario_ppo python=3.10
```



Activate the new conda environment:

```
conda activate mario_ppo
```



To install the required packaged:

```
pip install -r requirements.txt
```



#### TensorBoard:

To enable logging via TensorBoard, run the following from the `mario_ppo` dirctory:

```
tensorboard --logdir=logs/ppo
```







### OSError: 'GLIBCXX_3.4.32' not found

If you get the following error:

```
OSError: /lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/USER/miniconda3/envs/mario_ppo/lib/python3.10/site-packages/nes_py/lib_nes_env.cpython-310-x86_64-linux-gnu.so)
```



It can be fixed by installing the following:

```
sudo apt install gcc-11 g++-11
```



Then check if `GLIBCXX_3.4.32` is installed:
```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

This will print a list where you need to locate `3.4.32`. If it's there, you've got it.



You also likely will have to symlink this into your active venv:

```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6 
```



I assume this would work for a non-conda venv also, but I have no idea.