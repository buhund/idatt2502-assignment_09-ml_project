# Super Mario Bros PPO

**University project assignment, IDATT2502 Applied Machine Learning, BIDATA (Computer Engineer B.Sc., NTNU) 2024.**

Implementation of the Proximal Policy Optimization (PPO) algorithm for training a machine learning in Super Mario Bros.

The primary goal of the implementation is to compare it to other ML algorithms, first and foremost DDQN, in order to determine which has the better performance in learning to play Super Mario Bros.

All implementations being compared uses the same environment, `SuperMarioBros-1-1-v0` and `SimpleMovements`.



### Installation and requirements



Create a new environment with Python 3.10.15 (Conda):

```bash
conda create -n mario_ppo python=3.10
```



Activate the new conda environment:

```bash
conda activate mario_ppo
```



To install the required packaged, navigate to the directory `mario_ppo` 

`cwd` (current working directory) `mario_ppo`:

```bash
§ --cwd--: mario_ppo (current working directory)

pip install -r requirements.txt
```



#### TensorBoard:

To enable logging via TensorBoard, run the following from the `mario_ppo` dirctory:

Training logs

```bash
tensorboard --logdir=logs/tensorboard/mario_ppo_training/
```

Testing logs

```bash
tensorboard --logdir=logs/tensorboard/mario_ppo_testing/
```



### Running the agent

The agent have two modes: `train` and `test`.

Training mode will train the agent on the environment. This will write the policy updates to `agent.pth` and `critic.pth` files, which hold the latest best policy weights for navigating the environment.

Testing mode will load the `actor.pth` and `critic.pth`, and then run the agent in the environment however many times is set in the config file. This mode will not learn anything, nor update the `actor.pth` or `critic.pth` files. This is to test(!) how well the trained model performs after the training. The agent's performance will be logged to `logs/tensorboard/mario_ppo_testing/` and can be viewied in TensorBoard.

#### Run in terminal

The scripts run via terminal arguments. By invoking `python` on `main.py` and setting the `--mode` flagwith `train` or `test` arguments.



**Training:**

```bash
§ --cwd--: mario_ppo (current working directory)

python main.py --mode train
```



**Testing:**

```bash
§ --cwd--: mario_ppo (current working directory)

python main.py --mode test
```



It can also be run in an IDE, e.g. PyCharm, by loading the .xml run config files in `.run/`

```bash
/mario_ppo/.run/
├── mario_ppo_test.run.xml
└── mario_ppo_train.run.xml
```





#### Config

In `config.py`, you have to specify the parameters for the agent, especially the number of episodes for training and testing, as well as checkpoint interval.

```python
# Environment config
WORLD = 1
STAGE = 1
ENV_VERSION = "v0"
ENV_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-{ENV_VERSION}"
RENDER_MODE = True # True for visual gameplay, False for no visuals.

# Number of episodes to run
NUM_EPISODES = 30_000 # Episodes during training
NUM_TEST_EPISODES = 1_000 # Episodes during testing
# Checkpoint settings
CHECKPOINT_INTERVAL = 1000 # Checkpoint every X episodes
```

##### Rendering

To toggle displaying rendering, i.e. getting the window showing the agent playing Super Mario, you change the value `RENDER_MODE`:

- `True` to enable rendering
- `False` to disable rendering.



### OSError: 'GLIBCXX_3.4.32' not found

WARNING: You may very well break your user environment in some way by following the instructions under. Proceed at your own risk. You should probably not try this at home. Do it on your work laptop, where it doesn't matter if it breaks. This is also probably where Håkon would say to use Docker.



If you get the following error:

```bash
OSError: /lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/USER/miniconda3/envs/mario_ppo/lib/python3.10/site-packages/nes_py/lib_nes_env.cpython-310-x86_64-linux-gnu.so)
```



It can be fixed by installing the following:

```bash
sudo apt install gcc-11 g++-11
```



Then check if `GLIBCXX_3.4.32` is installed:
```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

This will print a list where you need to locate `3.4.32`. If it's there, you've got it.



You also likely will have to symlink this into your active venv:

```bash
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6 
```



I assume this would work for a non-conda venv also, but I have no idea.



## License

[GNU GENERAL PUBLIC LICENSE Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html)
