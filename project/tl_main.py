# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.core.experiment import Experiment as experiment
from tl_env import Experiment,flow_params
import logging

import datetime
import numpy as np
import time
import json
import sys 
import os
import tensorflow as tf
from time import strftime
from from_jiawei.ring_RL import DDPGAgent

import ray
from ray import tune
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from copy import deepcopy

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params

# RLlib algorithms
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
# from ray.rllib.agents.dqn.dqn import DQNTFPolicy # Not suitable for continuous space
from ray.rllib.agents.a3c.a3c import A3CTFPolicy
from ray.rllib.agents.pg.pg import PGTFPolicy
from ray.rllib.agents.ddpg.ddpg import DDPGTFPolicy
from ray.rllib.agents.marwil.marwil import MARWILPolicy

#import pandas as pd
f = open('log/reward_singelagent_minst4_100run_3000steps.txt','w')
exp = Experiment()
env = exp.env

print(env.initial_state)

num_runs=100
rl_actions=None
convert_to_csv=True

#num_steps = env.env_params.horizon

# raise an error if convert_to_csv is set to True but no emission
# file will be generated, to avoid getting an error at the end of the
# simulation
if convert_to_csv and env.sim_params.emission_path is None:
	raise ValueError(
		'The experiment was run with convert_to_csv set '
		'to True, but no emission file will be generated. If you wish '
		'to generate an emission file, you should set the parameter '
		'emission_path in the simulation parameters (SumoParams or '
		'AimsunParams) to the path of the folder where emissions '
		'output should be generated. If you do not wish to generate '
		'emissions, set the convert_to_csv parameter to False.')

info_dict = {}
if rl_actions is None:
	def rl_actions(*_):
		return None

state_dim = 162 #env.observation_space.shape[0]
action_dim = 1

print('State action space: ',state_dim,action_dim)

# Start agent definition
n_cpus = 4
n_rollouts = 100 # num_steps

gym_name = exp.env_name
obs_space = exp.obs_space
act_space = exp.act_space 

print('Obs and Act spaces: ',obs_space,act_space)

def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    horizon = flow_params['env'].horizon
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["num_workers"] = n_cpus
    config["train_batch_size"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = horizon
    config['log_level']="DEBUG"

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        print("policy_graphs", policy_graphs)
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update({'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})
    return alg_run, config

alg_run,config = setup_exps_rllib(flow_params,n_cpus,n_rollouts)

ray.init(num_cpus=n_cpus + 1)
trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 20,
        "checkpoint_at_end": True,
        "max_failures": 1,
        "stop": {
            "training_iteration": num_runs,
        },
    }
})
print(trials)


#return info_dict