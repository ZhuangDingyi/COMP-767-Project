from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams

from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

# the TestEnv environment is used to simply simulate the network
from flow.envs import TestEnv
# the Experiment class is used for running simulations
from flow.core.experiment import Experiment
# the base network class
from flow.networks import Network
# all other imports are standard
from flow.core.params import VehicleParams, NetParams, InitialConfig, EnvParams, \
    SumoParams, SumoCarFollowingParams, InFlows, SumoLaneChangeParams, TrafficLightParams
#from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from tl_custom_env import TrafficLightGridEnv,TrafficLightGridTestEnv,TrafficLightGridPOEnv,ADDITIONAL_ENV_PARAMS,ADDITIONAL_PO_ENV_PARAMS

from flow.controllers import SimLaneChangeController, GridRouter

# define parameters
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from copy import deepcopy
from gym.spaces.box import Box
from flow.core import rewards
from flow.envs.base import Env
from tl_net import offRampGrid,initial_config,net_params

import logging
import datetime
import numpy as np
import time
import os
from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env

# We firstly test on a single agent scenario
def para_produce_rl(HORIZON=3000):
    # Create default environment parameters
    env_params = EnvParams()

    # Vehicle definition
    vehicles = VehicleParams()
    num_vehicles = 1
    vehicles.add(
        veh_id="human",
        routing_controller=(GridRouter, {}),
        lane_change_controller=(SimLaneChangeController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
        ),
        lane_change_params=SumoLaneChangeParams(
                lane_change_mode=1621,
            ),
        num_vehicles=num_vehicles)

    # whether to allow turns at intersections
    ALLOW_TURNS = False
    '''
    # initialize traffic lights, used when you want define your own traffic lights
    tl_logic = TrafficLightParams(baseline=False) # To see static traffic lights in action, the `TrafficLightParams` object should be instantiated with `baseline=False`

    # when use off_ramp_grid.net.xml file, you should use a phase state example as "GGGgrrrrGGGgrrrr"
    # when use off_ramp_grid_turn.net.xml file, you should use a phase state example as "GGGggrrrrrGGGggrrrrr"
    if ALLOW_TURNS:
        phases = [{
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            # for actuated traffic lights, you can add these optional values below
            # "maxGap": int, describes the maximum time gap between successive vehicle sthat will cause the current phase to be prolonged
            # "detectorGap": int, determines the time distance between the (automatically generated) detector and the stop line in seconds
            # "showDetectors": bool, toggles whether or not detectors are shown in sumo-gui
            "state": "GGGggrrrrrGGGggrrrrr"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "yyyyyrrrrryyyyyrrrrr"
        }, {
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            "state": "rrrrrGGGggrrrrrGGGgg"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "rrrrryyyyyrrrrryyyyy"
        }]
        tl_logic.add("center0", phases=phases, programID=1, detectorGap=1,tls_type="actuated")
        tl_logic.add("center1", phases=phases, programID=1, detectorGap=1,tls_type="actuated")
        tl_logic.add("center2", phases=phases, programID=1, detectorGap=1,tls_type="actuated")
        tl_logic.add("center3", phases=phases, programID=1, detectorGap=1,tls_type="actuated")
    else:
        phases = [{
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            # for actuated traffic lights, you can add these optional values below
            # "maxGap": int, describes the maximum time gap between successive vehicle sthat will cause the current phase to be prolonged
            # "detectorGap": int, determines the time distance between the (automatically generated) detector and the stop line in seconds
            # "showDetectors": bool, toggles whether or not detectors are shown in sumo-gui
            "state": "GGGgrrrrGGGgrrrr"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "yyyyrrrryyyyrrrr"
        }, {
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            "state": "rrrrGGGgrrrrGGGg"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "rrrryyyyrrrryyyy"
        }]

        # THIS IS A BUG THAT I DON'T KNOW WHY IT HAPPENS!!!!!!
        phase0 = [{
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            "state": "GGrrGGrrGGrrGGrr"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "yyrryyrryyrryyrr"
        }, {
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            "state": "rrGGrrGGrrGGrrGG"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "rryyrryyrryyrryy"
        }]

        tl_logic.add("center0", phases=phases, programID=1, detectorGap=1,tls_type="actuated")
        tl_logic.add("center1", phases=phases, programID=1, detectorGap=1,tls_type="actuated")
        tl_logic.add("center2", phases=phases, programID=1, detectorGap=1,tls_type="actuated")
        tl_logic.add("center3", phases=phases, programID=1, detectorGap=1,tls_type="actuated")
    '''
    flow_params = dict(
    exp_tag='A traffic light grid with exit ramp from highway',
    env_name=TrafficLightGridPOEnv,
    network=offRampGrid,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        emission_path='./data',
        restart_instance=True,
    ),
    env=EnvParams(
        horizon=3000,    additional_params=ADDITIONAL_PO_ENV_PARAMS.copy(),
    ),
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    # used when you define your own traffic lights
    #tls=tl_logic,
    )
    flow_params['env'].horizon = HORIZON
    return flow_params

flow_params = para_produce_rl()

class Experiment:
    def __init__(self, flow_params=flow_params):
        """Instantiate Experiment."""
        # Get the env name and a creator for the environment.
        self.create_env, self.env_name = make_create_env(flow_params)

        # Create the environment.
        self.env = self.create_env()
        self.flow_params = flow_params

        # Register as rllib env
        register_env(self.env_name,self.create_env)

        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

