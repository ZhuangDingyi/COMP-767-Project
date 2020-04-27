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
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers import SimLaneChangeController, GridRouter

import sys
import os

# inflow configuration
inflow = InFlows()
# off ramp inflow

inflow.add(
    veh_type='human',
    edge='ramp',
    probability=0.99,
    depart_lane='free',
    depart_speed=20)



# other inflow edges
outer_edges = ['bot1_0', 'bot0_0', 'top1_2', 'top0_3', 'left2_0', 'left2_1', 'right0_0', 'right0_1']
for i in range(len(outer_edges)):
    inflow.add(
        veh_type='human',
        edge=outer_edges[i],
        probability=0.25,
        depart_lane='free',
        depart_speed=20)

# Network parameters
net_params = NetParams(
    inflows=inflow,
    # used when routes only allow straight forward in intersections
    # template=os.path.abspath('/home/dingyizhuang/notebookWorkplace/flow/examples/exp_configs/rl/multiagent/network/off_ramp_grid_straight.net.xml'),
    # used for all situations
    template=os.path.abspath('network/off_ramp_grid_turn.net.xml'),
    additional_params={
        "speed_limit": {
        "horizontal": 35,
        "vertical": 35
        },  # inherited from grid0 benchmark
        "grid_array": {
            "short_length": 300,
            "inner_length": 300,
            "long_length": 300,
            "row_num": 2,
            "col_num": 2,
            "cars_left": 5,
            "cars_right": 5,
            "cars_top": 5,
            "cars_bot": 5,
        },
        "horizontal_lanes": 1,
        "vertical_lanes": 1,
    },
)

# specify the edges vehicles can originate on
initial_config = InitialConfig(
    edges_distribution=['bot1_0', 'bot0_0', 'top1_2', 'top0_3', 'left2_0', 'left2_1', 'right0_0', 'right0_1', 'ramp']
)

# specify the routes for vehicles in the network
class offRampGrid(TrafficLightGridNetwork):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=initial_config,
                 traffic_lights=TrafficLightParams()):
        super(offRampGrid,self).__init__(name,vehicles,net_params,initial_config,traffic_lights)
    @property
    def node_mapping(self):
        """Map nodes to edges.

        Returns a list of pairs (node, connected edges) of all inner nodes
        and for each of them, the 4 edges that leave this node.

        The nodes are listed in alphabetical order, and within that, edges are
        listed in order: [bot, right, top, left].
        """
        mapping = {}
        mapping['center0'] = ['bot0_0','right0_0','top0_1','left1_0']
        mapping['center1'] = ['bot0_2','right0_1','top0_3','left1_1']
        mapping['center2'] = ['bot1_0','right1_0','top1_1','left2_0']
        mapping['center3'] = ['bot1_1','right1_1','top1_2','left2_1']
        #mapping['center4'] = ['bot0_1','ramp','top0_2']

        return sorted(mapping.items(), key=lambda x: x[0])
    
    def specify_routes(self, net_params):
        """See parent class."""
        ALLOW_TURNS = False
        if ALLOW_TURNS:    
            # allow all turns at intersections, but delete some routes
            return {'bot0_0': [
                        (['bot0_0', 'bot0_1', 'bot0_2', 'bot0_3'], 0.5), # wait for rewrite turns
                        (['bot0_0', 'left0_0'], 0.25),
                        (['bot0_0', 'right1_0', 'top1_0'], 0.1),
                        (['bot0_0', 'right1_0', 'right2_0'], 0.15),],
                    'top0_3': [
                        (['top0_3', 'top0_2', 'top0_1', 'top0_0'], 0.5), # wait for rewrite turns
                        (['top0_3', 'left0_1'], 0.25),
                        (['top0_3', 'right1_1', 'bot1_2'], 0.1),
                        (['top0_3', 'right1_1', 'right2_1'], 0.15),],
                    'bot1_0': [
                        (['bot1_0', 'bot1_1', 'bot1_2'], 0.5), # wait for rewrite turns
                        (['bot1_0', 'right2_0'], 0.25),
                        (['bot1_0', 'left1_0', 'top0_0'], 0.1),
                        (['bot1_0', 'left1_0', 'left0_0'], 0.15),],
                    'top1_2': [
                        (['top1_2', 'top1_1', 'top1_0'], 0.5), # wait for rewrite turns
                        (['top1_2', 'right2_1'], 0.25),
                        (['top1_2', 'left1_1', 'bot0_3'], 0.1),
                        (['top1_2', 'left1_1', 'left0_1'], 0.15),],
                    'left2_0': [
                        (['left2_0', 'left1_0', 'left0_0'], 0.5), # wait for rewrite turns
                        (['left2_0', 'top1_0'], 0.25),
                        (['left2_0', 'bot1_1', 'right2_1'], 0.1),
                        (['left2_0', 'bot1_1', 'bot1_2'], 0.15),],
                    'right0_0': [
                        (['right0_0', 'right1_0', 'right2_0'], 0.5), # wait for rewrite turns
                        (['right0_0', 'top0_0'], 0.25),
                        (['right0_0', 'bot0_1', 'bot0_2', 'left0_1'], 0.1),
                        (['right0_0', 'bot0_1', 'bot0_2', 'bot0_3'], 0.15),],
                    'left2_1': [
                        (['left2_1', 'left1_1', 'left0_1'], 0.5), # wait for rewrite turns
                        (['left2_1', 'top1_1', 'top1_0'], 0.15),
                        (['left2_1', 'top1_1', 'right2_0'], 0.1),
                        (['left2_1', 'bot1_2'], 0.25),],
                    'right0_1': [
                        (['right0_1', 'right1_1', 'right2_1'], 0.5), # wait for rewrite turns
                        (['right0_1', 'bot0_3'], 0.25),
                        (['right0_1', 'top0_2', 'top0_1', 'left0_0'], 0.1),
                        (['right0_1', 'top0_2', 'top0_1', 'top0_0'], 0.15),],

                    'ramp': [
                        (['ramp', 'bot0_2', 'bot0_3'], 0.5),
                        (['ramp', 'bot0_2', 'left0_1'], 0.25),
                        (['ramp', 'bot0_2', 'right1_1', 'bot1_2'], 0.1),
                        (['ramp', 'bot0_2', 'right1_1', 'right2_1'], 0.15),]}
        else:          
            # only allow straight forward in intersections
            return {'bot0_0': ['bot0_0', 'bot0_1', 'bot0_2', 'bot0_3'],
                      'top0_3': ['top0_3', 'top0_2', 'top0_1', 'top0_0'],
                      'bot1_0': ['bot1_0', 'bot1_1', 'bot1_2'],
                      'top1_2': ['top1_2', 'top1_1', 'top1_0'],
                      'left2_0': ['left2_0', 'left1_0', 'left0_0'],
                      'right0_0': ['right0_0', 'right1_0', 'right2_0'],
                      'left2_1': ['left2_1', 'left1_1', 'left0_1'],
                      'right0_1': ['right0_1', 'right1_1', 'right2_1'],

                      'ramp': ['ramp', 'bot0_2', 'bot0_3']}