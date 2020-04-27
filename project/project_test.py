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

# Experiment parameters
N_ROLLOUTS = 63  # number of rollouts per training iteration
N_CPUS = 63  # number of parallel workers

# Environment parameters
HORIZON = 400  # time horizon of a single rollout
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 25, 25, 25, 25

EDGE_INFLOW = 300  # inflow rate of vehicles at every edge
N_ROWS = 2  # number of row of bidirectional lanes
N_COLUMNS = 2  # number of columns of bidirectional lanes

# Create default environment parameters
env_params = EnvParams()

# Vehicle definition
vehicles = VehicleParams()
num_vehicles = (N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
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

# inflow configuration
inflow = InFlows()
# off ramp inflow
inflow.add(
    veh_type='human',
    edge='ramp',
    probability=0.1,
    departLane='free',
    departSpeed=20)

# other inflow edges
outer_edges = ['bot1_0', 'bot0_0', 'top1_2', 'top0_3', 'left2_0', 'left2_1', 'right0_0', 'right0_1']
for i in range(len(outer_edges)):
    inflow.add(
        veh_type='human',
        edge=outer_edges[i],
        probability=0.25,
        departLane='free',
        departSpeed=20)

# Network parameters
net_params = NetParams(
    inflows=inflow,
    # used when routes only allow straight forward in intersections
    # template=os.path.abspath('/home/dingyizhuang/notebookWorkplace/flow/examples/exp_configs/rl/multiagent/network/off_ramp_grid_straight.net.xml'),
    # used for all situations
    template=os.path.abspath('/home/dingyizhuang/notebookWorkplace/flow/examples/exp_configs/rl/multiagent/network/off_ramp_grid_turn.net.xml'),
    additional_params={
            "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
            "grid_array": {
                "short_length": 300,
                "inner_length": 300,
                "long_length": 300,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": 1,
            "vertical_lanes": 1,
        },
)

# specify the edges vehicles can originate on
initial_config = InitialConfig(
    edges_distribution=['bot1_0', 'bot0_0', 'top1_2', 'top0_3', 'left2_0', 'left2_1', 'right0_0', 'right0_1', 'ramp']
)

# whether to allow turns at intersections
ALLOW_TURNS = True

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
    tl_logic.add("center0", phases=phases, programID=1, tls_type="actuated")
    tl_logic.add("center1", phases=phases, programID=1, tls_type="actuated")
    tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")
    tl_logic.add("center3", phases=phases, programID=1, tls_type="actuated")
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

    tl_logic.add("center0", phases=phase0, programID=1, tls_type="actuated")
    tl_logic.add("center1", phases=phases, programID=1, tls_type="actuated")
    tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")
    tl_logic.add("center3", phases=phases, programID=1, tls_type="actuated")

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
        mapping['center1'] = ['bot0_1','ramp','top0_2']
        mapping['center2'] = ['bot0_2','right0_1','top0_3','left1_1']
        mapping['center3'] = ['bot1_0','right1_0','top1_1','left2_0']
        mapping['center4'] = ['bot1_1','right1_1','top1_2','left2_1']

        return sorted(mapping.items(), key=lambda x: x[0])

    def specify_routes(self, net_params):
        """See parent class."""
        
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

'''
env=EnvParams(
        horizon=1500,
        additional_params={
            "target_velocity": 50,
            "switch_time": 3,
            "num_observed": 2,
            "discrete": False,
            "tl_type": "actuated",
            "num_local_edges": 4,
            "num_local_lights": 4,
        }
'''

flow_params = dict(
    exp_tag='A traffic light grid with exit ramp from highway',
    env_name=AccelEnv,
    network=offRampGrid,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
    ),
    env=EnvParams(
        horizon=1500,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=tl_logic,
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

'''
def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with a single policy graph for all agents
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']
'''