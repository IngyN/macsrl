import time

import numpy as np

import gym

import torch

from copy import deepcopy

import lbforaging
from lbforaging.foraging import ForagingEnv

from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from .wrappers import RecordEpisodeStatistics

from .train import Train

# import csrl side of code
from csrl.oa import OmegaAutomaton




class LBFTrainRS(Train):
    """
    Training environment for the level-based foraging environment (LBF)
    """

    def __init__(self, args=None):
        """
        Create LBF Train instance
        """
        super(LBFTrainRS, self).__init__(args)
        self.label = None

    def parse_args(self):
        """
        parse own arguments including default args and rware specific args
        """
        self.parse_default_args()
        self.parser.add_argument(
            "--env", type=str, default=None, help="name of the lbf environment"
        )
        self.parser.add_argument(
            "--agents", type=str, default=2, help="number of agents"
        )
        self.parser.add_argument("--env_coef", type=float, default=0.0, help="environment reward coefficient for reward weighting"
        )
        self.parser.add_argument("--oa_coef", type=float, default=1.0, help="automaton reward coefficient for reward weighting"
        )
        self.parser.add_argument("--discount_oa", type=float, default=0.99, help="discount value applied to oa state rewards"
        )
        self.parser.add_argument("--ltl", type=str, default="(e | XF e)", help="LTL formula for the omega automaton"
        )
    
    # This is where to add labels based on desired functionality
    def _make_labels(self):
        pass
    
    def create_environment(self):
        """
        Create environment instance
        :return: environment (gym interface), env_name, task_name, n_agents, observation_sizes,
                 action_sizes
        """
        # load scenario from script
        
        # restrict type of foraging env with modified env.  
        
        if self.arglist.env is None:
            env = ForagingEnv(players=self.arglist.agents, max_player_level=2, field_size=(8,8), max_food=1, sight=8, force_coop=False, max_episode_steps=50)
        else:
            env = gym.make(self.arglist.env)
        env = RecordEpisodeStatistics(env, deque_size=10)
        
        task_name = self.arglist.env
        self.env_coef= self.arglist.env_coef
        self.oa_coef=self.arglist.oa_coef
        self.n_agents = env.n_agents
        self.reward_amount = 1 - self.arglist.discount_oa
        
        self._make_labels() # call the label making function if desired
        
        # create omega automaton
        self.oa = OmegaAutomaton(self.arglist.ltl, extra_aps = ('f','e'))
        
        # get number of actions to add
        # format for each state q of oa -> (action number, next state)
        # Assumption: actionspaces for all agents are the same. 
        self.sum_eps = 0
        self.eps_actions= [[] for i in range(self.oa.shape[1])]
        for q in range(self.oa.shape[1]):
            for ep in self.oa.eps[q]:
                self.sum_eps+= 1 
                self.eps_actions[q].append( (self.sum_eps+env.action_space[0].n-1, ep))
        # self.sum_eps
        
        #print(self.eps_actions)

        # print("Original Observation spaces: ", [env.observation_space[i] for i in range(n_agents)])
        # print("Original Action spaces: ", [env.action_space[i] for i in range(n_agents)])
        
        self.observation_sizes = self.extract_sizes(env.observation_space, k=1)
        self.action_sizes = self.extract_sizes(env.action_space, k=self.sum_eps)
        self.original_action_size = self.extract_sizes(env.action_space)
        
        # adjusted size add +1 to action space for each epsilon action
        action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.action_sizes[0])] * self.n_agents))
        
        # adjust observation size -> +1 for q from oa for every agent
        observation_space = gym.spaces.Tuple(tuple(self._get_observation_space(env.observation_space, self.n_agents)))
        
        print("Observation spaces: ", observation_space)
        print("Action spaces: ", action_space)
        self.env=env
        

        return (
            env,
            task_name,
            "lbf_reward_shaping",
            self.n_agents,
            observation_space,
            action_space,
            self.observation_sizes,
            self.action_sizes,
        )
    
    # observation size -> +1 for q from oa for every agent
    def _get_observation_space(self, o_space, n_agents):
        
        min_obs = deepcopy(o_space[0].low)
        max_obs = deepcopy(o_space[0].high)
        
        # add the low and high for q state from the automaton
        min_obs = np.append(min_obs, 0)
        max_obs = np.append(max_obs, self.oa.shape[1])
        
        return [gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)]*n_agents
        
        
    #  -> edited this 
    def extract_sizes(self, spaces, k = 0):
        """
        Extract space dimensions
        :param spaces: list of Gym spaces
        :return: list of ints with sizes for each agent
        """
        sizes = []
        for space in spaces:
            if isinstance(space, Box):
                size = sum(space.shape) +k
            elif isinstance(space, Dict):
                size = sum(self.extract_sizes(space.values()))
            elif isinstance(space, Discrete) or isinstance(space, MultiBinary):
                size = space.n +k
            elif isinstance(space, MultiDiscrete):
                size = sum(space.nvec)
            else:
                raise ValueError("Unknown class of space: ", type(space))
            sizes.append(size)
        return sizes
    
    def reset_environment(self):
        """
        Reset environment for new episode
        :return: observation (as torch tensor)
        """
        obs = np.zeros([self.n_agents, self.observation_sizes[0]])
        
        # reset environment
        temp_obs = self.env.reset() 
        # reset omega automaton
        self.q = self.oa.q0
        
        # add oa state to each agent's observation 
        for i in range(len(temp_obs)):
            obs[i] = np.append(temp_obs[i], self.q)
        
        # Make into 1 array
        obs = tuple([np.expand_dims(o, axis=0) for o in obs])
        return obs

    def select_actions(self, obs, explore=True):
        """
        Select actions for agents
        - get actions as torch Variables 
        - check if eps is available at this state
        - if yes modify available actions to add epsilon actions
        :param obs: joint observations for agents
        :return: actions, onehot_actions
        """
        available_actions = np.zeros([self.env.n_agents, self.action_sizes[0]], dtype=bool)
        
        # get valid actions from environment
        valid_actions= self.env.get_available_actions()
        
        # Initialize eps actions to False
        for i in range(self.env.n_agents):
            available_actions[i] = np.append(valid_actions[i],([False]*self.sum_eps))
        
        # Set eps actions to true if available
        if len(self.oa.eps[self.q]) > 0:
            for a in self.eps_actions[self.q]:
                for i in range(self.env.n_agents):
                    available_actions[i][a[0]] = True
        
        torch_agent_actions = self.alg.step(obs, explore, available_actions=available_actions)
        
        agent_actions = (torch_agent_actions).cpu()
        # convert actions to numpy arrays
        onehot_actions = [ac.data.numpy() for ac in agent_actions]
        # convert onehot to ints
        actions = np.argmax(onehot_actions, axis=-1)

        return actions, onehot_actions
    
    # Function to return labels -> this can be edited to add additional positional based labels
    def _get_labels(self ):
        pos = self.env.get_player_positions()
        total_food = self.env.get_total_food()
        
        # an array or positions containing labels can be created and accessed this way:
        if self.label is not None:
            labels = list(self.label[pos[0]][pos[1]])
        else:
            labels =[]
            
        if total_food > 0:
            labels.append('f')
        else:
            labels.append('e')

        return tuple(sorted(labels))
    
    def environment_step(self, actions):
        """
        Take step in the environment
        :param actions: actions to apply for each agent
        :return: reward, done, next_obs (as Pytorch tensors), info
        """
        # environment step
        # TODO: check if any of the actions is epsilon action
        # if epsilon -> transition + set as NONE
        # then call env.step.
        eps_actions =[]
        for i,a in enumerate(actions):
            if a > self.original_action_size[i]: # check if it's an epsilon action
                eps_actions.append(a)
                actions[i] = 0 # -> NONE for lb-foraging 
        
        if len(eps_actions)>0:
            eps_action = random.choice(eps_actions) # TODO I dont have a good way to choose if multiple oa epsilon actions -> randomly choose one of them. 
            for a in self.eps_actions[self.q]:
                if a[0] == eps_action: # found the right entry for eps action
                    self.q = a[1] # transition oa state
        
        temp_obs, reward, done, info = self.env.step(actions)
        
        # create observation array with augmented size (includes oa state)
        next_obs = np.zeros([self.n_agents, self.observation_sizes[0]])
        
        for i in range(len(temp_obs)):
            next_obs[i] = np.append(temp_obs[i], self.q) 
            
        next_obs = [np.expand_dims(o, axis=0) for o in next_obs]
        
        
        # get labels and transition oa state
        global_l=self._get_labels()
        self.q = self.oa.delta[self.q][global_l]
        
        # edit reward based on oa acceptance state 
        for i in range(len(reward)):
            reward[i] =self.env_coef*reward[i] + self.oa_coef*(self.reward_amount if self.oa.acc[self.q][global_l][0] else 0)
        
        return reward, done, next_obs, info

    def environment_render(self, plt=None, step=0, info="", inline=True):
        """
        Render visualisation of environment
        """
        if inline:
            plt.figure(3)
            plt.clf()
            plt.imshow(self.env.render(mode="rgb_array"))
            plt.title(f" Step:{step} {info}")
            plt.axis('off')
            
        else:
            self.env.render()
        time.sleep(0.1)


if __name__ == "__main__":
    train = LBFTrainRS()
    train.train()
