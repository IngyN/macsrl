from enum import Enum
import logging
import math

import numpy as np
from copy import deepcopy


class InteractiveLandmarkStatus(Enum):
    CLEARED = 1,
    POSITIVE = 2,
    NEGATIVE = 3


class InteractiveLandmarkBelief(Enum):
    CLEARED = 1,
    KNOWN = 2,
    QUESTIONMARK = 3


class Annotation:
    
    def __init__(self,mdp, oa, obs_radius = math.inf, direction=False, adv_radius= math.inf, interactive=[], agents=2, targets=None, debug=False):
        self.ego_radius_constant = obs_radius
        self.resources = None
        self.xmax_constant = mdp.shape[0]-1 # TODO this might be flipped
        self.ymax_constant = mdp.shape[1]-1
        self.adv_has_direction = direction
        self.adv_radius_constant = adv_radius
        self.nr_cameras = 0
        self.trap_label = 'B'
        self.nr_adversaries = agents-1
        self.landmark_label = interactive
        self.nr_interactive_landmarks = len(interactive)
        self.target_label = targets
        self.n_agents = agents
        self.has_resources = (self.resources is not None)
        self.scan_action = None
        self.adv_draw_area_boundaries = False
        self.has_goal_action=True
        
        # parse # oa actions
        self.epsilon_actions = 0
        self.act_index = {}
        for i,a in enumerate(mdp.A):
            self.act_index[i] = i
                
        for q in oa.eps:
            for e in q:
                self.act_index[e+len(mdp.A)] = len(mdp.A)+self.epsilon_actions
                self.epsilon_actions+= 1
                
        self.parse_mdp(mdp, agents, targets, debug=debug)

    def parse_mdp(self, mdp, agents, targets, debug=False):

        self.states = []
        for i in range(agents):
            self.states.append(f'state_{i}')
        
        self.actions = []
        for i in range(self.n_agents):
            self.actions.append(f'action_{i}')
        
        self.interactive_landmark_constants = np.zeros([self.nr_interactive_landmarks, 2])
        
        self.target_loc = np.zeros([len(self.target_label), 2])
        
        for x in range(mdp.shape[0]):
            for y in range(mdp.shape[1]):
                for j in range(self.nr_interactive_landmarks):
                    if debug:
                        print(f' x:{x}, y:{y}, {mdp.label[x][y]}')
                    if self.landmark_label[j] in mdp.label[x][y]:
                        self.interactive_landmark_constants[j] = [x,y]
                
                for i in range(len(self.target_label)):
                    if self.target_label[i] in mdp.label[x][y]:
                        self.target_loc[i] = [x,y]
            
            

    def has_static_targets(self):
        return self.static_targets
    
    def has_landmarks(self):
        return (len(self.landmarks) >0)
    
    def landmark_labels(self):
        return self.landmark_labels
    
    def trap_labels(self):
        return self.trap_labels
    
    def target_labels(self):
        return self.target_labels
    
    def adv_goal_labels(self):
        return self.adv_goal_labels

    
    