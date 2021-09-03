from csrl.mdp import GridMDP
from csrl.oa import OmegaAutomaton
from csrl import ControlSynthesis
import numpy as np

ltl = '(F G a | F G b) & G !c'  # LTL formula
oa = OmegaAutomaton(ltl)  # LDBA
print('LDBA Size (including the trap state):',oa.shape[1])
#LDBA Size (including the trap state): 4

shape = (5,4)  # Shape of the grid
structure = np.array([  # E:Empty, T:Trap, B:Obstacle
 ['E',  'E',  'E',  'E'],
 ['E',  'E',  'E',  'T'],
 ['B',  'E',  'E',  'E'],
 ['T',  'E',  'T',  'E'],
 ['E',  'E',  'E',  'E']
])
label = np.array([  # Labels
 [(),       (),     ('c',),()],
 [(),       (),     ('a',),('b',)],
 [(),       (),     ('c',),()],
 [('b',),   (),     ('a',),()],
 [(),       ('c',), (),    ('c',)]
],dtype=object)
grid_mdp = GridMDP(shape=shape,structure=structure,label=label)
grid_mdp.plot()

csrl = ControlSynthesis(grid_mdp,oa) # Product MDP

Q=csrl.q_learning(T=100,K=100000)  # Learn a control policy
value=np.max(Q,axis=4)
policy=np.argmax(Q,axis=4)
print(policy[0,0])
# array([[1, 3, 0, 2],
#        [2, 3, 3, 6],
#        [0, 3, 0, 2],
#        [6, 0, 5, 0],
#        [3, 0, 0, 0]])
