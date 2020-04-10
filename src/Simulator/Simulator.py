####################
# Simulator for task 1
####################
from typing import List

from RandomInit import RandomInit
from UniformInit import UniformInit
from holonomic_agent import Agent
from RunTask1 import RunTask1
from RunTask2 import RunTask2
from SimulatorRun import save_state
from PID import PID
import numpy as np
import random as rand


class Simulator:
    def __init__(self, timesteps: int, n: int, l: float, masvel: float, minrange=0.0, maxrange=99999.0, uniform=False):
        self.timesteps: int = timesteps
        self.N: int = n
        self.L: float = l
        self.masVel: float = masvel
        self.minRange: float = minrange
        self.maxRange: float = maxrange
        self.uniform: bool = uniform
        self.dt = 0.1

    def defineL(self):
        return self.L

    def defineN(self):
        return self.N

    def run(self, init=None, control=None, parameter = None, goal_set = True, Linit=None):
        n_agents = self.defineN()
        timesteps = self.timesteps
        dt= self.dt
        L = self.defineL()

        agents_list = []

        # Create list of agents
        if self.uniform:
            init=UniformInit(n_agents,L,0.06).create()
        else:
            if init is None:            
                init=RandomInit(n_agents,L,0.06).create()
            else:
                L = Linit            
        for i in range(n_agents):                
            agents_list.append(Agent(init[i], i, self.minRange, self.maxRange))

        # li ordino in base alla posizione
        agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

        # initialize agents and simulation state

        for i in range(n_agents): 
            agents_list[i].state, agents_list[i].vels = agents_list[i].observe(agents_list,L,i)

        # Initialize targets
        goal_list = [None]*n_agents
        

        if goal_set:
            agent_width = agents_list[0].width
            free_space = L - (n_agents*agent_width*2)
            # assert free_space > 0
            space_dist = free_space / (n_agents + 1)
            # first agent
            goal_list[0]= space_dist + agent_width
            if n_agents > 2:
                for i in range(1,n_agents-1):
                    goal_list[i]= space_dist + agent_width + (2*agent_width + space_dist)*i            
            if n_agents > 1:
                goal_list[n_agents-1]=L - (agent_width + space_dist)
        else:
            dist = L/(n_agents+1)            
            for i in range(n_agents):
                goal_list[i]=dist*(i+1)        

        # Parameters
        if parameter == 'add_vel':
            state_size = len(agents_list[i].state)+len(agents_list[i].vels)+3
        else:
            state_size = len(agents_list[i].state)+3 # x, y, theta + stato
        states = np.zeros((timesteps,n_agents,state_size)) 
        target_vels = np.zeros((timesteps, n_agents))
        errors = np.zeros((timesteps,), dtype=np.float32)
       
        #save initial state
        states[0]= save_state(agents_list, state_size)
   
        # initialize controller
        if control is None:
            controller = PID(-5,0,0) #(-5, 0, 0)
            comms = None
            for t in range (0, timesteps):
                for i in range( 0, len(agents_list)):
                    agents_list[i].state,  agents_list[i].vels= agents_list[i].observe(agents_list,L, i)
                    error = agents_list[i].getxy()[0]-goal_list[i]
                    v = controller.step(error, dt)
                    v = np.clip(v, -self.masVel, +self.masVel)
                    target_vels[t,i] = v
                # save state

                states[t] = save_state(agents_list, state_size)                
                state_error = states[t,:,0]    
                errors[t] = np.mean(abs(state_error-np.array(goal_list)))

                if agents_list[0].getxy()[0]<=0: #debug
                    import pdb; pdb.set_trace()  # breakpoint 8f194059 //

                # update agents
                for i,agent in enumerate(agents_list):
                    old_position = agent.getxy()[0]
                    agent.step(target_vels[t,i],dt)

                    #check collisions
                    if not agent.check_collisions(agents_list[:i]+agents_list[i+1:],L):
                        agent.setx(old_position)
                        target_vels[t,i]=0.0
                    


#################################################################
        else:
            net_controller = control.controller()
            net_run = RunTask1(L, controller=net_controller, dt=0.1)
            trace = net_run(states[0], agents_list, goal_list, self.masVel, state_size, epsilon=0.01, T=timesteps)
            states[:trace.state.shape[0],:,0]=trace.state
            states[:trace.sensing.shape[0],:,3:]=trace.sensing
            target_vels[:trace.control.shape[0]]=trace.control
            errors[:trace.error.shape[0]]=trace.error
            comms= trace.communication

#################################################################
        return states, target_vels, errors, comms

class SimulatorR(Simulator):
    def __init__(self, timesteps, N, mL, ML, masVel, minRange = 0.0, maxRange = 9999999.0, uniform = False):
        self.timesteps = timesteps
        self.N = N
        self.mL = mL
        self.ML = ML
        self.mas_vel = masVel
        self.dt =0.1
        self.min_range = minRange
        self.max_range = maxRange
        self.uniform = uniform

    def defineL(self):
        L = round(rand.random.uniform(self.mL, self.ML), 2)
        return L

class Simulator2(Simulator):
    def __init__(self, timesteps, N, L, masVel, minRange = 0.0, maxRange = 9999999.0, uniform = False):
        super(Simulator2, self).__init__(timesteps, N, L, masVel, minRange, maxRange, uniform)

    def run(self, init= None, control=None, parameter = None, Linit=None):
        n_agents = self.defineN()

        if init is not None:
            if len(init)!=n_agents:
                n_agents=len(init)

        timesteps = self.timesteps
        dt= self.dt
        L = self.defineL()

        agents_list = []

        # Create list of agents
        if self.uniform:
            init=UniformInit(n_agents,L,0.06).create()
        else:
            if init is None:            
                init=RandomInit(n_agents,L,0.06).create()
            else:                
                L = Linit            
        for i in range(n_agents):                
            agents_list.append(Agent(init[i], i, self.minRange, self.maxRange))

        # li ordino in base alla posizione
        agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

        # initialize agents and simulation state

        for i in range(n_agents): 
            agents_list[i].state, agents_list[i].vels = agents_list[i].observe(agents_list,L,i)

        # Initialize targets
        goal_list = [None]*n_agents
        for i in range(n_agents//2):
            goal_list[i]=0
        i=i+1
        for j in range(len(goal_list)-i):
            goal_list[i+j]=1        

        # Parameters
        state_size = len(agents_list[i].state)+3 # x, y, theta + stato
        states = np.zeros((timesteps,n_agents,state_size)) 
        target_colors = np.zeros((timesteps, n_agents))
        errors = np.zeros((timesteps,), dtype=np.float32)
       
        #save initial state
        states[0]= save_state(agents_list, state_size)
   
        # initialize controller
        if control is None:
            comms = None
            for t in range (0, timesteps):
                for i in range( 0, len(agents_list)):
                    agents_list[i].state,  agents_list[i].vels= agents_list[i].observe(agents_list,L, i)
                    agents_list[i].color = goal_list[i]
                    target_colors[t,i] = goal_list[i]
                # save state

                states[t] = save_state(agents_list, state_size)                
                errors[t] = 0.0

#################################################################
        ### ATTENZIONE CHE TEST SYNC LI O FATTI CON VECCHIA VERSIONE IN CUI IL NET CONTROLLER ERA SEMPRE SEQUENZIALE
        ### Domanda funiona megli sync - sync.  sequenziale - sync. oppure sync - sequenziale?
        else:
            net_controller = control.controller()            
            net_run = RunTask2(L, controller=net_controller, dt=0.1)
            trace = net_run(states[0], agents_list, goal_list, self.masVel, state_size, epsilon=0.01, T=timesteps)
            states[:trace.state.shape[0],:,0]=trace.state
            states[:trace.sensing.shape[0],:,3:]=trace.sensing
            target_colors[:trace.control.shape[0]]=trace.control
            errors[:trace.error.shape[0]]=trace.error
            comms= trace.communication

#################################################################
        return states, target_colors, errors, comms

class SimulatorN(Simulator2):
    def __init__(self, timesteps, Ns, L, masVel, minRange = 0.0, maxRange = 9999999.0, uniform = False):
        self.timesteps = timesteps
        self.Ns = Ns
        self.L = L
        self.mas_vel = masVel
        self.dt =0.1
        self.min_range = minRange
        self.max_range = maxRange
        self.uniform = uniform

    def defineN(self):
        N = rand.choice(self.Ns)
        return N

