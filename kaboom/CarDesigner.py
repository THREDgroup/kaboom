import numpy as np

from kaboom import carObjective
from kaboom.agent import Agent
from kaboom.params import Params
from kaboom.solution import Solution
from kaboom.kaboom import teamWorkSharing
from kaboom import kaiStyle as kai



from kaboom import modelFunctions as m

class CarDesigner(Agent):
    """
    These Agents design an SAE racecar

    """
    def __init__(self, p):
        """
        Create a CarDesigner agent
        
        Parameters:
        ----------
        p : Params object, contains current model Parameters
        """
        Agent.__init__(self,p)
        self.myDims = np.ones(p.nDims)
        self.car = carObjective.startCarParams()
        self.r = carObjective.normalizedCarVector(self.car)
        
    def reset(self,p):
        """ Reset the agent to it's initial birth conditions. """
        self.temp = self.startTemp
        self.speed = kai.calcAgentSpeed(self.kai.KAI,p)
        self.memory = []
        
        self.car = carObjective.startCarParams()
        self.r = carObjective.normalizedCarVector(self.car)

        self.nmoves = 0
        self.score = np.inf
        
    def moveTo(self,r):
        """
        Change the agent's current solution to r and update it's score.

        Also add the new solution to the agent's memory and increment the
        agent's move countself.

        Parameters:
        ----------
        r : list, shape = [nDims]
            the new solution to move to
        """

        self.r = r
        self.car = carObjective.normCarVector_to_car(r) 
        self.score = self.f()
        self.memory.append(Solution(self.r,self.score,type(self)))
        self.nmoves += 1
        
    def startAt(self,r):
        """
        Define the initial position of the agent and wipe its memory.

        Parameters:
        ----------
        position : list, the initial solution to start at
        """
        self.r = r
        self.car = carObjective.normCarVector_to_car(r) 
        self.memory = [Solution(r=self.r,score=self.f(),agent_class=type(self))]
        
        
        
    def f(self): #evaluate objective function for this agent's current solution
        self.car = carObjective.normCarVector_to_car(self.r)
        return carObjective.objective(self.car)
    def fr(self,new_r):
        newCar = carObjective.normCarVector_to_car(new_r)
        return carObjective.objective(newCar)

    #constrain vector solution [r] to the feasible space 
    #returns constrained solution
    def constrain(self,r,p):
        self.car = carObjective.normCarVector_to_car(r)
        carConstrained = carObjective.constrain(self.car)
        rConstrained = carObjective.normalizedCarVector(carConstrained)
        return rConstrained #[-1 for i in range(len(x))]    
    
    
#wrapper for teamWorkSharing used for multiprocessing
def carTeamWorkProcess(processID,p,agentConstructor = CarDesigner):
    team = teamWorkSharing(p,agentConstructor)
    return team


p=Params()
p.nDims = 56
p.nAgents = 6
p.nTeams = 3
p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
p.teamDims = m.teamDimensions(p.nDims,p.nTeams)


p.groupConformityBias=False
c=CarDesigner(p)
p.AVG_SPEED = .001
p.steps = 20
p.meetingTimes = 10


#c.myDims = np.zeros(56)
#c.myDims[10]=1
t = teamWorkSharing(p,CarDesigner)