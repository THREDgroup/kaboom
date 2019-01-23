"""
This module has functions for running the simulation with CarDesigner agents.
"""
from kaboom.kaboom import teamWorkSharing
from kaboom.params import Params
from kaboom.CarDesigner import CarDesigner

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
