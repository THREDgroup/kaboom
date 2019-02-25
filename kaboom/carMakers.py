"""
This module has functions for running the simulation with CarDesigner agents.
"""
from kaboom import modelFunctions as m
from kaboom.kaboom import teamWorkSharing
from kaboom.params import Params
from kaboom.CarDesigner import CarDesigner

def carTeamWorkProcess(processID,p,agentConstructor = CarDesigner):
    """A wrapper for teamWorkSharing used for multiprocessing
    
    parameters:
    processID : int
    p : Params object
    agentConstructor : constructor for desired Agent class 
        
    returns:
    team : Team object
        
    """
    team = teamWorkSharing(p,agentConstructor)
    return team
