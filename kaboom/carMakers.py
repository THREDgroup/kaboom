"""
This module has functions for running the simulation with CarDesigner agents.
"""
from kaboom import modelFunctions as m
from kaboom.kaboom import teamWorkSharing
from kaboom.params import Params
from kaboom.CarDesigner import CarDesigner

def carTeamWorkProcess(processID,p,agentConstructor = CarDesigner):
    """A wrapper for teamWorkSharing used for multiprocessing"""
    team = teamWorkSharing(p,agentConstructor)
    return team
