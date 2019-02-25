"""
This module has functions for running the simulation with CarDesigner agents.
"""
import os
import pandas as pd

from kaboom import modelFunctions as m
from kaboom.kaboom import teamWorkSharing
#from kaboom.params import Params
#from kaboom.CarDesigner import CarDesigner
from kaboom.CarDesigner import CarDesignerWeighted


def carTeamWorkProcess(processID,p,agentConstructor = CarDesignerWeighted):
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


def runCarDesignProblem(p):
    """Run the Car Design problem and automatically configure the team 
    
    This function automatically configures p.nDims, p.nTeams, p.nTeams and 
    the team assignments/ problem decomposition p.teamDims and p.agentTeams
    
    parameters:
    p : Params object
        
    returns:
    team : Team object
        
    """
    #Load the car design problem variables and information
    myPath = os.path.dirname(__file__)
    paramsDF = pd.read_csv(myPath+"/SAE/paramDBreduced.csv")
    paramsDF = paramsDF.drop(["used"],axis=1)
    paramsDF.head()
    
    #change team size and specialization
    p.nAgents = 33
    p.nTeams = 11
    p.nDims = len(paramsDF)
    p.steps = 100

    #problem decomposition according to the car sub-teams:
    teams = ['brk', 'c', 'e', 'ft', 'fw', 'ia','fsp','rsp', 'rt', 'rw', 'sw']
    paramTeams = paramsDF.team
    p.nTeams = len(teams)
    teamDimensions_semantic = [[ 1 if paramTeam == thisTeam else 0 for paramTeam in paramTeams] for thisTeam in teams]
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = teamDimensions_semantic
    
    #run the simulation
    team = teamWorkSharing(p,CarDesignerWeighted)
    
    return team
