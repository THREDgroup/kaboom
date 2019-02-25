"""
Contains function to run an example case of the beam designer problem
"""
from kaboom import modelFunctions as m
from kaboom.params import Params
from kaboom.BeamDesigner import BeamDesigner
from kaboom.kaboom import teamWorkSharing

def runBeamDesignProblem():
    """
    Run an example case of the beam designer problem
    """
    p= Params()
    p.nAgents = 8
    p.nDims = 4
    p.nTeams = 2
    p.reps = 32
    p.steps = 100
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = m.teamDimensions(p.nDims,p.nTeams)
    
    t= teamWorkSharing(p,BeamDesigner)
    return t 