"""
Run car design problem for semantic and blind problem decomposition

This recreates the fourth experiment from IDETC , and demonstrates that a team
can perform better if it uses a random allocation of variables to sub-teams
rather than the semantic decomposition (engine variables together, cabin 
variables together, etc).

The results are plotted and saved to /results/ folder
"""
import numpy as np
import time as timer
import multiprocessing
import pandas as pd
from matplotlib import pyplot as plt
#import pickle
import itertools
import os

from kaboom import helperFunctions as h
from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.carMakers import carTeamWorkProcess
from kaboom.kaboom import saveResults
from kaboom.CarDesigner import CarDesignerWeighted

# Does it matter which variables we allocate to which subteams?
def run(numberOfCores=4):
    t0 = timer.time()
    p=Params()
    
    #change team size and specialization
    p.nAgents = 33
    p.nDims = 56
    p.steps = 100 #100
    p.reps = 16
    
    
    myPath = os.path.dirname(__file__)
    parentPath = os.path.dirname(myPath)
    paramsDF = pd.read_csv(parentPath+"/SAE/paramDBreduced.csv")
    paramsDF = paramsDF.drop(["used"],axis=1)
    paramsDF.head()
    
    teams = ['brk', 'c', 'e', 'ft', 'fw', 'ia','fsp','rsp', 'rt', 'rw', 'sw']
    teamsDict = { i:teams[i] for i in range(10)}
    paramTeams = paramsDF.team
    p.nTeams = len(teams)
    #in the semantic division of the problem, variables are grouped by parts of
    #the car (eg, wheel dimensions; engine; brakes)
    teamDimensions_semantic = [[ 1 if paramTeam == thisTeam else 0 for paramTeam in paramTeams] for thisTeam in teams]
    
    
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = teamDimensions_semantic
       
    
    if __name__ == '__main__' or 'kaboom.IDETC_studies.iv_problemDecomposition':
        pool = multiprocessing.Pool(processes = 4)
        teamObjectsSemantic = pool.starmap(carTeamWorkProcess, zip(range(p.reps),itertools.repeat(p),itertools.repeat(CarDesignerWeighted)))
        pool.close()
        pool.join()
    print("finished semantic: "+str(timer.time()-t0))
    
    #    name="allocation_semantic"
    #    directory = saveResults(teamObjectsSemantic,p,name)
    
    #NOW WITH BLIND TEAM DIMENSIONS INSTEAD OF SEMANTIC
    
    #assign dimensions blindly to teams,with even # per team (as possible)
    teamDimensions_blind = m.teamDimensions(p.nDims,p.nTeams)
    p.teamDims = teamDimensions_blind
    
    if __name__ == '__main__' or 'kaboom.IDETC_studies.iv_problemDecomposition':
        pool = multiprocessing.Pool(processes = numberOfCores)
        teamObjectsBlind = pool.starmap(carTeamWorkProcess, zip(range(p.reps),itertools.repeat(p),itertools.repeat(CarDesignerWeighted)))
        pool.close()
        pool.join()
    #    name="allocation_blind"
    #    directory = saveResults(teamObjectsBlind,p,name)
    print("finished blind: "+str(timer.time()-t0))
    
    #inverted scores so that its a maximization problem
    #(in the plot, higher scores are better)
    semanticScores = [t.getBestScore()*-1 for t in teamObjectsSemantic]
    blindScores = [t.getBestScore()*-1 for t in teamObjectsBlind]
    
    
    #Plot results: 
    plt.boxplot([semanticScores,blindScores],labels= ["semantic","blind"],showfliers=True)
    plt.ylabel("car design performance")
    
    plt.savefig(myPath+"/results/iv_problemDecomposition.pdf")
    plt.show()
    plt.clf()    
    
    print("effect size:" )
    print(h.effectSize(semanticScores,blindScores))
    print("ANOVA p score: ")
    print(h.pScore(semanticScores,blindScores))
    
    
