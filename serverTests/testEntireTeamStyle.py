import numpy as np
import time as timer
import multiprocessing
import pandas as pd
from matplotlib import pyplot as plt
#import pickle
import itertools

from kaboom import helperFunctions as h
from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.carMakers import carTeamWorkProcess
from kaboom.kaboom import saveResults
from kaboom.CarDesigner import CarDesignerWeighted

# Does it matter which variables we allocate to which subteams?

t0 = timer.time()
p=Params()

#change team size and specialization
p.nAgents = 33
p.nDims = 56
p.steps = 100
p.reps = 4#8 #16
#
#nAgentsPerTeam = [1,2,3,4,8,16,32]#[32,16]# [8,4,3,2,1] 

paramsDF = pd.read_csv("../kaboom/SAE/paramDBreduced.csv")
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
   

#
##request @ 32 nodes 4gb: 800 sec per [100 step simulation of 32 agents x 32 reps ]
##request: 800 sec * 7*3 = 4.7 hours -> request 6 hrs

styleTeams = []
aiScores = [45,75,115,135]
for aiScore in aiScores:
    p.aiScore = aiScore
    p.aiRange = 0
    teamObjects = []
    if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
        pool = multiprocessing.Pool(processes = 4)
        allTeams = pool.starmap(carTeamWorkProcess, zip(range(p.reps),itertools.repeat(p),itertools.repeat(CarDesignerWeighted)))
    #            scoresA.append([t.getBestScore() for t in allTeams])
    #            teams.append(allTeams)
        for t in allTeams:
            teamObjects.append(t)
        pool.close()
        pool.join()
    print("time to complete: "+str(timer.time()-t0))
    styleTeams.append(teamObjects)

for i,style in enumerate(styleTeams):
    kaiScore = aiScores[i]
    scores = [t.getBestScore() for t in style]
    ai = [kaiScore for i in range(p.reps)]
    plt.scatter(ai,scores)
    
plt.scatter(np.ones(16)*95,[-43697.83046474878,
 -53539.600755698295,
 -51475.71127076317,
 -42580.22982036467,
 -55799.1550135683,
 -48337.7879095872,
 -53567.45303120078,
 -45038.94097887598,
 -54929.55670206938,
 -45854.197285548806,
 -53866.196861206874,
 -56265.54094660408,
 -55839.239946908034,
 -50304.24237876096,
 -46157.816942948375,
 -41412.904269363855], c='purple')

#name="allocation_semantic"
#directory = saveResults(teamObjectsSemantic,p,name)

#Plot results:
#semanticScores = [t.getBestScore() for t in teamObjectsSemantic]