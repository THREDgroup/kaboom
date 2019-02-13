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
p.steps = 100 #100
p.reps = 16
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
teamObjectsSemantic = []
if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
    pool = multiprocessing.Pool(processes = 4)
    allTeams = pool.starmap(carTeamWorkProcess, zip(range(p.reps),itertools.repeat(p),itertools.repeat(CarDesignerWeighted)))
#            scoresA.append([t.getBestScore() for t in allTeams])
#            teams.append(allTeams)
    for t in allTeams:
        teamObjectsSemantic.append(t)
    pool.close()
    pool.join()
print("time to complete: "+str(timer.time()-t0))

name="allocation_semantic"
directory = saveResults(teamObjectsSemantic,p,name)


#NOW WITH BLIND TEAM DIMENSIONS INSTEAD OF SEMANTIC

#assign dimensions blindly to teams,with even # per team (as possible)
teamDimensions_blind = m.teamDimensions(p.nDims,p.nTeams)
p.teamDims = teamDimensions_blind
teamObjectsBlind = []
if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
    pool = multiprocessing.Pool(processes = 4)
    allTeams = pool.starmap(carTeamWorkProcess, zip(range(p.reps),itertools.repeat(p),itertools.repeat(CarDesignerWeighted)))
#            scoresA.append([t.getBestScore() for t in allTeams])
#            teams.append(allTeams)
    for t in allTeams:
        teamObjectsBlind.append(t)
    pool.close()
    pool.join()
name="allocation_blind"
directory = saveResults(teamObjectsBlind,p,name)
print("time to complete: "+str(timer.time()-t0))


#Plot results:
semanticScores = [t.getBestScore() for t in teamObjectsSemantic]
plt.scatter([t.nMeetings for t in teamObjectsSemantic],semanticScores)
blindScores = [t.getBestScore() for t in teamObjectsBlind]
plt.scatter([t.nMeetings for t in teamObjectsBlind],blindScores)
plt.legend(['semantic','blind'])
print(np.mean(semanticScores))
print(np.mean(blindScores))
print(h.pScore(semanticScores,blindScores))



#
#from kaboom.params import Params

solBlind = [t.getBestSolution() for t in teamObjectsBlind]
rBlind = np.array([list(sol.r) for sol in solBlind])
solSemantic = [t.getBestSolution() for t in teamObjectsSemantic]
rSemantic = np.array([list(sol.r) for sol in solSemantic])

for i in range(len(rSemantic[0])):
    p = h.pScore(rBlind[:,i],rSemantic[:,i])
    if p <.05:
        print('variable '+str(i))
        print(p)
        
y = np.transpose([semanticScores, blindScores])
x = [ [rSemantic[i], rBlind[i]] for i in range(len(rSemantic))]
np.shape(x)