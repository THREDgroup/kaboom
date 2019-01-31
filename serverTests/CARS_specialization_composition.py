import numpy as np
import time as timer
import multiprocessing
from matplotlib import pyplot as plt
#import pickle
import itertools

from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.carMakers import carTeamWorkProcess
from kaboom.kaboom import saveResults


# Structure vs Composition
# Optimal structure of 32-agent team for 3 allocation strategies

t0 = timer.time()
p=Params()

#change team size and specialization
p.nAgents = 32
p.nDims = 56
p.steps = 100
p.reps = 32

nAgentsPerTeam = [1,2,3,4,8,16,32]#[32,16]# [8,4,3,2,1]    

#request @ 32 nodes 4gb: 800 sec per [100 step simulation of 32 agents x 32 reps ]
#request: 800 sec * 7*3 = 4.7 hours -> request 6 hrs

#resultMatrix = []
#teamObjects = []

teamObjectsFlat = [] 
for i in range(3):#range(3):
    if i == 0: #homogeneous
        p.aiRange = 0
        p.aiScore = 95
        p.curatedTeams = True
    elif i == 1:
        p.aiRange = 70
        p.aiScore = 95
        p.curatedTeams = False
    elif i == 2:
        p.aiScore = None
        p.aiRange = None
        p.curatedTeams = False
#    scoresA = []
#    teams = []
    for subteamSize in nAgentsPerTeam: 
        p.nTeams = int(p.nAgents/subteamSize)
        p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
        p.teamDims = m.teamDimensions(p.nDims,p.nTeams)
        if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
            pool = multiprocessing.Pool(processes = 16)
            allTeams = pool.starmap(carTeamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
#            scoresA.append([t.getBestScore() for t in allTeams])
#            teams.append(allTeams)
            for t in allTeams:
                teamObjectsFlat.append(t)
            print('finished one set: '+str(timer.time()-t0))
        pool.close()
        pool.join()
#    resultMatrix.append(scoresA)
#    teamObjects.append(teams)
    print("completed one composition type")
print("time to complete: "+str(timer.time()-t0))

name="structure_composition_100step"
directory = saveResults(teamObjectsFlat,p,name)



##Plot results:
#
#
#import pickle
#f = open('/Users/samlapp/SAE_ABM/kaboom/serverTests/results/1548947310.495178structure_composition_100step/results.obj', 'rb')
#teamObjectsFlat = pickle.load(f)
#
#teamObjects = np.reshape(teamObjectsFlat,[3,7,2])
#
#
#for i in range(3):
#    nAgents = [p.nAgents for teamSet in teamObjects[i] for team in teamSet]
##    nTeams = [len(team.specializations) for teamSet in teamObjects[i] for team in teamSet]
#    subTeamSize = [nAgentsPerTeam[j] for j in range(len(teamObjects[i])) for k in range(p.reps)]
#    teamScore =  [team.bestScore for teamSet in teamObjects[i] for team in teamSet]
#    print("Diverse team, size %s in %s dim space: " % (32,p.nDims))
##     plt.scatter(subTeamSize,teamScore,label='team size: '+str(teamSizes[i]))
#    m.plotCategoricalMeans(subTeamSize,np.array(teamScore)*-1)
#    
#plt.xlabel("subteam size")
##     plt.xticks([1,4,8,16,32])
#plt.ylabel("performance")
##     plt.show()
#plt.legend(['homogeneous','heterogeneous70','organic'])
#plt.title("composition vs structure")

#    plt.savefig('./results/viii_structure_composition.pdf')

#    return teamObjects
