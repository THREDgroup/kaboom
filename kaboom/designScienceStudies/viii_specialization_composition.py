import numpy as np
import time as timer
import multiprocessing
from matplotlib import pyplot as plt
#import pickle
import itertools

#parameters for KABOOM
#import kaboom
from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.kaboom import teamWorkProcess

# Structure vs Composition
# Optimal structure of 32-agent team for 3 allocation strategies

def run():
    t0 = timer.time()
    p=Params()

    p.nAgents = 32
    nAgentsPerTeam = [1,2,3,4,8,16,32]#[32,16]# [8,4,3,2,1]
    p.nDims = 32

    p.reps = 32


    #choose one problem (middle, favors mid-range)
    roughnesses = np.logspace(-1,.7,num=6,base=10)
    speeds = np.logspace(-1,.7,num=6,base=10) / 100
    p.amplitude = roughnesses[3]
    p.AVG_SPEED = speeds[3]


    resultMatrix = []
    teamObjects = []
    for i in range(3):
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
        scoresA = []
        teams = []
        for subteamSize in nAgentsPerTeam:
            p.nTeams = int(p.nAgents/subteamSize)
            p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
            p.teamDims = m.teamDimensions(p.nDims,p.nTeams)
            if __name__ == '__main__' or 'kaboom.designScienceStudies.viii_specialization_composition':
                pool = multiprocessing.Pool(processes = 4)
                allTeams = pool.starmap(teamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
                scoresA.append([t.getBestScore() for t in allTeams])
                teams.append(allTeams)
            pool.close()
            pool.join()
        resultMatrix.append(scoresA)
        teamObjects.append(teams)
        print("completed one")
    print("time to complete: "+str(timer.time()-t0))

    for i in range(3):
        nAgents = [len(team.agents) for teamSet in teamObjects[i] for team in teamSet]
        nTeams = [len(team.specializations) for teamSet in teamObjects[i] for team in teamSet]
        subTeamSize = [int(len(team.agents)/len(team.specializations)) for teamSet in teamObjects[i] for team in teamSet]
        teamScore =  [team.getBestScore() for teamSet in teamObjects[i] for team in teamSet]
        print("Diverse team, size %s in %s dim space: " % (32,p.nDims))
    #     plt.scatter(subTeamSize,teamScore,label='team size: '+str(teamSizes[i]))
        m.plotCategoricalMeans(subTeamSize,np.array(teamScore)*-1)

    plt.xlabel("subteam size")
    #     plt.xticks([1,4,8,16,32])
    plt.ylabel("performance")
    #     plt.show()
    plt.legend(['homogeneous','heterogeneous70','organic'])
    plt.title("composition vs structure")
    plt.savefig('./results/viii_structure_composition.pdf')
