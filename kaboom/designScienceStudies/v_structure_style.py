"""
Run study of how specialization affects performance for different compositions.

This script recreates the results shown in Figure 13 of [1]. It compares the
optimal number of specialized teams for a specific problem and three different
homogeneous team compositions: adaptive, mid-range, and innovative KAI styles.


[1] Lapp, S., Jablokow, J., McComb, C. (2019). "KABOOM: An Agent-Based Model for Simulating Cognitive Style in Team Problem Solving". Unpulished manuscript.
"""

import numpy as np
import time as timer
import multiprocessing
from matplotlib import pyplot as plt
#import pickle
import itertools

#parameters for KABOOM
#import kaboom
from kaboom.params import Params

#import kaiStyle as kai
from kaboom import modelFunctions as m
#import helperFunctions as h
from kaboom.kaboom import teamWorkProcess

def run():
    #Strategy 4: Homogenous teams of 3 styles
    t0 = timer.time()
    p=Params()

    # teamSizes =[32]
    p.nAgents = 32
    nAgentsPerTeam = [32,16,8,4,3,2,1]
    # nTeams = [1,2,4,8,16,32]
    p.nDims = 32
    #p.reps=1

    # choose one Team allocation strategy
    p.aiRange = 0#0
    aiScores = [55,95,135]#140# 300

    #choose one problem (middle, favors mid-range)
    roughnesses = np.logspace(-1,.7,num=6,base=10)
    speeds = np.logspace(-1,.7,num=6,base=10) / 100
    p.amplitude = roughnesses[3]
    p.AVG_SPEED = speeds[3]


    resultMatrixH3 = []
    teamObjectsH3 = []
    for aiScore in aiScores:
        p.aiScore = aiScore
        scoresA = []
        teams = []
        for subteamSize in nAgentsPerTeam:
            p.nTeams = int(p.nAgents/subteamSize)
            p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
            p.teamDims = m.teamDimensions(p.nDims,p.nTeams)
            if __name__ == '__main__' or 'kaboom.designScienceStudies.v_structure_style':
                pool = multiprocessing.Pool(processes = 4)
                allTeams = pool.starmap(teamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
                scoresA.append([t.getBestScore() for t in allTeams])
                teams.append(allTeams)
            pool.close()
            pool.join()
        resultMatrixH3.append(scoresA)
        teamObjectsH3.append(teams)
        print("completed one")
    print("time to complete: "+str(timer.time()-t0))

    # In[ ]:


    #plot score vs structure for different team sizes
    for i in range(len(aiScores)):
        nAgents = [len(team.agents) for teamSet in teamObjectsH3[i] for team in teamSet]
        nTeams = [len(team.specializations) for teamSet in teamObjectsH3[i] for team in teamSet]
        subTeamSize = [int(len(team.agents)/len(team.specializations)) for teamSet in teamObjectsH3[i] for team in teamSet]
        teamScore =  [team.getBestScore() for teamSet in teamObjectsH3[i] for team in teamSet]
    #     print("homgeneous team, size %s in %s dim space: " % (teamSizes[i],nDims))
    #     plt.scatter(subTeamSize,teamScore,label='team size: '+str(teamSizes[i]))
        m.plotCategoricalMeans(subTeamSize,np.array(teamScore)*-1)
    #     plt.xscale('log')
        plt.xlabel("subteam size")
        plt.xticks(nAgentsPerTeam)
        plt.ylabel("performance")
    #     plt.show()
    plt.legend(aiScores)
    plt.title("homogeneous team of 3 styles")
    #np.savetxt("./results/C_3homog/params.txt",[makeParamString()], fmt='%s')
    #scoreMatrix = [ [ [t.getBestScore() for t in eachStructure] for eachStructure in eachStyle] for eachStyle in teamObjectsH3]
    #rFile = './results/C_3homog/'+'scoreMatrix_homo_3styles_32.obj'
    #rPickle = open(rFile, 'wb')
    #pickle.dump(scoreMatrix, rPickle)
    plt.savefig("./results/v_structure_3styles.pdf")
