import numpy as np
import time as timer
import multiprocessing
from matplotlib import pyplot as plt
#import pickle
import itertools

from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.kaboom import teamWorkProcess

def run():
    
    # Study performance on subteam-size vs number of subteam plot
    # for different subteam allocation strategies 
    #Organic Teams
    
    t0 = timer.time()
    p=Params()
    
    teamSizes =[32]#[8,16,32]
    nAgentsPerTeam = [1,2,3,4,8,16,32]#range(1,8+1)
    p.nDims = 32
    
    p.pComm = 0.2 # np.linspace(0,.5,10)
    #p.reps = 1
    
    #choose one problem (middle, favors mid-range)
    roughnesses = np.logspace(-1,.7,num=6,base=10)
    speeds = np.logspace(-1,.7,num=6,base=10) / 100
    p.amplitude = roughnesses[3] 
    p.AVG_SPEED = speeds[3] 
    
    
    resultMatrixOrganic = []
    teamObjectsOrganic = []
    for nAgents in teamSizes:
        p.nAgents = nAgents
        scoresA = []
        teams = []
        for subteamSize in nAgentsPerTeam: 
            p.nTeams = int(p.nAgents/subteamSize)
            p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
            p.teamDims = m.teamDimensions(p.nDims,p.nTeams)
            if __name__ == '__main__' or 'kaboom.test.iv_specialization_organic':
                pool = multiprocessing.Pool(processes = 4)
                allTeams = pool.starmap(teamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
                scoresA.append([t.getBestScore() for t in allTeams])
                teams.append(allTeams)
            pool.close()
            pool.join()
        resultMatrixOrganic.append(scoresA)
        teamObjectsOrganic.append(teams)
        print("completed one")
    print("time to complete: "+str(timer.time()-t0))
    
    
    # In[113]:
    
    
    #plot score vs structure for different team sizes
    for i in range(len(teamSizes)):
        nAgents = [len(team.agents) for teamSet in teamObjectsOrganic[i] for team in teamSet]
        nTeams = [len(team.specializations) for teamSet in teamObjectsOrganic[i] for team in teamSet]
        subTeamSize = [int(len(team.agents)/len(team.specializations)) for teamSet in teamObjectsOrganic[i] for team in teamSet]
        teamScore =  [team.getBestScore() for teamSet in teamObjectsOrganic[i] for team in teamSet]
        print("Diverse team, size %s in %s dim space: " % (teamSizes[i],p.nDims))
    #     plt.scatter(subTeamSize,teamScore,label='team size: '+str(teamSizes[i]))
        invY = np.array(teamScore) * -1
        m.plotCategoricalMeans(subTeamSize,invY)
        plt.xlabel("subteam size")
        plt.xscale('log',basex=2)
        plt.xticks([1,2,3,4,8,16,32])
        plt.ylabel("performance")
    #     plt.show()
    plt.legend(teamSizes)
    
    # np.savetxt("./results/C1.1_organicSpecialization/params.txt",[makeParamString()], fmt='%s')
    # scoreMatrix = [ [ [t.getBestScore() for t in eachStructure] for eachStructure in eachSize] for eachSize in teamObjectsOrganic]
    # rFile = './results/C1.1_organicSpecialization/'+'scoreMatrix.obj'
    # rPickle = open(rFile, 'wb')
    # pickle.dump(scoreMatrix, rPickle)
    plt.savefig("./results/iv_specialization_organicTeams.pdf")

