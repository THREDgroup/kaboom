import numpy as np
import time as timer
import multiprocessing
from matplotlib import pyplot as plt
#import pickle
import itertools

from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.kaboom import teamWorkProcess

# # A 1.5 Teams of different style

# In[22]:

def run():
    t0 = timer.time()
    p=Params()
    
    
    #change team size and specialization
    p.nAgents = 12
    p.nTeams = 4
    p.nDims = 12
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = m.teamDimensions(p.nDims,p.nTeams)
    
    pComms = np.linspace(0,1,11)
    
    p.reps = 32 # 10 #$40 #5
    
    aiScores = [60,95,130]#100#300
    p.aiRange = 0
    # aiRanges = np.linspace(0,100,10)
    
#    meetingTimes = 100 
    
    resultsA14 = []
    for aiScore in aiScores:
        p.aiScore = aiScore
        allTeamObjects = []
        for pComm in pComms:   
            if __name__ == '__main__' or 'kaboom.test.ii_comm_v_style':
                pool = multiprocessing.Pool(processes = 4)
                allTeams = pool.starmap(teamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
                print('next. time: '+str(timer.time()-t0))
                for team in allTeams:
                    allTeamObjects.append(team)
                    
                pool.close()
                pool.join()
        resultsA14.append(allTeamObjects)
        scores = [t.getBestScore() for t in allTeamObjects]
        pcs = [pc for pc in pComms for i in range(p.reps)]
        m.plotCategoricalMeans(pcs,np.array(scores)*-1)
        plt.show()
        # allTeams = [t for tl in allTeamObjects for t in tl]
    print("time to complete: "+str(timer.time()-t0))
    
    
    # In[47]:
    
    for allTeamObjects in resultsA14:
    
        allScores = np.array([t.getBestScore() for t in allTeamObjects])*-1
        kai = allTeamObjects[0].agents[0].kai.KAI
        nS = [t.nMeetings for t in allTeamObjects]
    #     plt.scatter(nS,allScores, c=[.9,.9,.9])
        pC = [pc for pc in pComms for i in range(p.reps)]
    #     plt.show()
    #     plt.scatter(pC,allScores, label=kai)
        c = m.plotCategoricalMeans(pC,allScores)
        plt.plot(pComms,c)
    #     name="A1.5_commRate_vStyle32_kai"+str(kai)
    #     directory = saveResults(allTeamObjects,name)
    plt.legend(aiScores)
    plt.xlabel('prob of communication (c)')
    plt.ylabel('performance')
    plt.savefig("./results/ii_commRate_vStyle32"+"/plot.pdf")
    #plt.savefig("./results/A1.5_commRate_vStyle32"+"/plot.png",dpi=300)

