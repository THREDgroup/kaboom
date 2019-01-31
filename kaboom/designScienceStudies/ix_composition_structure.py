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
    
    # A 1.6 composition vs commRate
    
    p= Params()
    
    p.nAgents = 20
    p.nTeams = 4
    p.nDims = 20
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = m.teamDimensions(p.nDims,p.nTeams) #np.ones([nTeams,nDims])
    #p.reps=1
    pComms = np.linspace(0,1,11)
    
    p.aiScore = 95
    #meetingTimes = 100 
    t0 = timer.time()
    
    resultsA16 = []
    for i in range(3):
        if i == 0: #homogeneous
            p.aiScore = 95
            p.aiRange = 0
            p.curatedTeams = False
        elif i == 1: #hetero70
            p.aiScore = 95
            p.aiRange = 70
            p.curatedTeams = True
        elif i == 2: #organic
            p.aiScore = None
            p.aiRange = None
            p.curatedTeams = False
    
        allTeamObjects = []
        for pComm in pComms:  
            p.pComm = pComm
            if __name__ == '__main__' or'kaboom.test.ix_composition_structure':
                pool = multiprocessing.Pool(processes = 4)
                allTeams = pool.starmap(teamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
                print('next. time: '+str(timer.time()-t0))
                for team in allTeams:
                    allTeamObjects.append(team)
                    
                pool.close()
                pool.join()
        resultsA16.append(allTeamObjects)
        scores = [t.getBestScore() for t in allTeamObjects]
        pcs = [pc for pc in pComms for i in range(p.reps)]
        m.plotCategoricalMeans(pcs,np.array(scores)*-1)
        plt.show()
        # allTeams = [t for tl in allTeamObjects for t in tl]
    print("time to complete: "+str(timer.time()-t0))
    
    
    comps = ['homogeneous','heterogeneous70','organic']
    for i in range(3):
        allTeamObjects = resultsA16[i]
        
        allScores = np.array([t.getBestScore() for t in allTeamObjects])*-1
        
        nS = [t.nMeetings for t in allTeamObjects]
    #     plt.scatter(nS,allScores, c=[.9,.9,.9])
        pC = [pc for pc in pComms for i in range(p.reps)]
    #     plt.show()
    #     plt.scatter(pC,allScores, label=kai)
        c = m.plotCategoricalMeans(pC,allScores)
        
    #    name="A1.6_commRate_vStyle_"+comps[i]
    #     directory = saveResults(allTeamObjects,name)
    plt.legend(comps)
    plt.savefig("./results/ix_composition_specialization.pdf")
