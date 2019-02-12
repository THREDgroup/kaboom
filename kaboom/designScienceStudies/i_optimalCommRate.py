"""
Simulate team performance for different communication rates, and plot.

This script recreates the results shown in Figure 10 of [1]. It demonstrates the
curvilinear tradeoff between pairwise communication frequency and performance:
communication improves performance up to a point, then decreases performance.


[1] Lapp, S., Jablokow, J., McComb, C. (2019). "KABOOM: An Agent-Based Model for Simulating Cognitive Style in Team Problem Solving". Unpulished manuscript.
"""
import numpy as np
import time as timer
import multiprocessing
from matplotlib import pyplot as plt
import os
import itertools

from kaboom import modelFunctions as m
from kaboom.params import Params
from kaboom.kaboom import teamWorkProcess


# # A 1.1
# Test team performance of a homogeneous mid-range team
#for different pair-wise communication rates
# results demonstrate a tradeoff where optimal communication is ~0.4
def run():
    t0 = timer.time()
    p=Params()


    #change team size and specialization
    p.nAgents = 12
    p.nTeams = 4
    p.nDims = 12
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = m.teamDimensions(p.nDims,p.nTeams)

    pComms = np.linspace(0,1,6)

    roughnesses = np.logspace(-1,.7,num=6,base=10)
    p.amplitude = roughnesses[3]
    speeds = np.logspace(-1,.7,num=6,base=10) / 100
    p.AVG_SPEED = speeds[3]

    allTeamObjects = []
    for pComm in pComms:
        if __name__ == '__main__' or 'kaboom.designScienceStudies.i_optimalCommRate':
            pool = multiprocessing.Pool(processes = 4)
            allTeams = pool.starmap(teamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
            print('next. time: '+str(timer.time()-t0))
            for team in allTeams:
                allTeamObjects.append(team)
            pool.close()
            pool.join()
            print('finished one communication frequency')
        else:
            print( __name__)
    # allTeams = [t for tl in allTeamObjects for t in tl]
    print("time to complete: "+str(timer.time()-t0))

    #save results object to file
    # name="sharingRate"
    # directory = saveResults(allTeamObjects,name)

    r = allTeamObjects
    pC = [prob for prob in pComms for i in range(p.reps)]
    perf = np.array([t.getBestScore() for t in r])*-1
    nMeetings = [t.nMeetings for t in r]
    plt.scatter(pC,perf,c=[.9,.9,.9])
    m.plotCategoricalMeans(pC,perf)
    plt.xlabel('prob of communication (c)')
    plt.ylabel('performance')
    myPath = os.path.dirname(__file__)
    plt.savefig(myPath+"/results/i_communicationFrequency.pdf")
    plt.show()

    #alternatively, plot performance vs the actual number of pairwise interactions
    plt.scatter(nMeetings, perf)
#    plt.savefig(myPath +'/results/vsNmeetings.pdf')
