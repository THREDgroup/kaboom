import numpy as np
import scipy
import time as timer
import multiprocessing
from matplotlib import pyplot as plt
#import pickle
import itertools

from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.kaboom import teamWorkProcess


def run():
    
    # # A 3.1
    # ## Frequency of team meetings
    
        
    t0 = timer.time()
    p=Params()
    
    p.nAgents = 20
    p.nTeams = 4
    p.nDims = 20
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = m.teamDimensions(p.nDims,p.nTeams) #np.ones([nTeams,nDims])
    
    p.curatedTeams = True
    
    #for test
    #p.reps=1
    #p.steps = 10
    
    #    pComm = 0.2
    # aiScores = np.linspace(50,150,5)
    #    aiScore = None
    # aiRanges = np.linspace(0,70,5)
    
    varyMeetingTimes = [int(p.steps/i) for i in range(1,10)]
    varyMeetingTimes.append(100000)
    # starts = np.linspace(0,1,5)
    
    meetingStartTimes = []
    for meetingTimes in varyMeetingTimes:   
        p.meetingTimes=meetingTimes
        if __name__ == '__main__' or 'kaboom.test.iii_teamMeetings':
            pool = multiprocessing.Pool(processes = 4)
            allTeams = pool.starmap(teamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
            pool.close()
            pool.join()
            print('next')
        meetingStartTimes.append(allTeams)
    print("time to complete: "+str(timer.time()-t0))
    # chaching()
    
    
    
    allTeams = [ t for tt in meetingStartTimes for t in tt]
    #directory = m.saveResults(allTeams,'numberOfTeamMeetings')
        
    
    teamScores = [t.getBestScore() for t in allTeams]
    teamScoresGrouped = [[t.getBestScore() for t in tt] for tt in meetingStartTimes ]
    
    nMeetings = [t.nTeamMeetings for tt in meetingStartTimes for t in tt]
    
    perf = np.array(teamScores)*-1
    plt.scatter(nMeetings,perf, c=[.9,.9,.9])
    means = m.plotCategoricalMeans(nMeetings,perf)
    plt.xlabel("number of team meetings")
    plt.ylabel("performance")
    # print(pScore(teamScoresGrouped[0],teamScoresGrouped[3]))
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(nMeetings,teamScores)
    x = np.linspace(0,10,10)
    plt.plot(x,(x*slope+intercept)*-1)
    p_value
    plt.savefig("./results/iii_team_meetings_"+str(timer.time())+".pdf")
