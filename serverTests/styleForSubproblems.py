import numpy as np
import time as timer
import multiprocessing
import pandas as pd
#from matplotlib import pyplot as plt
import pickle
import itertools
import os

from kaboom import helperFunctions as h
from kaboom import kaiStyle as kai
from kaboom.params import Params
from kaboom import modelFunctions as m
#from kaboom.carMakers import carTeamWorkProcess
from kaboom.kaboom import saveResults
from kaboom.kaboom import createTeam
from kaboom.CarDesigner import CarDesignerWeighted

#create custom teams where all are mid-range except one team varies:
def createCustomTeam(p,
                     agentConstructor = CarDesignerWeighted,
                     subTeamToVary = 0,
                     subTeamKAI = 95):
    """
    Create a team of agents before running it through the simulation.

    Design the team with a specific composition of KAI scores,
    and subdivision of a problem into specialized subteams

    Parameters
    ----------
    p : Params object, contains current model Parameters
    including p.nAgents, p.nTeams, p.nDims, p.AVG_SPEED, p.AVG_TEMP

    AgentConstructor : constructor class (default = Steinway)

    Returns
    -------
    myTeam: Team object with desired characteristics, ready to run in the model
    """
    p.aiScore = 95
    p.aiRange = 0

    myTeam = createTeam(p,agentConstructor)
    
    #now, we need to make a specific sub-team have a different ai composition
    for i, a in enumerate(myTeam.agents):
        if a.team == subTeamToVary:
            a.kai = kai.findAiScore(subTeamKAI)
            a.speed = kai.calcAgentSpeed(a.kai.KAI,p)
            a.temp = kai.calcAgentTemp(a.kai.E,p)
            a.decay = kai.calculateAgentDecay(a,p.steps)

    for a in myTeam.agents:
        a.startSpeed = h.cp(a.speed)
        a.startTemp = h.cp(a.temp)

    return myTeam

#Define a custom experiment for this one
def teamWorkCustomTeams(processID,p,teamNo,kaiScore):
    """
    Run the simulation for the team, for p.steps iterations

    This is the main method used to run the simulation. It creates a team,
    simulates problem solving with interactions and team meetings, and returns
    the Team object containing the history and results of the simulation.

    Parameters
    ----------
    p : Params object, contains current model Parameters
    including p.nAgents, p.nTeams, p.nDims, p.AVG_SPEED, p.AVG_TEMP

    AgentConstructor : constructor class (default = Steinway)

    Returns
    -------
    myTeam: Team object post-simulation containing simulation results & history
    including myTeam.agents (list of Agent objects)
    find the best score with myTeam.getBestScore()
    """
    np.random.seed()
    myTeam = createCustomTeam(p,CarDesignerWeighted,teamNo,kaiScore)

    i = 0 #not for loop bc we need to increment custom ammounts inside loop
    while i < p.steps:
        myTeam.nMeetings += myTeam.step(p)
        score = myTeam.getBestCurrentScore() #getBestCurrentScore
        myTeam.scoreHistory.append(score)
        if (i+1)%p.meetingTimes == 0:
            cost = myTeam.haveInterTeamMeeting(p)
            i += cost #TEAM_MEETING_COST
        i += 1

    return myTeam



# Do certain sub-teams have style preference adaptive/innovative?

t0 = timer.time()
p=Params()

#change team size and one sub-teams style:
p.nAgents = 33
p.nDims = 56
p.steps = 100 #100
p.reps = 4#16


myPath = os.path.dirname(__file__)
paramsDF = pd.read_csv("../SAE/paramDBreduced.csv")
paramsDF = paramsDF.drop(["used"],axis=1)
paramsDF.head()

#assign the actual specialized teams:
teams = ['brk', 'c', 'e', 'ft', 'fw', 'ia','fsp','rsp', 'rt', 'rw', 'sw']
teamsDict = { i:teams[i] for i in range(10)}
paramTeams = paramsDF.team
p.nTeams = len(teams)
teamDimensions_semantic = [[ 1 if paramTeam == thisTeam else 0 for paramTeam in paramTeams] for thisTeam in teams]
#teamDimensions_blind = m.specializedTeams(p.nAgents,p.nTeams)
p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
p.teamDims = teamDimensions_semantic
   

aiScores = [45,135]
#ts = []
#for k in aiScores:
#    ts.append(teamWorkCustomTeams(0,p,0,k))
#
##request @ 32 nodes 4gb: 800 sec per [100 step simulation of 32 agents x 32 reps ]
##request: 800 sec * 7*3 = 4.7 hours -> request 6 hrs
allSubTeams = []
for teamNo in range(len(teams)):
    compareStyles = []
    for aiScore in aiScores:
        teamObjects = []
        if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
            pool = multiprocessing.Pool(processes = 4)
            allTeams = pool.starmap(teamWorkCustomTeams, 
                                    zip(range(p.reps),
                                    itertools.repeat(p),
                                    itertools.repeat(teamNo),
                                    itertools.repeat(aiScore)))
        #            scoresA.append([t.getBestScore() for t in allTeams])
        #            teams.append(allTeams)
            for t in allTeams:
                teamObjects.append(t)
            pool.close()
            pool.join()
        print("time to complete: "+str(timer.time()-t0))
        compareStyles.append(teamObjects)
    allSubTeams.append(compareStyles)
    
#SAVE!
name="customStyleForTeams"
flatResults = np.ndarray.flatten(np.array(allSubTeams))
directory = saveResults(flatResults,p,name)
for i,r in enumerate(flatResults):
    rFile = myPath+"/styleResults/result"+str(i)+"_"+str(timer.time())+".obj"
    rPickle = open(rFile, 'wb')
    pickle.dump(r,rPickle)
    
#
#for i in range(len(teams)):
#    teami = allSubTeams[i]
#    team0Ascores = [t.getBestScore() for t in teami[0]]
#    team0Iscores = [t.getBestScore() for t in teami[1]]
#    plt.scatter([i for j in range(p.reps)],team0Ascores,color='b')
#    plt.scatter([i for j in range(p.reps)],team0Iscores,color='r')
#    pS = h.pScore(team0Ascores,team0Iscores)
#    print(pS)
#    if pS<.05:
#        print("scores are significantly different for team "+teams[i]+":")
#        print(str(np.mean(team0Ascores)) +" vs "+str(np.mean(team0Iscores)))
#        
#plt.plot([0,10],[-49916,-49916],'purple')
#plt.scatter(np.ones(16)*11,[-43697.83046474878,
# -53539.600755698295,
# -51475.71127076317,
# -42580.22982036467,
# -55799.1550135683,
# -48337.7879095872,
# -53567.45303120078,
# -45038.94097887598,
# -54929.55670206938,
# -45854.197285548806,
# -53866.196861206874,
# -56265.54094660408,
# -55839.239946908034,
# -50304.24237876096,
# -46157.816942948375,
# -41412.904269363855], c='purple')
#        
#
#for i in range(len(teams)):
#    teami = allSubTeams[i]
#    team0Ascores = np.mean([t.getBestScore() for t in teami[0]])
#    team0Iscores = np.mean([t.getBestScore() for t in teami[1]])
#    plt.scatter([i for j in range(1)],team0Ascores,color='b')
#    plt.scatter([i for j in range(1)],team0Iscores,color='r')
#    