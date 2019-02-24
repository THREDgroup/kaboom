#saving the session workspace varibles to serverTests/individualTeamStyles
import numpy as np
import time as timer
import multiprocessing
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import itertools
import os

from kaboom import helperFunctions as h
from kaboom import kaiStyle as kai
from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.carMakers import carTeamWorkProcess
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


#create custom teams where all are mid-range except one team varies:
def createSuperteam(p,subTeamStyles,
                     agentConstructor = CarDesignerWeighted):
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
        subTeamKAI = subTeamStyles[a.team]
        a.kai = kai.findAiScore(subTeamKAI)
        a.speed = kai.calcAgentSpeed(a.kai.KAI,p)
        a.temp = kai.calcAgentTemp(a.kai.E,p)
        a.decay = kai.calculateAgentDecay(a,p.steps)

    for a in myTeam.agents:
        a.startSpeed = h.cp(a.speed)
        a.startTemp = h.cp(a.temp)

    return myTeam


#create custom teams where we get random agents but put them in best places
def createOrganicSuperteam(p,subTeamStyles,
                     agentConstructor = CarDesignerWeighted):
    p.aiScore = None
    p.aiRange = None

    myTeam = createTeam(p,agentConstructor)
    agentsList = myTeam.agents
    agentStyles = [a.kai.KAI for a in agentsList]
    agentsSortedByStyle = [a for _,a in sorted(zip(agentStyles,agentsList))]
    subteamsSortedByStyle = [t for _,t in sorted(zip(subTeamStyles,range(len(subTeamStyles))))]
    newSubTeamAllocations = [t for t in subteamsSortedByStyle for i in range(int(p.nAgents/p.nTeams))]

    #now, re-allocate the same agents to teams suited to their style:
    for i,a in enumerate(agentsSortedByStyle):
        a.team = newSubTeamAllocations[i]

    myTeam.agents = agentsSortedByStyle
    
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

#Define a custom experiment for this one
def teamWorkSuperteam(processID,p,subTeamStyles):
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
    myTeam = createSuperteam(p,subTeamStyles,CarDesignerWeighted)

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


#Define a custom experiment for this one
def teamWorkOrganicSuperteam(processID,p,subTeamStyles):
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
    myTeam = createOrganicSuperteam(p,subTeamStyles,CarDesignerWeighted)

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
p.reps = 4


myPath = os.path.dirname(__file__)
parentDir = os.path.dirname(myPath)
paramsDF = pd.read_csv(parentDir+"/kaboom/SAE/paramDBreduced.csv")
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
   

#RUN SPECIALIED SUBTEAMS

aiScores = [45,70,120,145]
#ts = []
#for k in aiScores:
#    ts.append(teamWorkCustomTeams(0,p,0,k))
#
##request @ 32 nodes 4gb: 800 sec per [100 step simulation of 32 agents x 32 reps ]
##request: 800 sec * 7*3 = 4.7 hours -> request 6 hrs

#allSubTeams = []
#for teamNo in []:#range(len(teams)):
#    compareStyles = []
#    for aiScore in aiScores:
#        teamObjects = []
#        if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
#            pool = multiprocessing.Pool(processes = 16)
#            allTeams = pool.starmap(teamWorkCustomTeams, 
#                                    zip(range(p.reps),
#                                    itertools.repeat(p),
#                                    itertools.repeat(teamNo),
#                                    itertools.repeat(aiScore)))
#        #            scoresA.append([t.getBestScore() for t in allTeams])
#        #            teams.append(allTeams)
#            for t in allTeams:
#                teamObjects.append(t)
#            pool.close()
#            pool.join()
#        print("time to complete: "+str(timer.time()-t0))
#        compareStyles.append(teamObjects)
#    allSubTeams.append(compareStyles)
#    
##SAVE!
#name="customStyleForTeams"
#flatResults = np.ndarray.flatten(np.array(allSubTeams))
#directory = saveResults(flatResults,p,name)
#for i,r in enumerate(flatResults):
#    rFile = myPath+"/styleResults/result"+str(i)+"_"+str(timer.time())+".obj"
#    rPickle = open(rFile, 'wb')
#    pickle.dump(r,rPickle) 
#    
    
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
controlScores = [-43697.83046474878,
 -53539.600755698295,
 -51475.71127076317,
 -42580.22982036467,
 -55799.1550135683,
 -48337.7879095872,
 -53567.45303120078,
 -45038.94097887598,
 -54929.55670206938,
 -45854.197285548806,
 -53866.196861206874,
 -56265.54094660408,
 -55839.239946908034,
 -50304.24237876096,
 -46157.816942948375,
 -41412.904269363855]
#        
#
#for i in range(len(teams)):
#    teami = allSubTeams[i]
#    team0Ascores = np.mean([t.getBestScore() for t in teami[0]])
#    team0Iscores = np.mean([t.getBestScore() for t in teami[1]])
#    plt.scatter([i for j in range(1)],team0Ascores,color='b')
#    plt.scatter([i for j in range(1)],team0Iscores,color='r')
#    
    
#    
###plot kai vs performance for one subteam:
##from matplotlib import pyplot as plt
#scores = [t.getBestScore() for style in team5 for t in style]
#ais = [t.agents[5].kai.KAI for style in team5 for t in style]
#m.plotCategoricalMeans(ais,scores)
##plt.scatter([95 for i in range(len(controlScores))],controlScores)
##plt.scatter(ais,scores)
#plt.show()
#
#scores = [t.getBestScore() for style in team6 for t in style]
#ais = [t.agents[6].kai.KAI for style in team6 for t in style]
##plt.scatter([95 for i in range(len(controlScores))],controlScores)
#m.plotCategoricalMeans(ais,scores)
##plt.scatter(ais,scores)
#plt.show()
#
#scores = [t.getBestScore() for style in team7 for t in style]
#ais = [t.agents[7].kai.KAI for style in team7 for t in style]
##plt.scatter([95 for i in range(len(controlScores))],controlScores)
#m.plotCategoricalMeans(ais,scores)
##plt.scatter(ais,scores)
#plt.show()
#
#
#scores = [t.getBestScore() for style in team9 for t in style]
#ais = [t.agents[9].kai.KAI for style in team9 for t in style]
##plt.scatter([95 for i in range(len(controlScores))],controlScores)
#m.plotCategoricalMeans(ais,scores)
##plt.scatter(ais,scores)
#plt.legend([5,6,7,9])




#RUN SUPERTEAM
subTeamStyles = [145.0, 145.0, 110.0, 145.0, 139.0, 127.0, 145.0, 45.0, 145.0, 145.0, 145.0]


#ts = []
#for k in aiScores:
#    ts.append(teamWorkCustomTeams(0,p,0,k))
#
p.reps = 16
p.steps = 100

superTeamObjects2 = []
if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
    pool = multiprocessing.Pool(processes = 4)
    allTeams = pool.starmap(teamWorkSuperteam, 
                            zip(range(p.reps),
                            itertools.repeat(p),
                            itertools.repeat(subTeamStyles)))
#            scoresA.append([t.getBestScore() for t in allTeams])
#            teams.append(allTeams)
    for t in allTeams:
        superTeamObjects2.append(t)
    pool.close()
    pool.join()
print("time to complete: "+str(timer.time()-t0))


superTeamScores2 = [t.getBestScore() for t in superTeamObjects2]
plt.scatter([1 for i in superTeamScores2],superTeamScores2)
#plt.scatter([1],)
plt.scatter([0 for i in controlResults2],controlResults2)


box=plt.boxplot([np.array(allHomogeneousScores)*-1, np.array(controlScores)*-1, np.array(superTeamScores2)*-1],labels= ["homogeneous","organic","superteam"],showfliers=False)
#plt.savefig("/Users/samlapp/SAE_ABM/figs/carSuperteamsPerformance1.pdf")

#plt.xticks()
h.pScore(allHomogeneousScores,superTeamScores2)
effectSize(controlScores,superTeamScores2)


#plt.ylim([-60000,-40000])
#
#
#controlTeams = []
#if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
#    pool = multiprocessing.Pool(processes = 4)
#    allTeams = pool.starmap(carTeamWorkProcess, 
#                            zip(range(p.reps),
#                            itertools.repeat(p),
#                            itertools.repeat(CarDesignerWeighted)))
##            scoresA.append([t.getBestScore() for t in allTeams])
##            teams.append(allTeams)
#    for t in allTeams:
#        controlTeams.append(t)
#    pool.close()
#    pool.join()
#print("time to complete: "+str(timer.time()-t0))
#
#
#controlResults2 = [t.getBestScore() for t in controlTeams]
#
##SAVE!
#name="customStyleForTeams"
#flatResults = np.ndarray.flatten(np.array(allSubTeams))
#directory = saveResults(flatResults,p,name)
#for i,r in enumerate(flatResults):
#    rFile = myPath+"/styleResults/result"+str(i)+"_"+str(timer.time())+".obj"
#    rPickle = open(rFile, 'wb')
#    pickle.dump(r,rPickle) 

#SuperTeamScores2:
#-46461.027450347676,
# -54519.676205915675,
# -52740.81354906518,
# -58769.53119480033,
# -54393.41597929167,
# -56644.15184100017,
# -51969.893963338036,
# -57028.36142510394,
# -59086.515442166754,
# -56617.530503163645,
# -55923.171807895764,
# -51438.02773070671,
# -57004.68240556857,
# -53604.21190104142,
# -58087.478110760756,
# -56914.52937878722]


#%% 


#RUN ORGANIC SUPERTEAM
subTeamStyles = [145.0, 145.0, 110.0, 145.0, 139.0, 127.0, 145.0, 45.0, 145.0, 145.0, 145.0]

p.reps = 4
p.steps = 100

organicSuperTeamObjects = []
if __name__ == '__main__':# or 'kaboom.test.viii_specialization_composition':
    pool = multiprocessing.Pool(processes = 4)
    allTeams = pool.starmap(teamWorkSuperteam, 
                            zip(range(p.reps),
                            itertools.repeat(p),
                            itertools.repeat(subTeamStyles)))
#            scoresA.append([t.getBestScore() for t in allTeams])
#            teams.append(allTeams)
    for t in allTeams:
        organicSuperTeamObjects.append(t)
    pool.close()
    pool.join()
print("time to complete: "+str(timer.time()-t0))


organicSTscores = [t.getBestScore() for t in superTeamObjects2]
plt.scatter([1 for i in organicSTscores],organicSTscores)
#plt.scatter([1],)
#plt.scatter([0 for i in controlResults2],controlResults2)


box=plt.boxplot([np.array(controlScores)*-1, np.array(organicSTscores)*-1],labels= ["organic","organic superteam"],showfliers=True)
#plt.savefig("/Users/samlapp/SAE_ABM/figs/carSuperteamsPerformance1.pdf")

#plt.xticks()
#h.pScore(allHomogeneousScores,superTeamScores2)
print("effect size:" )
print(effectSize(controlScores,organicSTscores))
