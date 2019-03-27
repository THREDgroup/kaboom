"""
Strategically assign agents to sub-teams based on their KAI styles

This recreates the third experiment from IDETC, and demonstrates that a team
can perform better if it assigns its agents to sub-teams based on their
cognitive style and the best style for each sub-team. The control group is
a team composed of random agents and assigned to random teams, and the
strategic teams are composed of random agents assigned to teams strategically.

The results are plotted and saved to /results/ folder
"""
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


#create custom teams where we get random agents but put them in best places
def createOrganicSuperteam(p,subteamsSortedByStyle,
                     agentConstructor = CarDesignerWeighted):
    p.aiScore = None
    p.aiRange = None

    myTeam = createTeam(p,agentConstructor)
    agentsList = myTeam.agents
    agentStyles = [a.kai.KAI for a in agentsList]
    agentsSortedByStyle = [a for _,a in sorted(zip(agentStyles,agentsList))]
    #[t for _,t in sorted(zip(subTeamStyles,range(len(subTeamStyles))))]
    newSubTeamAllocations = [t for t in subteamsSortedByStyle for i in range(int(p.nAgents/p.nTeams))]

    #now, re-allocate the same agents to teams suited to their style:
    for i,a in enumerate(agentsSortedByStyle):
        a.team = newSubTeamAllocations[i]

    myTeam.agents = agentsSortedByStyle

    for a in myTeam.agents:
        a.startSpeed = h.cp(a.speed)
        a.startTemp = h.cp(a.temp)
        a.myDims = p.teamDims[a.team]

    return myTeam

#Define a custom experiment for this one
def teamWorkOrganicSuperteam(processID,p,subteamsSortedByStyle):
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
    myTeam = createOrganicSuperteam(p,subteamsSortedByStyle,CarDesignerWeighted)

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

def run(numberOfCores = 4):
    t0 = timer.time()
    p=Params()

    #change team size and one sub-teams style:
    p.nAgents = 33
    p.nDims = 56
    p.steps = 100 #100
    p.reps = 16

    #organic composition: select agents randomly from population
    p.aiScore = None
    p.aiRange = None

    myPath = os.path.dirname(__file__)
    parentDir = os.path.dirname(myPath)
    paramsDF = pd.read_csv(parentDir+"/SAE/paramDBreduced.csv")
    paramsDF = paramsDF.drop(["used"],axis=1)
    paramsDF.head()

    #assign the actual specialized teams:
    teams = ['brk', 'c', 'e', 'ft', 'fw', 'ia','fsp','rsp', 'rt', 'rw', 'sw']
    paramTeams = paramsDF.team
    p.nTeams = len(teams)
    teamDimensions_semantic = [[ 1 if paramTeam == thisTeam else 0 for paramTeam in paramTeams] for thisTeam in teams]
    #teamDimensions_blind = m.specializedTeams(p.nAgents,p.nTeams)
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = teamDimensions_semantic


    #First run the control group: teams with organic composition
    if __name__ == '__main__'or 'kaboom.IDETC_studies.iii_strategicTeams':
        pool = multiprocessing.Pool(processes = numberOfCores)
        controlTeams = pool.starmap(carTeamWorkProcess,
                                zip(range(p.reps),
                                itertools.repeat(p),
                                itertools.repeat(CarDesignerWeighted)))
    #            scoresA.append([t.getBestScore() for t in allTeams])
    #            teams.append(allTeams)
        pool.close()
        pool.join()

    controlScores = [t.getBestScore()*-1 for t in controlTeams]


    #Run strategic teams
    subteamsSortedByStyle = [7, 8, 0, 10, 3, 2, 6, 4, 1, 5, 9]
#    namedSortedTeams = [teams[i] for i in subteamsSortedByStyle]


    strategicTeamObjects = []
    if __name__ == '__main__' or 'kaboom.IDETC_studies.iii_strategicTeams':
        pool = multiprocessing.Pool(processes = 4)
        allTeams = pool.starmap(teamWorkOrganicSuperteam,
                                zip(range(p.reps),
                                itertools.repeat(p),
                                itertools.repeat(subteamsSortedByStyle)))
    #            scoresA.append([t.getBestScore() for t in allTeams])
    #            teams.append(allTeams)
        for t in allTeams:
            strategicTeamObjects.append(t)
        pool.close()
        pool.join()
    print("time to complete: "+str(timer.time()-t0))

    strategicScores = [t.getBestScore()*-1 for t in strategicTeamObjects]

    plt.boxplot([np.array(controlScores), np.array(strategicScores)],labels= ["control","strategic allocation"],showfliers=True)
    plt.ylabel("car design performance")

    plt.savefig(myPath+"/results/iii_carStrategicTeamAssignment.pdf")
    plt.show()
    plt.clf()

    print("Results figure saved to "+myPath+"/results/iii_carStrategicTeamAssignment.pdf")

    print("effect size:" )
    print(h.effectSize(controlScores,strategicScores))
    print("ANOVA p score: ")
    print(h.pScore(controlScores,strategicScores))
