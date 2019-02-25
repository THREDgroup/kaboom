#saving the session workspace varibles to serverTests/individualTeamStyles
import numpy as np
import time as timer
import multiprocessing
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import os

from kaboom import helperFunctions as h
from kaboom import kaiStyle as kai
from kaboom.params import Params
from kaboom import modelFunctions as m
#from kaboom.kaboom import saveResults
from kaboom.kaboom import createTeam
from kaboom.CarDesigner import CarDesignerWeighted

#This experiment addresses the question:
# Do certain sub-teams of the car problem favor a KAI style?

#WARNING: This could take a long time and a lot of computational power. 
#Consider running  on a server. Set numberOfCores to the number of parallel
#comupting nodes/cores available. The simulation can be run in parallel up to
# p.reps times (= 16 by default)

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
    
    subTeamToVary : int, identifies the sub-team that will have style modified

    Returns
    -------
    myTeam: Team object with desired style composition, ready for the model
    """
    
    #start with all mid-range agents
    p.aiScore = 95
    p.aiRange = 0

    myTeam = createTeam(p,agentConstructor)
    
    #now, we need to make one specific sub-team have a different ai composition
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

def teamWorkCustomTeams(processID,p,teamNo,kaiScore):
    """
    Run the simulation for the team, for p.steps iterations

    This is the main method used to run the simulation for teams where one 
    sub-team has a modified style. It creates a team,
    simulates problem solving with interactions and team meetings, and returns
    the Team object containing the history and results of the simulation.

    Parameters
    ----------
    
    processID : int, an ID for multiprocessing purposes 
    
    p : Params object, contains current model Parameters
    including p.nAgents, p.nTeams, p.nDims, p.AVG_SPEED, p.AVG_TEMP

    AgentConstructor : constructor class (default = Steinway)
    
    teamNo : int, identifies which sub-team's style will be varied
    
    kaiScore : int, indicates the style to give all agents on one 
    subteam (all other agents will have KAI = 95)

    Returns
    -------
    myTeam: Team object post-simulation containing simulation results & history
    including myTeam.agents (list of Agent objects)
    You can find the best score with myTeam.getBestScore()
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

   
def run(numberOfCores = 4):
    
    t0 = timer.time()
    p=Params()
    
    #change team size and one sub-teams style:
    p.nAgents = 33
    p.nDims = 56
    p.steps = 100
    p.reps = 16
    
    #set up the dimensions of the car problem
    myPath = os.path.dirname(__file__)
#    print(myPath)
    parentDir = os.path.dirname(myPath)
    paramsDF = pd.read_csv(parentDir+"/SAE/paramDBreduced.csv")
    paramsDF = paramsDF.drop(["used"],axis=1)
    
    #assign the specialized teams:
    teams = ['brk', 'c', 'e', 'ft', 'fw', 'ia','fsp','rsp', 'rt', 'rw', 'sw']
    teamsFullName = ['brakes', 'cabin', 'engine', 'front tires', 'front wing',
                     'impact attenuator','front suspension','rear suspension', 
                     'rear tires', 'rear wing', 'side wings']
    paramTeams = paramsDF.team
    p.nTeams = len(teams)
    teamDimensions_semantic = [[ 1 if paramTeam == thisTeam else 0 for paramTeam in paramTeams] for thisTeam in teams]
    p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
    p.teamDims = teamDimensions_semantic
       
    
    
    aiScores = [45,70,95,120,145] #set of KAI scores to test each subteam with
    
    eachSubteamVaried = []
    for teamNo in range(len(teams)):
        compareStyles = []
        for aiScore in aiScores:
            if __name__ == '__main__' or 'kaboom.IDETC_studies.iii_subteamStyle':
                pool = multiprocessing.Pool(processes = numberOfCores)
                allTeams = pool.starmap(teamWorkCustomTeams, 
                                        zip(range(p.reps),
                                        itertools.repeat(p),
                                        itertools.repeat(teamNo),
                                        itertools.repeat(aiScore)))
                pool.close()
                pool.join()
            print("time to complete: "+str(timer.time()-t0))
            compareStyles.append(allTeams)
        eachSubteamVaried.append(compareStyles)
        
    #TODO: fix results-saving mechanism
    #Save results to ./results/
#    name="customStyleForTeams"
#    flatResults = np.ndarray.flatten(np.array(eachSubteamVaried))
#    directory = saveResults(flatResults,p,name)
    
    #plot results:
    for i,subTeamResults in enumerate(eachSubteamVaried):
        subteam_varied = teamsFullName[i]
        #the scores are inverted *-1 so that it's maximization not minimization
        #(higher scores are better in the plot)
        scores = [t.getBestScore()*-1 for oneStyle in subTeamResults for t in oneStyle]
        stylesForEachScore = [kai for kai in aiScores for i in range(p.reps)]
        plt.scatter(stylesForEachScore,scores, c=[.9,.9,.9])
        m.plotCategoricalMeans(stylesForEachScore,scores)
        
        quadraticFit = np.polyfit(stylesForEachScore,scores, 2)
        quadraticModel = np.poly1d(quadraticFit)
        x = np.linspace(45,145,101)
        plt.plot(x, quadraticModel(x),c='red')
        
        plt.title("Varying the style of the %s subteam" % subteam_varied)
        plt.savefig(myPath+"/results/subteam_plots/subteamStyle_%s.pdf" % subteam_varied)
        plt.show()
        plt.clf()
