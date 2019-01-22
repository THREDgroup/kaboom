#Functions implementing logic for the KABOOM model

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi
from kaboom import helperFunctions as h

#determines whether an agent will move to a worse solution or keep current solution
#according to simulated annealing algorithm, accept worse solution with probability: exp(difference in quality / temp)
def pickWorseScore(betterScore,worseScore,temperature):
    if temperature <=1E-6: #at low temp: never pick worse answers, and avoid devide by 0
        return False
    if np.random.uniform(0,1) < np.exp((betterScore-worseScore)/temperature):  #
        return True
    return False

#use matplotlib.pyplot to plot means and standard deviations for each distinct x value
#returns list of mean values for each distinct x value
def plotCategoricalMeans(x,y):
    categories = np.unique(x)
    means = []
    sds = []
    for c in categories:
        yc = [y[i] for i in range(len(y)) if x[i] == c]
        means.append(np.mean(yc))
        sds.append(np.std(yc))
    plt.errorbar(categories,means,yerr=sds,marker='o',ls='none')

    return means

#determine how far an agent moves in a given step
# multiplies a value drawn from chi distribution by [speed], returns travel distance
dfConstant=1.9
def travelDistance(speed): #how far do we go? chi distribution, but at least go 0.1 * speed
    r = np.max([chi.rvs(dfConstant),0.1])
    return r * speed

#create weights for calculating weighted mean memory of agent's past solutions
#the weights on memories follows the serial position effect (stronger at beginning and end than the middle)
def memoryWeightsPrimacy(n):
    if n==1:
        return np.array([1])
    weights = np.arange(n-1,-1,-1)**3*0.4 + np.arange(0,n,1)**3
    weights = weights / np.sum(weights)
    return weights

#plot the biased memory weights:
#plt.plot(np.linspace(0,1,100),memoryWeightsPrimacy(100))
# plt.savefig('./serialPosition.pdf')

#assign [nAgents] agents to [nTeams] subteams
#returns a vector of length [nAgents] with values [0:nTeams-1] mapping agents to teams
def specializedTeams(nAgents,nTeams):
    agentTeams = np.array([a%nTeams for a in range(nAgents)])
    return agentTeams

#allocate [nDims] dimensions of a problem to [nTeams] specialized sub-nTeams
def teamDimensions(nDims,nTeams):
    teamDims = np.array([[1 if t%nTeams == dim%nTeams else 0 for dim in range(nDims)] for t in range(nTeams)])
    return teamDims

# Objective Function: composite of sinusoid and parabola
def objectiveTune(x,roughness=.05,w=100):
    a = roughness
    b = 2
    x = np.array(x)
    xEach = -1*a*np.sin((x*w+np.pi/2)) + b*(x*1.5)**2 + roughness #min is zero at zero vector
    y = sum(xEach)
    return y

#### sharing solutions and pairwise communcation ####

#when agents communicate, they attempt to share solutions with eachother
#returns True if communication succeeds, else False
def tryToShare(a1,a2,p):
    deltaAi = abs(a1.kai.KAI - a2.kai.KAI) #harder to communicate above 20, easy below 10
#    deltaR = np.linalg.norm(a1.r - a2.r)
    successful =  tryComm(deltaAi,p)#deltaR)
    if successful: #in share(), agents might adopt a better solution depending on their temperature
        share(a1,a2,p.selfBias)
        return True
    return False

#likelihood of communication success depends on cognitive gap [deltaAi]
#returns True if communication succeeds, else False
def tryComm(deltaAi,p):
    c = np.random.uniform(p.commBonus,p.commBonus+p.commRange) #increasing commBonus makes sharing easier
    return (deltaAi < c)

#when sharing is successful, each agent considers the others' solution
def share(a1,a2,selfBias=0): #each agent chooses whether to accept shared solution or not
    copyOfA1 = h.cp(a1)
    considerSharedSoln(a1,a2,selfBias)
    considerSharedSoln(a2,copyOfA1,selfBias)
    return True

#given another agent [sharer]'s solution, this agent [me] decides whether to accept or reject it
def considerSharedSoln(me,sharer,selfBias=0):
        candidateSolution = sharer.r
        candidateScore = me.fr(candidateSolution)
        myScore = me.score - selfBias #improve my score by selfBias (currently set to zero)
        #Quality Bias Reduction would go here, if implemented
        if(candidateScore<myScore):
            if not pickWorseScore(candidateScore,myScore,me.temp): #sometimes choose better, not always
                me.moveTo(candidateSolution)  #(but never take a worse score from a teammate)
