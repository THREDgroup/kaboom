"""
This module contains the methods for implementing KABOOM model logic.

It includes functions to implement simulated annealing as a
transition from stochastic to deterministic search; sharing of solutions
between agents; dividing a problem into sub-problems and assigning agents to
sub-teams; plotting results; and implementing biases for memory weights.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi

from kaboom import helperFunctions as h

def pickWorseScore(betterScore,worseScore,temperature):
    """
    Determine if an agent moves to a worse solution or keeps current solution.

    An agent considers whether to keep their current solution or accept an
    inferior candidate solution.
    According to the simulated annealing algorithm, the probability of accepting
    an inferior solution should decline throughout the course of the simulation
    so that the beginning is stochastic with a gradual transition to greedy /
    downhill / deterministic search. The probability of accepting a worse
    solution is inspired by the thermodynamic exponential form:
    P = exp(difference in quality / temp).

    Parameters
    ----------
    betterScore : float, quality of current solution
    worseScore : float, quality of candidate solution
    temperature : float, parameter of simulated annealing scheme (Agent.temp)

    Returns
    -------
    returns True if agent accepts candidate solution, else False
    """
    if temperature <=1E-6: #at low temp: never pick worse answers (avoid /0 )
        return False
    relativeQuality = np.exp((betterScore-worseScore)/temperature)
#    print(betterScore)
#    print(worseScore)
#    print(temperature)
    if relativeQuality >= 1:
        return True #avoid errors with inf value
    if np.random.uniform(0,1) < relativeQuality:  #
        return True
    return False

#use matplotlib.pyplot to
#
def plotCategoricalMeans(x,y):
    """
    Plot means and standard deviations for each distinct x value

    An agent considers whether to keep their current solution or accept an
    inferior candidate solution.
    According to the simulated annealing algorithm, the probability of accepting
    an inferior solution should decline throughout the course of the simulation
    so that the beginning is stochastic with a gradual transition to greedy /
    downhill / deterministic search. The probability of accepting a worse
    solution is inspired by the thermodynamic exponential form:
    P = exp(difference in quality / temp).

    Parameters
    ----------
    x : list of float, independent variable
    y : list of float, dependent variable
    WARNING: Requires len(y) = len(x)

    Returns
    -------
    means : list of floats, shape = [len(unique(x))]
    List of mean values of y for each distinct x value
    """
    categories = np.unique(x)
    means = []
    sds = []
    for c in categories:
        yc = [y[i] for i in range(len(y)) if x[i] == c]
        means.append(np.mean(yc))
        sds.append(np.std(yc))
    plt.errorbar(categories,means,yerr=sds,marker='o',ls='none')

    return means

#degrees of freedom constant (shape parameter for chi distribution)
dfConstant=1.9
def travelDistance(speed):
    """
    Determine how far an agent moves in a given step

    Draw from a chi distribution, then multiply by Agent's speed
    Minimum travel distance is 0.1 * speed

    Parameters
    ----------
    speed : float, Agent's current speed aka step size (Agent.speed)

    Returns
    -------
    distanceToTravel : float
    """
    r = np.max([chi.rvs(dfConstant),0.1])
    distanceToTravel = r * speed
    return distanceToTravel

#create weights for calculating weighted mean memory of agent's past solutions
#the weights on memories follows the serial position effect (stronger at beginning and end than the middle)
def memoryWeightsPrimacy(n):
    """
    Create weights for finding weighted avg. memory of agent's past solutions.

    The weights follow a convex curve reflecting the primacy and recency
    cognitive biases of memory, which note that people recall the first and most
    recent memories better than intermediate memories. These weights are then
    used in calculating the agent's percieved position of all past solutions
    by taking a weighted-average. The weights are normalized to sum to 1.

    Parameters
    ----------
    n : int, number of memories = number of weights to create

    Returns
    -------
    weights : np.array of float, shape = [n]
    """
    if n==1:
        return np.array([1])
    weights = np.arange(n-1,-1,-1)**3*0.4 + np.arange(0,n,1)**3
    weights = weights / np.sum(weights)
    return weights

#plot the biased memory weights:
#plt.plot(np.linspace(0,1,100),memoryWeightsPrimacy(100))
# plt.savefig('./serialPosition.pdf')

def specializedTeams(nAgents,nTeams):
    """
    Assign [nAgents] agents to [nTeams] sub-teams.

    Agents are divided into specialized sub-teams. The sub-teams have equal
    size up to a remainder (eg, 3 agents on 2 subteams make teams of 2 and 1)

    Parameters
    ----------
    nAgents : int, number of agents on entire team
    nTeams : int, number of sub-teams to create

    Returns
    -------
    agentTeams : np.array of int, shape = [nAgents], where each entry gives
    the team number (zero indexed) each agent is assigned to.
    """
    agentTeams = np.array([a%nTeams for a in range(nAgents)])
    return agentTeams

def teamDimensions(nDims,nTeams):
    """
    allocate [nDims] dimensions of a problem to [nTeams] specialized sub-teams

    Parameters
    ----------
    nDims : int, number of dimensions in the objective function
    nTeams : int, number of sub-teams to create

    Returns
    -------
    teamDims : np.array of 0 and 1, shape = [nTeams, nDims]. Each teamDims[i,:]
    is a row with 1 for dimensions owned by team_i and 0 for other dimensions.
    """
    teamDims = np.array([[1 if t%nTeams == dim%nTeams else 0 for dim in range(nDims)] for t in range(nTeams)])
    return teamDims

def objectiveTune(x,a=.05,w=100):
    """
    An objective function that has a composite sinusoid and parabola

    This is the objective function used in the Design Science journal paper.
    It is symmetric in all [nDims = len(x)] dimensions of the problem.
    The sinusoidal amplitude (alpha in the paper) is called [a] here. The
    beta parameter for scaling the solution space size is absorbed into
    the parameter AVG_SPEED  which scales agents' step sizes,
    rather than being implemented here.

    When a is small (<.1), the function becomes more parabolic,
    but when a is large (>1) the relative importance of the parabolic term is
    small compared to the sinusoidal term. This leads to differences in the
    cognitive style that performs best for different problemsself.

    Parameters
    ----------
    x : list of float, size = [nDims] the solution to be evaluated

    Returns
    -------
    y : float, value of the objective function = performance
    NOTE: the problem is defined as a MINIMIZATION problem here, so that
    lower scores are better. In the paper all scores are multiplied by -1 to
    create a maximization problem where higher scores are better.
    """
    x = np.array(x)
    #sum together a sinusoidal and parabolic term
    xEach = -1*a*np.sin((x*w+np.pi/2)) + 2*(x*1.5)**2 + a #min is 0 (at origin)
    y = sum(xEach)
    return y

#### sharing solutions and pairwise communcation ####

def tryToShare(a1,a2,p):
    """
    Two agents perform pairwise communication (attept to share solutions).

    The probability of successful communication depends on the cognitive gap
    (differnce in KAI score). If communication succeeds, they share solutions
    and decide whether to accept the other's solution. If it fails, they do
    nothing this turn.

    Parameters
    ----------
    a1, a2 : Agent objects, the two agents communicating
    p : Params object, contains current model Parameters

    Returns
    -------
    True if communication is successful, else False

    """
    deltaAi = abs(a1.kai.KAI - a2.kai.KAI) #harder to communicate above 20, easy below 10
    successful =  tryComm(deltaAi,p)#deltaR)
    if successful:
        #in share(), agents may adopt better solution depending on temp
        share(a1,a2,p.selfBias)
        return True
    return False

#likelihood of communication success depends on cognitive gap [deltaAi]
#returns True if communication succeeds, else False
def tryComm(deltaAi,p):
    """
    Determine if pairwise communication succeeds or fails

    Probability of success depends on the difference in 2 agents KAI scores

    Parameters
    ----------
    deltaAi : the difference in the 2 agents' KAI scores
    p : Params object, contains current model Parameters

    Returns
    -------
    True if communication is successful, else False

    """
    c = np.random.uniform(p.commBonus,p.commBonus+p.commRange) #increasing commBonus makes sharing easier
    return (deltaAi < c)

def share(a1,a2,selfBias=0):
    """
    Each agent chooses whether to accept shared solution or not

    Parameters
    ----------
    a1, a2 : Agent objects, the two agents communicating
    p : Params object, contains current model Parameters
    selfBias : float, (default: 0) agents percieve their own solution quality
    as this much better than reality.

    Returns
    -------
    True

    """
    copyOfA1 = h.cp(a1)
    considerSharedSoln(a1,a2,selfBias)
    considerSharedSoln(a2,copyOfA1,selfBias)
    return True

def considerSharedSoln(me,sharer,selfBias=0):
    """
    Agent decides whether to accept or reject a shared solution

    The agent evaluates the solution provided by another agent. If it percieves
    the shared solution as better, it stochastically decides whether to
    accept it with pickWorseScore(). If it accepts the shared solution,
    it moves to that solution. Agents never accept a shared solution that they
    percieve as worse than their current solution.

    Parameters
    ----------
    me: Agent object, the agent considering a new solution
    sharer: Agent object, the agent sharing a solution
    selfBias : float, (default: 0) agents percieve their own solution quality
    as this much better than reality.

    """
    candidateSolution = sharer.r
    candidateScore = me.fr(candidateSolution)
    myScore = me.score - selfBias #improve my score by selfBias
    #Quality Bias Reduction would go here, if implemented
    if(candidateScore<myScore):
        if not pickWorseScore(candidateScore,myScore,me.temp):
            me.moveTo(candidateSolution)
