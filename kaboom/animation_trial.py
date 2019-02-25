import numpy as np
from matplotlib import pyplot as plt
from kaboom import helperFunctions as h
from kaboom import modelFunctions as m
import itertools

import os

from kaboom.params import Params
from kaboom import modelFunctions as m

from kaboom.kaboom import teamWorkSharing

p= Params()
p.nAgents = 8
p.nDims = 4
p.nTeams = 2
p.reps = 32
p.steps = 50
#p.steps = 50
p.agentTeams = m.specializedTeams(p.nAgents,p.nTeams)
p.teamDims = m.teamDimensions(p.nDims,p.nTeams)

aiScores = np.linspace(45,145,9)
allScores = []
for aiScore in aiScores:
    scores = []
    for i in range(p.reps):
        t= teamWorkSharing(p,BeamDesigner)
        scores.append(t.getBestScore())
    allScores.append(scores)
    print('next')


allScoresFlipped = np.array(allScores)*-1
allAiScores = [ai for ai in aiScores for i in range(p.reps)]
allScoresFlat= [ s for r in allScoresFlipped for s in r]
for i,aiScore in enumerate(aiScores):
    plt.scatter(np.ones(p.reps)*aiScore, allScoresFlipped[i])
    

plt.scatter(allAiScores,allScoresFlat,c=[.9,.9,.9])
    
m.plotCategoricalMeans(allAiScores,allScoresFlat)

qm = np.polyfit(allAiScores,allScoresFlat, 2)
qmodel = np.poly1d(qm)
x = np.linspace(45,145,101)
y = qmodel(x)
bestStyle = x[np.argmin(y)]
print("Best style:" +str(bestStyle))
plt.plot(x, qmodel(x))
#plt.ylim([-17,-14])
    
plt.savefig("/Users/samlapp/SAE_ABM/figs/beamDesigners_quadratic.pdf")


#for each pair
for pair in itertools.combinations(range(len(allScores)),2):
    
    pS = h.pScore(allScores[pair[0]],allScores[pair[1]])
    fxSize = effectSize(allScores[pair[0]],allScores[pair[1]])
    if fxSize >0.2:
        print(pair)
        print(pS)
        print(fxSize)

#    plt.scatter([aiScores],[np.mean(allScores[i])],c='black')
#
#files = []
#mem = t.agents[0].memory
#for i,mi in enumerate(mem):
#    if i%(int(len(mem)/10))==0:
#        b = Beam(mi.r)
#        b.draw()
#        fname = '_tmp%03d.png' % i
#        plt.savefig(fname)
#        files.append(fname)
#        
#print('Making movie animation.mpg - this may take a while')
#subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
#                "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)
#
## cleanup
#for fname in files:
#    os.remove(fname)
    
