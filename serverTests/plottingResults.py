#plotting results

import pickle
import numpy as np
from matplotlib import pyplot as plt
from kaboom import modelFunctions as m

#f = open('/Users/samlapp/SAE_ABM/serverResults/1548796149.5020783nShares_long_smallTeam/results.obj', 'rb')
f = open('/Users/samlapp/SAE_ABM/serverResults/1548906459.554479nShares_short/results.obj', 'rb')

allTeamObjects = pickle.load(f)
print(np.shape(allTeamObjects))

pComms = np.linspace(0,1,6)
reps = 32

allScores = np.array([t.bestScore for t in allTeamObjects])*-1
nS = [t.nMeetings for t in allTeamObjects]
plt.scatter(nS,allScores, c=[.9,.9,.9])
pC = [pc for pc in pComms for i in range(reps)]
plt.xlabel("number of pairwise interactions")
plt.ylabel("performance")
#plt.ylim([-30000,-27000])
plt.show()
plt.scatter(pC,allScores, c=[.9,.9,.9])
c = m.plotCategoricalMeans(pC,allScores)

#domain = m.saveResults(allTeamObjects,"A1.1_sharingRate_12agents")

# reload results
# domain = '/Users/samlapp/SAE_ABM/results/A1.1_optimalSharingRate_long_largerTeam'
# f = open(domain+'/results.obj','rb')
# r = pickle.load(f)
# pComms = np.linspace(0,1,6)
# reps = 16
# r0 = r[0]

r = allTeamObjects
pC = [prob for prob in pComms for i in range(reps)]
perf = np.array([t.bestScore for t in r])*-1
nMeetings = [t.nMeetings for t in r]
plt.scatter(pC,perf,c=[.9,.9,.9])
m.plotCategoricalMeans(pC,perf)
plt.xlabel('prob of communication (c)')
plt.ylabel('performance')
#plt.ylim([-30000,-27000])
#    plt.savefig('./pcomm.pdf')
plt.show()

