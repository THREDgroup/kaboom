# # Specialization/Structure vs Style C3
import numpy as np
import time as timer
#import multiprocessing
from matplotlib import pyplot as plt
#import pickle
#import itertools
import scipy

from kaboom.params import Params
from kaboom import modelFunctions as m
from kaboom.kaboom import robustnessTest

def run():
    
    # ## B 2.3
    # Robustness on two diagonals
    # 
    
    # PROBLEM SET 1
    #problems favor a range of styles from adaptive to mid to innovative
    
    t0 = timer.time()
    p=Params()
    
    p.curatedTeams = True
    
    p.reps = 32
    
    allTeams = []
    allTeamScores = []
    p.aiScore = 100
    aiRanges = np.linspace(0,100,6)
    
#    pComm = 0.2
    
    #the diagonal across preferred style:
    roughnesses = np.logspace(-1,.7,num=6,base=10)#[.2,.6,1.8,5.4]
    roughnesses = roughnesses[1:]
    speeds = np.logspace(-1,.7,num=6,base=10)/100 # [.001,.004,.016,.048]
    speeds = speeds[1:]
    
    problemSet = [roughnesses,speeds]
    
    for aiRange in aiRanges:
        p.aiRange = aiRange
        teams = []
        teamScores = []
        for i in range(p.reps):
            scores, t = robustnessTest(p,problemSet)
            teams.append(t)
            teamScores.append(scores)
        allTeams.append(teams)
        allTeamScores.append(teamScores)
        print('next')
    print('time: %s' % (timer.time() - t0))
    
    
    #
    #teams = [ t for tt in allTeams for t in tt]
    #directory = saveResults(teams,'robustnessTest_diag1')
    #rFile = directory+'/'+'scoreMatrix.obj'
    #rPickle = open(rFile, 'wb')
    #pickle.dump(allTeamScores,rPickle)
    # f = open('/Users/samlapp/SAE_ABM/results/1542634807.0205839robustnessTest/scoreMatrix.obj','rb')
    # sm = pickle.load(f)
    
    
    # In[67]:
    
    
    #STADARDIZE for each problem!
    ats = np.array(allTeamScores)
    problemMeans = [ np.mean(ats[:,:,i]) for i in range(len(problemSet[0]))]
    problemSds = [ np.std(ats[:,:,i]) for i in range(len(problemSet[0]))]
    allTeamScoresStandardized = ats
    for j in range(len(allTeamScoresStandardized)):
        for i in range(len(problemSet[0])):
            for k in range(p.reps):
                allTeamScoresStandardized[j,k,i] = (ats[j,k,i]-problemMeans[i])/problemSds[i]
    np.shape(allTeamScoresStandardized)
    
    meanScores = [ np.mean(t) for teamSet in allTeamScoresStandardized for t in teamSet ]
    meanGrouped = [ [np.mean(t) for t in teamSet] for teamSet in allTeamScoresStandardized]
    sdScores = [ np.std(t) for teamSet in allTeamScoresStandardized for t in teamSet ]
    sdGrouped = [ [np.std(t) for t in teamSet] for teamSet in allTeamScoresStandardized]
    ranges = [ t.dAI for teamSet in allTeams for t in teamSet ]
    robustness = np.array(meanScores)-np.array(sdScores)
    robustnessGrouped = np.array(meanGrouped)-np.array(sdGrouped)
    
    
    # meanScores = np.array(meanScores)*-1
    
    plt.scatter(ranges,np.array(meanScores),c=[.9,.9,.9])
    # plt.title("9 problem matrix")
    
    cms = m.plotCategoricalMeans(ranges, meanScores)
    #plt.savefig('results/vii_set1_robustnessInv.pdf')
    
    stat, pscore = scipy.stats.ttest_ind(meanGrouped[0],meanGrouped[2])
    print("significance: p= "+str(pscore))
    corr, _ = scipy.stats.pearsonr(ranges[0:p.reps*4],meanScores[0:p.reps*4])
    print('Pearsons correlation: %.3f' % corr)
    
    # In[2]
    # PROBLEM SET 2
    # Now the other diagonal (all 5 problems prefer mid-range style)
    
    allTeamsD2 = []
    allTeamScoresD2 = []
#    aiScore = 100
    aiRanges = np.linspace(0,100,6)
    
    #the diagonal across preferred style:
    roughnesses = np.logspace(-1,.7,num=6,base=10)#[.2,.6,1.8,5.4]
    roughnesses = roughnesses[1:]
    speeds = np.logspace(-1,.7,num=6,base=10)/100 # [.001,.004,.016,.048]
    speeds = speeds[1:]
    #reverse the order: pair large speed (small space) with small roughness
    speeds = speeds[::-1]
    
    problemSet = [roughnesses,speeds]
    
    for aiRange in aiRanges:
        p.aiRange = aiRange
        teams = []
        teamScores = []
        for i in range(p.reps):
            scores, t = robustnessTest(p,problemSet)
            teams.append(t)
            teamScores.append(scores)
        allTeamsD2.append(teams)
        allTeamScoresD2.append(teamScores)
        print('next')
    print('time: %s' % (timer.time() - t0))
    
    
    #teams = [ t for tt in allTeamsD2 for t in tt]
    #directory = saveResults(teams,'robustnessTest_diag2')
    #rFile = directory+'/'+'scoreMatrix.obj'
    #rPickle = open(rFile, 'wb')
    #pickle.dump(allTeamScores,rPickle)
    # f = open('/Users/samlapp/SAE_ABM/results/1542634807.0205839robustnessTest/scoreMatrix.obj','rb')
    # sm = pickle.load(f)
    
    #STADARDIZE for each problem!
    ats = np.array(allTeamScoresD2)
    problemMeans = [ np.mean(ats[:,:,i]) for i in range(len(problemSet[0]))]
    problemSds = [ np.std(ats[:,:,i]) for i in range(len(problemSet[0]))]
    allTeamScoresStandardized = ats
    for j in range(len(allTeamScoresStandardized)):
        for i in range(len(problemSet[0])):
            for k in range(p.reps):
                allTeamScoresStandardized[j,k,i] = (ats[j,k,i]-problemMeans[i])/problemSds[i]
    np.shape(allTeamScoresStandardized)
    
    
    meanScoresD2 = [ np.mean(t) for teamSet in allTeamScoresStandardized for t in teamSet ]
    meanGroupedD2 = [ [np.mean(t) for t in teamSet] for teamSet in allTeamScoresStandardized]
    sdScoresD2 = [ np.std(t) for teamSet in allTeamScoresStandardized for t in teamSet ]
    sdGroupedD2 = [ [np.std(t) for t in teamSet] for teamSet in allTeamScoresStandardized]
    ranges = [ t.dAI for teamSet in allTeams for t in teamSet ]
    # robustness = np.array(meanScores)-np.array(sdScores)
    # robustnessGrouped = np.array(meanGrouped)-np.array(sdGrouped)
    
    #flip the scores
    meanScoresD2 = np.array(meanScoresD2)*-1
    
    
    plt.scatter(ranges,np.array(meanScoresD2),c=[.9,.9,.9])
    
    cms = m.plotCategoricalMeans(ranges, np.array(meanScoresD2))
    
    stat, p = scipy.stats.ttest_ind(meanGroupedD2[0],meanGroupedD2[3])
    print("significance: p= "+str(p))
    corr, _ = scipy.stats.pearsonr(ranges[0:p.reps*4],meanScoresD2[0:p.reps*4])
    print('Pearsons correlation: %.3f' % corr)
    
    
    # plt.scatter(ranges,np.array(meanScoresD2),c=[.9,.9,.9])
    
    cms = m.plotCategoricalMeans(ranges, np.array(meanScoresD2))
    cms2 = m.plotCategoricalMeans(ranges, np.array(meanScores))
    plt.xlabel('maximum cognitive gap (style diversity)')
    plt.ylabel('performance')
    plt.legend(['probem set 1','problem set 2'])
    
    plt.savefig('results/vii_diverseTeams_2problemsets.pdf')
    
    # stat, p = scipy.stats.ttest_ind(meanGroupedD2[0],meanGroupedD2[3])
    # print("significance: p= "+str(p))
    # corr, _ = pearsonr(ranges[0:reps*4],meanScoresD2[0:reps*4])
    # print('Pearsons correlation: %.3f' % corr)
