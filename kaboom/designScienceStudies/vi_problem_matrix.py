"""
Run a study of how different style teams perform on 25 different problems.

This script recreates the results shown in Figure X of [1]. It demonstrates...

[1] Lapp, S., Jablokow, J., McComb, C. (2019). "KABOOM: An Agent-Based Model for Simulating Cognitive Style in Team Problem Solving". Unpulished manuscript.
"""
import numpy as np
import time as timer
import multiprocessing
import matplotlib
from matplotlib import pyplot as plt
#import pickle
import itertools
import os

from kaboom.params import Params
from kaboom.kaboom import teamWorkProcess

# # make a matrix of 25 problems for all rougnesses and all speeds
def run():
    t0 = timer.time()
    p=Params()
    p.curatedTeams = True

#    pComm = 0.2 # np.linspace(0,.5,10)

    aiScores = np.linspace(60,140,7)
    p.aiRange = 0

    #pick 5 logarithmically spaced values of roughness (amplitude) and speed
    roughnesses = np.logspace(-1,.7,num=6,base=10)
    roughnesses=roughnesses[1:]
    speeds = np.logspace(-1,.7,num=6,base=10)/100
    speeds=speeds[1:]

    count = 1
    roughness_ai_speed_matrix = []
    for i in range(len(roughnesses)):
        p.amplitude = roughnesses[i]#,8,16,32,64]:
        ai_speed_matrix = []

        for j in range(len(speeds)):
            p.AVG_SPEED = speeds[j]
            scoresForAI = []
            teams = []
            for aiScore in aiScores:
                p.aiScore = aiScore
                if __name__ == '__main__' or 'kaboom.designScienceStudies.vi_problem_matrix':
                    pool = multiprocessing.Pool(processes = 4)
                    allTeams = pool.starmap(teamWorkProcess, zip(range(p.reps),itertools.repeat(p)))
                    scoresForAI.append([t.getBestScore() for t in allTeams])
                    for t in allTeams: teams.append(t)
                    pool.close()
                    pool.join()
            ai_speed_matrix.append(scoresForAI)

            print("completed %s" %count)
            count+=1

        roughness_ai_speed_matrix.append(ai_speed_matrix)

    print("time to complete: "+str(timer.time()-t0))


    # In[416]:
    #
    #f = open('/Users/samlapp/SAE_ABM/results/B1_matrix_of_problems/ProblemMatrixScores.obj','rb')
    #roughness_ai_speed_matrix = pickle.load(f)
    #np.shape(roughness_ai_speed_matrix)
    #aiScores = np.linspace(60,140,7)
    #aiRanges = np.linspace(0,50,5)
    #reps = 8


    # In[420]:


    index = 0
    for i in range(len(roughness_ai_speed_matrix)):
        r =roughness_ai_speed_matrix[i]
        for j in range(len(r)):
            index+=1
            plt.subplot(len(roughness_ai_speed_matrix),len(r),index)

            s = r[j]
            allScores = [ sc for row in s for sc in row]
            allScores = np.array(allScores)*-1
            myKai = [ai for ai in aiScores for i in range(p.reps)]
    #         plt.scatter(myKai,allScores,c=[.9,.9,.9])
    #         plotCategoricalMeans(myKai,allScores)

            pl = np.polyfit(myKai,allScores,2)
            z = np.poly1d(pl)
            x1 = np.linspace(min(myKai),max(myKai),100)
            plt.plot(x1,z(x1),color='red')
    #         print(roughnesses[i])
    #         plt.title("roughness %s speed %s" % (roughnesses[i], speeds[i]))
    #         plt.show()
    #         break
    plt.savefig('./results/vi_problemMatrixDetail'+str(timer.time())+'.pdf')


    # In[29]:

    index = 0

    bestScoreImg = np.zeros([len(roughnesses),len(speeds)])
    for i in range(len(roughness_ai_speed_matrix)):
        r =roughness_ai_speed_matrix[i]
        for j in range(len(r)):
            index+=1

            s = r[j]
            allScores = [ sc for row in s for sc in row]
            myKai = [ai for ai in aiScores for i in range(p.reps)]
    #         plt.scatter(myKai,allScores,c=[.9,.9,.9])
    #         plotCategoricalMeans(myKai,allScores)

            pl = np.polyfit(myKai,allScores,2)
            z = np.poly1d(pl)
            x1 = np.linspace(min(myKai),max(myKai),100)
            y = z(x1)
            bestScore = round(x1[np.argmin(y)])
            bestScoreImg[i,j]=bestScore
    #         plt.plot(x1,z(x1),color='red')
    #         plt.title("roughness %s speed %s" % (roughnesses[i], speeds[i]))
    #         plt.show()
    #         break

    plt.subplot(1,1,1)
    cmapRB = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","red"])
    plt.imshow(bestScoreImg,cmap=cmapRB)
    plt.xlabel('decreasing solution space size')
    plt.ylabel('decreasing amplitude of objective function sinusoid')

    plt.colorbar()
    myPath = os.path.dirname(__file__)
    plt.savefig(myPath+"/results/vi_matrixOfProblems_bestStyle.pdf")


#save results:

# np.savetxt('./results/matrix_of_problems/params.txt',[makeParamString()], fmt='%s')
# np.savetxt('./results/matrix_of_problems/roughnesses_rowsI.txt',[roughnesses], fmt='%s')
# np.savetxt('./results/matrix_of_problems/speeds_columnsJ.txt',[speeds], fmt='%s')

# rFile = './results/matrix_of_problems/'+'ProblemMatrixScores.obj'
# rPickle = open(rFile, 'wb')
# pickle.dump(roughness_ai_speed_matrix, rPickle)
