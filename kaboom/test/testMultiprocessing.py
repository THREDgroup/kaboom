import multiprocessing
import itertools
from kaboom.kaboom import teamWorkProcess
from kaboom.params import Params

p = Params()
p.steps = 10
p.reps  = 4

allTeamObjects = []
if __name__ == '__main__':
      myPool = multiprocessing.Pool(processes = 4)
      allTeams = myPool.starmap(teamWorkProcess, zip(range(p.reps), itertools.repeat(p) ) )
      for team in allTeams:
          allTeamObjects.append(team)
      myPool.close()
      myPool.join()
      
print("team scores: ")
print([t.getBestScore() for t in allTeamObjects])