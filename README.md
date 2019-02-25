# kaboom

kaboom is a Python library for the KAI Agent-Based Organizational Optimization
Model (KABOOM) described in a paper currently being submitted for publication.
This agent-based model simulates the performance of teams of engineers solving
a multivariate design problem, and incorporates the cognitive styles and
solution-sharing interactions of agents. Cognitive styles of agents in KABOOM
are based on the [Kirton Adaption-Innovation Inventory](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C39&as_vis=1&q=Kirton+adaption-innovation+in+the+context&oq=kirton+adaption+innovation+in+the+) which describes
a range of problem-solving styles from adaptive (incremental change) to
innovative (radical change).

## Installation

<!-- -->

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install kaboom from Github.

```bash
pip install git+http://github.com/THREDgroup/kaboom.git
```

## Running IDETC Studies
Run in your favorite python compiler

```python
import kaboom

# runs the first study, which analyzes the effect of a team's
#cognitive style composition on performance for 2 problems
kaboom.IDETC_studies.i_teamStyle.run()
# output figure is saved to kaboom/kaboom/IDETC_studies/results/ directory

#run the other experiments from IDETC paper:
kaboom.IDETC_studies.ii_subteamStyle.run()
kaboom.IDETC_studies.iii_strategicTeams.run()
kaboom.IDETC_studies.iv_problemDecomposition.run()
#results figures are saved to kaboom/kaboom/IDETC_studies/results/
```

## Usage

```python
import kaboom

#create a parameters object:
parameters = kaboom.params.Params()

#modify simulation parameter values as desired
parameters.nDims = 14

#run the simulation with an abstract sinusoid objective function
team = kaboom.kaboom.teamWorkSharing(parameters)

#check team performance
#invert score *-1 so that higher score = better performance
print( team.getBestScore()*-1 )

#check other outcomes, such as the number of pairwise interactions:
print ( team.nMeetings )

```
## Running a Car Design problem

```python
from kaboom.carMakers import runCarDesignProblem

#create a parameters object
#parameters (p.nAgents = 33, p.nTeams = 11, p.nDims = 56) are automatically set for this problem.
parameters = kaboom.params.Params()

#run the simulation with the car designer objective
team = runCarDesignProblem(parameters)

#check the performance of the team
#invert score *-1 so that higher score = better performance
print( team.getBestScore()*-1 )

```

## Running a Beam Design problem

```python
from kaboom.runBeamDesigners import runBeamDesignProblem
team = runBeamDesignProblem() #run beam design simulation
print( team.getBestScore()*-1 ) #performance: higher = better
```


## Running Simulations in Parallel

```python
#run simulations in parallel with multiprocessing
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


#now create your own experiment, varying a parameter and checking the performance
```

<!--
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate. -->

## License
[MIT](https://choosealicense.com/licenses/mit/)
