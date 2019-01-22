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

Download kaboom and navigate to the top level kaboom directory, then use the package manager [pip](https://pip.pypa.io/en/stable/) to install kaboom.

```bash
cd kaboom
pip install .
```

## Running Example Studies

```python
import kaboom

# runs the first study, which analyzes performance
# for different communication rates
kaboom.test.i_optimalCommRate.run()

```

## Usage

```python
import kaboom

#create a parameters object:
parameters = kaboom.params.Params()

#modify simulation parameter values as desired
parameters.nDims = 14

#run the simulation
team = kaboom.kaboom.teamWorkSharing(parameters)

#check the performance of the team (lower scores are better!)
print( team.getBestScore() )

#check other outcomes, such as the number of pairwise interactions:
print ( team.nMeetings )

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
