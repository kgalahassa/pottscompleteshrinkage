# Potts Complete Shrinkage
Potts Clustering with Complete Shrinkage

## Installation
Install using pip
```pip install pottscompleteshrinkage```

## Requirements
* Python 3.6 or greater
* numpy
* pandas

## Usage
Import the Potts Complete Shrinkage module

```import pottsshrinkage.completeshrinkage as PCS```

Choose the number of colors

```q = 20```

Compute Initial Potts Clusters as a first Random Partition (with Potts Model)

```InitialPottsClusters = PCS.InitialPottsConfiguration(Train_PottsData_demo, q, Kernel='Mercel')```

Choose your temperature (T) level

```T = 1000```

Set the bandwidth of the model

```sigma = 1```

Set the Number of Random_Partitions you want to simulate

```Number_of_Random_Partitions = 3```

Set your initial (random) Potts partition as computed above

```Initial_Partition = InitialPottsClusters```

Set the Minimum Size desired for each partition generated

```MinClusterSize = 5```

Run your Potts Complete Shrinkage Model to simulate the Randomly Shrunk Potts Partitions. Partitions_Sets is a dictionary that can be saved 
with pickle package.

```Partitions_Sets,Spin_Configuration_Sets = PCS.Potts_Random_Partition (Train_PottsData_demo, T, sigma, Number_of_Random_Partitions, MinClusterSize, Initial_Partition,  Kernel='Mercel')```

## Pypi Project Page
 https://pypi.org/project/pottscompleteshrinkage/1.0.0/
 
 
## Execution Code Pipeline in Jupyter Notebook 
 https://github.com/kgalahassa/pottscompleteshrinkage-notebook 