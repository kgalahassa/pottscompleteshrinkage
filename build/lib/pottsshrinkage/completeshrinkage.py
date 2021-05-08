import pandas as pd
import random
import numpy as np
from numpy import linalg as LA
import csv
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
covariables_train = iris.data[:, :3]  # we only take the first three features.

#resptraining = pd.read_csv(args.resptraining, sep=',',header=None, decimal ='.')

response_train = iris.target
DatasetName = "Jura"

NRP  = 10
CMinSize = 5

#covtesting=pd.read_csv(args.covtesting, sep=',',header=None, decimal ='.')

#covariables_test = np.loadtxt(args.covtesting, delimiter=',' )

#resptesting=pd.read_csv(args.resptesting, sep=',',header=None, decimal ='.')

#response_test = np.loadtxt(args.resptesting, delimiter=',' )

####################################################################################################################################################################################
####################################################################################################################################################################################

#Pour le training, build training data
xtrain,ytrain = covariables_train, response_train

#Pour tester build test data
#xtest,ytest = covariables_test, response_test 

Train_PottsData = xtrain

#Test_PottsData = xtest

#------------------------------------------ -------------------------------------------------------#
#            Les trois paramètres à complèter dans une version ultérieure                          #
#--------------------------------------------------------------------------------------------------#

q = 20
T =1000
sigma = 1

#--------------------------------------------------------------------------------------------------#

Initial_Spin_Configuration = []

for i in range(len(Train_PottsData)):
    
    Initial_Spin_Configuration.append(random.randint(1,q))


from collections import defaultdict
# function for adding edge to graph 
graph = defaultdict(list) 


# Python program to print connected  
# components in an undirected graph
#https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
class Graph: 
      
    # init function to declare class variables 
    def __init__(self,V): 
        self.V = V 
        self.adj = [[] for i in range(V)] 
  
    def DFSUtil(self, temp, v, visited): 
  
        # Mark the current vertex as visited 
        visited[v] = True
  
        # Store the vertex to list 
        temp.append(v) 
  
        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False: 
                  
                # Update the list 
                temp = self.DFSUtil(temp, i, visited) 
        return temp 
  
    # method to add an undirected edge 
    def addEdge(self, v, w): 
        self.adj[v].append(w) 
        self.adj[w].append(v) 
  
    # Method to retrieve connected components 
    # in an undirected graph 
    def connectedComponents(self): 
        visited = [] 
        cc = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                cc.append(self.DFSUtil(temp, v, visited)) 
        return cc
    
# This code is contributed by Abhishek Valsan    






from collections import OrderedDict

def findneighbors(i, Train_PottsData, Initial_Spin_Configuration, T, sigma, k_voisins = 10):
    
    Compute_Norms  = {}
    
    for j in range(len(Train_PottsData)):
        
        bond_ij_proba = 1 - np.exp(-(1/T)*(LA.norm(Train_PottsData[i,:] - Train_PottsData[j,:]))/(sigma))
        
        bond_ij = np.random.binomial(size=1, n=1, p=bond_ij_proba) 
        
        if (i != j and Initial_Spin_Configuration[i] == Initial_Spin_Configuration[j] and bond_ij[0]==1 ):
            
            Compute_Norms[j] = LA.norm(Train_PottsData[i,:] - Train_PottsData[j,:])
                                       

    OrderedCompute_Norms = OrderedDict(sorted(Compute_Norms.items(), key=lambda x: x[1]))

    OCN_size  = len(OrderedCompute_Norms)
    
    SelectedOrderedCompute_Norms = list(OrderedCompute_Norms)#[(OCN_size -k_voisins):OCN_size ]
                                       
    return SelectedOrderedCompute_Norms      



#for i in range(len(Train_PottsData)):
    
    #let's get the top neighbors of observation i
    
#    Selected_Neighbors = findneighbors(i, Train_PottsData, Initial_Spin_Configuration, T, sigma, k_voisins = 1)
    
#    for j in Selected_Neighbors:
        
        #addEdge(graph,i,j)
#        My_Potts_Graph.addEdge(i,j)


#Potts_Clusters = My_Potts_Graph.connectedComponents() 

#print(Potts_Clusters, "DONE")


def InitialPottsConfiguration(Train_PottsData_demo, q, Kernel="Mercel"):
    
    Initial_Spin_Configuration_demo = []

    for i in range(len(Train_PottsData_demo)):
    
        Initial_Spin_Configuration_demo.append(random.randint(1,q))
    
    My_Potts_Graph_demo = Graph(len(Train_PottsData_demo))
    
    for s in range(len(Train_PottsData_demo)):

        #let's get the top neighbors of observation i

        Selected_Neighbors_demo = findneighbors(i, Train_PottsData_demo, Initial_Spin_Configuration_demo, T, sigma, k_voisins = 1)

        for j in Selected_Neighbors_demo:

            #addEdge(graph,i,j)
            My_Potts_Graph_demo.addEdge(i,j)


    Potts_Clusters_demo = My_Potts_Graph_demo.connectedComponents()
    
    return Potts_Clusters_demo 
    


def Compute_Partition (Train_PottsData, _Spin_Configuration, T, sigma):
    
    
    """ 
    
    Given the Data and Spin Configuration, this function compute the Partition
    
    Parameters : 
    ----------
    
    PottsData: the features data, X
    
    Initial_Spin_Configuration : Initial Spin configuration for all observations
    
    T : The temperature 
    
    sigma : The bandwitch
    
    """
    
    _My_Potts_Graph = Graph(len(Train_PottsData))
    
    for i in range(len(Train_PottsData)):
        #let's get the top neighbors of observation i

        Selected_Neighbors = findneighbors(i, Train_PottsData, _Spin_Configuration, T, sigma, k_voisins = 1)

        for j in Selected_Neighbors:

            #addEdge(graph,i,j)
            _My_Potts_Graph.addEdge(i,j)

                
    _Potts_Clusters = _My_Potts_Graph.connectedComponents() 
    
    return _Potts_Clusters

def Partitions_Clusters_Adjustments (New_Partition, Min_Cluster_Size):
    
    Copy_New_Partition = New_Partition.copy()
    
    List_of_clusters_size = [len(cluster) > Min_Cluster_Size for cluster in New_Partition]
    
    Somme_List_of_clusters_size = np.sum(List_of_clusters_size)
    
    
    if len(List_of_clusters_size)>0 and len(Copy_New_Partition)> Somme_List_of_clusters_size: 
        
        Position_Cluster_To_be_adjusted = List_of_clusters_size.index(0) 
        
    return adjusted_Partition


###Some supplementaries Code to adjust the clustering

import numpy as np
from numpy import linalg as LA

def ChangePartition(Partition, min_size, Train_PottsData):

    MyPartition = Partition.copy()
    
    max_iter = len(MyPartition)
    
    SearchCluster = [len(Cluster) > min_size for Cluster in MyPartition]
    
    if 0 in SearchCluster:
        ChooseTheFirstCluster = SearchCluster.index(0)

        distancelist = []
        
        for _i in range(max_iter):
                
                if _i!= ChooseTheFirstCluster:
                
                   distancelist.append(compute_distance(MyPartition[_i],MyPartition[ChooseTheFirstCluster], Train_PottsData))
        
                else:
                     distancelist.append(0)   
                                       
        distancelist[ChooseTheFirstCluster] =  np.sum(distancelist)
                                       
        Cluster_To_MergeWith = np.argmin(distancelist)
                                       
        MyPartition[Cluster_To_MergeWith] = MyPartition[Cluster_To_MergeWith]+MyPartition[ChooseTheFirstCluster]
                                       
        del MyPartition[ChooseTheFirstCluster]
                                       
    return MyPartition                                   

def compute_distance(cluster1,cluster2, Train_PottsData):

    cluster_1 = cluster1.copy()
    cluster_2 = cluster2.copy()
    
    
    all_distances= []
    
    
    for j in cluster_1:
        
        all_distances.extend([LA.norm(Train_PottsData[j,:] - Train_PottsData[k,:]) for k in cluster_2])
        
        
    return float(np.max(all_distances))

def AdjustPartition(GeneratedPartition, min_cluster_size, Train_PottsData):
    
    NewPartition0 = GeneratedPartition

    NewPartition1 = ChangePartition(GeneratedPartition, 1, Train_PottsData)


    while NewPartition1 != NewPartition0:

          NewPartition0 = NewPartition1
            
          #Nous souhaitons ajuster les partitions à une taille de cluster minimum==5  
          NewPartitionAdj = ChangePartition(NewPartition0, min_cluster_size, Train_PottsData) 

          NewPartition1 = NewPartitionAdj

    return NewPartition1

###########
def Potts_Random_Partition (Train_PottsData, T, sigma, Number_of_Random_Partitions, MinClusterSize, Initial_Partition,  Kernel="Mercel") : 
    
    
    """ 
    
    This function generates _Random_Partitions for a given initial Potts_Clusters
    
    Parameters
    ----------
    
    Initial_Partition : A given initial (random partition) in defaultdict(list) format
    
    Number_of_Random_Partitions: Number of expected random partitions, must be greater than 0 preferably
    
    
    Return    
    ------
    
    Full_Observations_Spin_Configuration : A full list of spin configuration for each generated partition 
    
    Full_Partition_Sets : A full list of all generated partitions
    
    
    """
    
    Full_Observations_Spin_Configuration = defaultdict(list) 
    
    Full_Partition_Sets = defaultdict(list) 
    
    Actual_Partition = Initial_Partition
    
    k = 0
    
    while k < (Number_of_Random_Partitions + 1):
        
        
            #Create the Clustter Component spin configuration 

            _Cluster_Spin_Configuration = []

            for h in range(len(Actual_Partition)):

                _Cluster_Spin_Configuration.append(random.randint(1,q))

            #Find observation spin configuration

            Observations_Spin_Configuration = []

            for observation in range(len(Train_PottsData)):

                Observation_Cluster_index = [ int(observation in Cluster) for Cluster in  Actual_Partition ].index(1)

                Observations_Spin_Configuration.append(_Cluster_Spin_Configuration[Observation_Cluster_index])
            
            
            Full_Observations_Spin_Configuration[k] = Observations_Spin_Configuration
            
            
            New_Partition = Compute_Partition (Train_PottsData, Observations_Spin_Configuration, T, sigma)

            #print(New_Partition)
            
            AdjustedPartition = AdjustPartition(New_Partition, MinClusterSize, Train_PottsData)
            
            List_of_clusters_size = [len(cluster) for cluster in AdjustedPartition]
            
            if  int(np.min(List_of_clusters_size)) >= 1 : 
                
                                
                Full_Partition_Sets[k] = AdjustedPartition

                k = k + 1
                print("We are at step: %i"%k)
                print("Clusters Size of Current Partition", List_of_clusters_size)
                print("Partition is:", AdjustedPartition)
            Actual_Partition = AdjustedPartition
            
    return Full_Partition_Sets, Full_Observations_Spin_Configuration


def main():
    
    print("Run 'pottscomplete -h' to see my complete usage manual.")

import time
start_time = time.time()

myNRP = NRP

#Partitions_Sets,Spin_Configuration_Sets = Potts_Random_Partition (Train_PottsData, T, sigma, Number_of_Random_Partitions, MinClusterSize, Initial_Partition,  Kernel="Mercel")

#print("%i Partitions generated-- %s seconds ---DONE!" % (NRP, time.time() - start_time))

#import pickle
#output = open('%s_%i_Partitions_constraints_%i_Sets.pkl'%(DatasetName,myNRP,CMinSize), 'wb')
#pickle.dump(Partitions_Sets, output)
#output.close()
