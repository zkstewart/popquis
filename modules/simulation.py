import os
import sys

import numpy as np

from concurrent.futures import ProcessPoolExecutor
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.population import Population
from modules.spreadsheet import Spreadsheet
from modules.statistics import Calculator, RandomNumberGenerator

class Configuration:
    '''
    
    Args:
        popSize -- an integer giving the maximum population size to generate simulation
                   configurations for
    Attributes:
        popBalance -- a list of integers: [10, 20, 30, 40, 50]
        phenotypeError -- a list of integers: [0, 10, 20, 30, 40, 50]
        combos -- a dictionary where keys are tuples as: (balance, error)
                  and values are lists of integers giving population size increments
    '''
    def __init__(self, popSize):
        self.popSize = popSize
        self.popBalance = [ x/100 for x in range(10, 60, 10) ] # [10, 20, 30, 40, 50]
        self.phenotypeError = [ x/100 for x in range(0, 60, 10) ] # [0, 10, 20, 30, 40, 50]
        self.get_variable_combinations()
    
    @property
    def popSize(self):
        return self._popSize
    
    @popSize.setter
    def popSize(self, value):
        if not isinstance(value, int):
            raise TypeError(f"popSize must be an int, not '{type(value).__name__}'")
        if value < 10:
            raise ValueError("popSize should be 10 or more, in order to provide meaningful data")
        
        self._popSize = value
    
    def get_variable_combinations(self):
        STEPSIZE = 5
        HEURISTIC_MIN = 10
        HEURISTIC_SCALE = 0.05
        DOWNSIZE_MIN = 2 # half of size
        DOWNSIZE_MAX = 4 # quarter of size
        
        # Obtain a full list of variable combinations with non-fractional pop. sizes
        combos = { x: [] for x in product(self.popBalance, self.phenotypeError) }
        for key, sizeList in combos.items():
            balance, error = key
            for size in range(STEPSIZE, self.popSize+STEPSIZE, STEPSIZE):
                # Check if groups can be segregated into balanced groups
                numGroup1 = size * balance
                if numGroup1 % 1 != 0: # if group1 is a whole number, group2 is as well
                    continue
                
                # Check if each group can be mixed with a whole-numbered amount of errors
                numGroup1 = int(numGroup1)
                numGroup2 = size - numGroup1
                if (numGroup1 * error % 1 != 0) or (numGroup2 * error % 1 != 0):
                    continue
                
                sizeList.append(size)
        
        # Limit computation by thinning the number of combinations
        "Depending on the efficiency of downstream steps, this may be removed"
        self.combos = {}
        for key in combos.keys():
            rawValue = combos[key]
            if (len(rawValue) > HEURISTIC_MIN) and (len(rawValue) > (len(rawValue) * HEURISTIC_SCALE)):
                stored = False
                for downsize in range(DOWNSIZE_MIN, DOWNSIZE_MAX+1):
                    if len(rawValue) % downsize == 0:
                        self.combos[key] = [ rawValue[i] for i in range(0, len(rawValue), downsize)]
                        if self.combos[key][-1] != rawValue[-1]: # make sure it has the first and last values
                            self.combos[key].append(rawValue[-1])
                        stored = True
                        break
                if not stored: # unable to downsize
                    self.combos[key] = rawValue
            else:
                self.combos[key] = rawValue
        
        # Change the data type of the popSize lists to be numpy arrays
        for key in self.combos.keys():
            self.combos[key] = np.array(self.combos[key])
    
    def __iter__(self):
        for key, value in self.combos.items():
            yield key, value
    
    def __repr__(self):
        return "<Configuration object;popSize={0}>".format(
            self.popSize,
        )

class Coordinator:
    '''
    For efficiency reasons, this Class will not validate any inputs. All integer parameters should be
    pre-validated prior to instantiating a Coordinator object.
    
    Args:
        locations -- a Locations object with attributes including those detailed below.

    Attributes:
        group1Npy -- string pointing to a memory-mappable numpy file containing an array with
                     shape (num_individuals, num_genotypes, ploidy)
        group2Npy -- as per group1Npy but for the group2 Population
        storageDir -- a string pointing to the location where .npy files reside
    '''
    def __init__(self, locations):
        self.group1Npy = locations.group1Npy
        self.group2Npy = locations.group2Npy
        self.storageDir = locations.storageDir
    
    @property
    def group1Npy(self):
        return self._group1Npy
    
    @group1Npy.setter
    def group1Npy(self, value):
        if not isinstance(value, str):
            raise TypeError(f"group1Npy must be a str, not '{type(value).__name__}'")
        if not os.path.isfile(value):
            raise ValueError(f"Coordinator expected group1Npy to be a file at '{value}'")
        
        self._group1Npy = value
    
    @property
    def group2Npy(self):
        return self._group2Npy
    
    @group2Npy.setter
    def group2Npy(self, value):
        if not isinstance(value, str):
            raise TypeError(f"group2Npy must be a str, not '{type(value).__name__}'")
        if not os.path.isfile(value):
            raise ValueError(f"Coordinator expected group2Npy to be a file at '{value}'")
        
        self._group2Npy = value
    
    def run(self, configuration, threads, bootstraps=1000, power=4):
        '''
        Parameters:
            configuration -- a Configuration class object
            threads -- an integer giving the number of parallel processes to run where possible
            bootstraps -- an integer giving the number of bootstrap replicates to run
            power -- the power value to raise the Euclidean distance to; default and recommended is 4
        '''
        # Load one of the Populations to obtain part of its shape
        "We need to know how many variants we should be assessing"
        group1 = Population(self.group1Npy)
        group1.load()
        numVariants = group1.variants
        
        # Coordinate the simulation process
        ongoingSeed = 0
        with ProcessPoolExecutor(max_workers=threads) as executor:
            for (popBalance, phenotypeError), popSizes in configuration:
                # Obtain the Spreadsheet this configuration will have results stored within
                spreadsheet = Spreadsheet(self.storageDir, popBalance, phenotypeError, popSizes)
                
                # See if this configuration has been completely processed
                spreadsheet.load() # runs some internal validations of popBalance, phenotypeError, and popSizes
                if spreadsheet.array is not None:
                    expectedShape = (len(popSizes), bootstraps, numVariants)
                    if spreadsheet.shape == expectedShape:
                        continue
                
                # Bootstrap replication of this parameter combination
                futures = []
                for totalPopSize in popSizes:
                    for replication in range(bootstraps):
                        ongoingSeed += 1
                        # Derive the size of each group
                        group1Size = int(totalPopSize * popBalance)
                        group2Size = totalPopSize - group1Size
                        
                        # Derive the number of error samples in each group
                        numGroup1Errors = int(group1Size * phenotypeError)
                        numGroup2Errors = int(group2Size * phenotypeError)
                        
                        # Submit the task to the dedicated executor management function
                        future = executor.submit(self.analyse_population_segregation, self.group1Npy, self.group2Npy,
                                                 group1Size, group2Size, numGroup1Errors, numGroup2Errors,
                                                 power, ongoingSeed)
                        futures.append(future)
                
                # Extract and join resulting arrays
                resultsArray = np.stack([ x.result() for x in futures ]) # shape = (popSize*bootstraps, numVariants)
                resultsArray = np.stack(np.split(resultsArray, len(popSizes))) # shape = (popsize, bootstraps, numVariants)
                
                # Store results in Spreadsheet
                spreadsheet.array = resultsArray
                spreadsheet.save()
    
    def analyse_population_segregation(self, group1Npy, group2Npy, group1Size, group2Size,
                                       numGroup1Errors, numGroup2Errors, power, seed):
        '''
        Thread worker of self.run() for use with ProcessPoolExecutor.submit()
        '''
        # Load Populations and RandomNumberGenerator
        group1 = Population(group1Npy)
        group1.load()
        group2 = Population(group2Npy)
        group2.load()
        rng = RandomNumberGenerator(seed)
        
        # Form segregant groups out of a mixture of error and/ correct samples
        if numGroup1Errors == 0 or numGroup2Errors == 0:
            assert numGroup1Errors == 0 and numGroup2Errors == 0, "sanity check that population phenotype errors are balanced"
            g1Array = group1.retrieve(rng.generate_random_indices(group1.individuals, group1Size))
            g2Array = group2.retrieve(rng.generate_random_indices(group2.individuals, group2Size))
        else:
            numGroup1Correct = group1Size - numGroup1Errors
            numGroup2Correct = group2Size - numGroup2Errors
            
            # Obtain group individuals as mixed bags to ensure proper replacement
            "We can't sample the groups in separate function calls lest it risk using an identical individual in both groups"
            g1Mixed = group1.retrieve(rng.generate_random_indices(group1.individuals, numGroup1Correct + numGroup2Errors))
            g2Mixed = group2.retrieve(rng.generate_random_indices(group2.individuals, numGroup2Correct + numGroup1Errors))
            
            g1Array = np.vstack((g1Mixed[0:numGroup1Correct], g2Mixed[numGroup2Correct:])) # good + bad
            g2Array = np.vstack((g2Mixed[0:numGroup2Correct], g1Mixed[numGroup1Correct:])) # good + bad
        
        # Reshape arrays for computation
        g1Array = g1Array.reshape(group1.variants, group1Size * group1.ploidy)
        g2Array = g2Array.reshape(group2.variants, group2Size * group2.ploidy)
        
        # Calculate Euclidean distance for each genotype and return
        return Calculator.euclidean_distance(g1Array, g2Array, power)
    
    def __repr__(self):
        return "<Coordinator object;group1Npy={0};group2Npy={1}>".format(
            self.group1Npy,
            self.group2Npy
        )
