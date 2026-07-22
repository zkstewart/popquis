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
    Methods:
        get_variable_combinations -- called on object initialisation to set self.combos
    '''
    POPMIN = 10
    
    @staticmethod
    def downsize_popsize_increments(popSize, variableList):
        '''
        Reduces the number of variable combinations to limit simulation computational burden.
        Operates under a scale akin to:
            popSize == 10 means we want at most 10 variables
            popSize == 100 means we want at most 20 variables
            popSize == 1000 means we want at most 40 variables
        And so on. Each 10x increase in popSize leads to a 2x increase in the target number
        of variables.
        
        Parameters:
            popSize -- an integer giving the maximum popSize
            variableList -- a list containing values as part of the variable combinations
        Returns:
            array -- a numpy array with potential subsetting of the original input list if
                     its length exceeded a threshold
        '''
        DOWNSIZE_BASE = 10
        
        target = np.ceil(DOWNSIZE_BASE * np.power(2, np.log10(popSize)-1)).astype(int)
        if len(variableList) > target:
            return np.array([ variableList[i] for i in Calculator.evenly_spaced_sampling(len(variableList), target) ])
        else:
            return np.array(variableList)
    
    def __init__(self, popSize):
        self.popSize = popSize
        self.popBalance = [ x/100 for x in range(10, 60, 10) ] # [10, 20, 30, 40, 50]
        self.phenotypeError = [ x/100 for x in range(0, 60, 10) ] # [0, 10, 20, 30, 40, 50]
        self.get_variable_combinations()
        self.isConfiguration = True # object type validator
    
    @property
    def popSize(self):
        return self._popSize
    
    @popSize.setter
    def popSize(self, value):
        if not isinstance(value, int):
            raise TypeError(f"popSize must be an int, not '{type(value).__name__}'")
        if value < Configuration.POPMIN:
            raise ValueError(f"popSize should be {Configuration.POPMIN} or more, in order to provide meaningful data")
        
        self._popSize = value
    
    def get_variable_combinations(self):
        # Obtain a full list of variable combinations with non-fractional pop. sizes
        combos = { x: [] for x in product(self.popBalance, self.phenotypeError) }
        for key, sizeList in combos.items():
            balance, error = key
            for size in range(Configuration.POPMIN, self.popSize+1):
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
            self.combos[key] = Configuration.downsize_popsize_increments(self.popSize, combos[key])
        
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
    Methods:
        run -- pipeline function for multithreaded computation of the Euclidean distance segregation
               statistics for each simulated variable combination
        analyse_population_segregation -- thread worker function called by run()
    '''
    def __init__(self, locations):
        self.group1Npy = locations.group1Npy
        self.group2Npy = locations.group2Npy
        self.storageDir = locations.storageDir
        self.isCoordinator = True # object type validator
    
    @property
    def group1Npy(self):
        return self._group1Npy
    
    @group1Npy.setter
    def group1Npy(self, value):
        if not isinstance(value, str):
            raise TypeError(f"group1Npy must be a str, not '{type(value).__name__}'")
        if not os.path.isfile(value):
            raise FileNotFoundError(f"Coordinator expected group1Npy to be a file at '{value}'")
        
        self._group1Npy = value
    
    @property
    def group2Npy(self):
        return self._group2Npy
    
    @group2Npy.setter
    def group2Npy(self, value):
        if not isinstance(value, str):
            raise TypeError(f"group2Npy must be a str, not '{type(value).__name__}'")
        if not os.path.isfile(value):
            raise FileNotFoundError(f"Coordinator expected group2Npy to be a file at '{value}'")
        
        self._group2Npy = value
    
    @staticmethod
    def analyse_population_segregation(group1Npy, group2Npy, group1Size, group2Size,
                                       numGroup1Errors, numGroup2Errors, power, seed):
        '''
        Thread worker of a Coordinator object's .run() for use with ProcessPoolExecutor.submit()
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
                if spreadsheet.ed is not None:
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
                        future = executor.submit(Coordinator.analyse_population_segregation, self.group1Npy, self.group2Npy,
                                                 group1Size, group2Size, numGroup1Errors, numGroup2Errors,
                                                 power, ongoingSeed)
                        futures.append(future)
                
                # Extract and join resulting arrays
                if len(futures) != 0:
                    resultsArray = np.stack([ x.result() for x in futures ]) # shape = (popSize*bootstraps, numVariants)
                    resultsArray = np.stack(np.split(resultsArray, len(popSizes))) # shape = (popsize, bootstraps, numVariants)
                else:
                    if len(popSizes) != 0:
                        raise Exception("Coordinator failed as futures list is empty but popSizes is not empty")
                    resultsArray = None
                
                # Store results in Spreadsheet
                spreadsheet.ed = resultsArray
                spreadsheet.save()
    
    def __repr__(self):
        return "<Coordinator object;group1Npy={0};group2Npy={1}>".format(
            self.group1Npy,
            self.group2Npy
        )

class Critic:
    '''
    Args:
        locations -- a Locations object with a .storageDir attribute for use by this Critic object
        breeder -- a Breeder object for use solely when initialising this object; breeder is
                   NOT stored as an attribute of this object.
    Attributes:
        storageDir -- a string as from Locations.storageDir wherein Spreadsheet .npz files
                      can be found
        genomeMap -- a GenomeMap object enabling array indexing for QTL positions
        qtlRanges -- a list of tuples with structure akin to:
                     [
                         (genomeMapIndexStart, genomeMapIndexEnd),
                         ...
                     ]; matches up with self.genomeMap
    Methods:
        _define_qtl_ranges -- initialisation of a Critic object automatically calls this private method
                              to set self.qtlRanges
        run -- pipeline function for multithreaded computation of the R^2 line fitting statistic
               for assessment of whether the Euclidean distance segregation would enable identification
               of a QTL peak
    '''
    def __init__(self, locations, breeder):
        self.storageDir = locations.storageDir
        self._define_qtl_ranges(breeder)
        self.isCritic = True # object type validator
    
    @staticmethod
    def triangle_fit(y):
        '''
        Fits a triangle shape to the y-axis (ED^4) data points. This is done as 1) a triangle
        with points at the minimum y value (left and right) with the maximum at the centre.
        It is also done as 2) the same concept but with a plateau at a local minimum
        on the left and right borders.
        
        The goal is to find a distinct and noticeable peak in the statistics occurring at the site
        where the simulated QTL exists i.e., in the centre of the chromosome. A simple line is
        used to avoid potential overfitting, and to conform to an intuitive sense of how a QTL
        should manifest visually. This trend is measured with R-squared.
        
        Parameters:
            y -- a numpy array of numeric values for the ED^4 segregation of the SNPs
        '''
        # Get the triangle points
        minY = np.min(y)
        maxY = np.max(y)
        midY = (minY + maxY) / 2
        
        centreIndex = len(y) / 2
        quarterIndex = centreIndex / 2
        
        # Handle flat lines
        diffY = max(maxY*0.1, minY*0.50) # account for flat lines by enforcing some difference between min and max
        if diffY < 1e-3:
            diffY = 1e-3 # mitigate issues with extremely low ED^4 values
        
        if (minY+diffY) >= maxY: # we need a noticeable difference between min and max for QTL detection
            maxY += diffY
        
        # Triangle 1: full range peak (^)
        slopeUp = np.linspace(minY, maxY, num=np.floor(centreIndex).astype(int))
        slopeDown = np.linspace(maxY, minY, num=np.ceil(centreIndex).astype(int))
        fullTriangleY = np.concatenate((slopeUp, slopeDown))
        fullRsquared = Calculator.r_squared(y, fullTriangleY)
        
        # Triangle 2: subrange peak (_^_)
        leftFlatIndex = 0
        while leftFlatIndex < quarterIndex: # only plateau up to 1/4 into the 'plot'
            if y[leftFlatIndex] > midY:
                break
            leftFlatIndex += 1 # check the next position in this quadrant
        
        if leftFlatIndex != 0: # this is 0 if the starting position is >= midY
            leftAvgY = np.mean(y[0:leftFlatIndex])
        else:
            leftAvgY = minY
        slopeUp = np.concatenate((
            np.array([ leftAvgY for _ in range(leftFlatIndex)]), # plateau
            np.linspace(leftAvgY, maxY, num=np.floor(centreIndex).astype(int) - leftFlatIndex) # peak (incline)
        ))
        
        rightFlatIndex = len(y)-1
        while rightFlatIndex > (np.ceil(centreIndex).astype(int) + quarterIndex): # only plateau for the last 1/4 of the 'plot'
            if y[rightFlatIndex] > midY:
                break
            rightFlatIndex -= 1 # crawl back into this quadrant
        
        if rightFlatIndex != len(y)-1: # this is the final index if the ending position is >= midY
            rightAvgY = np.mean(y[rightFlatIndex:])
        else:
            rightAvgY = minY
        slopeDown = np.concatenate((
            np.linspace(maxY, rightAvgY, num=rightFlatIndex - np.floor(centreIndex).astype(int)), # peak (decline)
            np.array([ rightAvgY for _ in range(len(y) - rightFlatIndex)]) # plateau
        ))
        
        subrangeTriangleY = np.concatenate((slopeUp, slopeDown))
        subrangeRsquared = Calculator.r_squared(y, subrangeTriangleY)
        
        # Return the best R^2 value
        if (fullRsquared >= subrangeRsquared) or np.isnan(subrangeRsquared):
            return fullTriangleY, fullRsquared
        else:
            return subrangeTriangleY, subrangeRsquared
    
    def _define_qtl_ranges(self, breeder):
        '''
        Parameters:
            breeder -- a Breeder object with .markerIndices attribute for identifying the row
                       index of the simulated QTL(s)
        '''
        # Store a modified GenomeMap recording the position of any QTLs
        self.genomeMap = breeder.genomeMap
        self.genomeMap.df["QTL"] = [ x in breeder.markerIndices for x in list(self.genomeMap.df.index) ]
        
        # Establish the qtlRanges attribute
        self.qtlRanges = []
        for chromID in self.genomeMap.chromIDs:
            # Subset the genomeMap's underlying DataFrame for relevant values
            chromDF = self.genomeMap.df[self.genomeMap.df["CHR.PHYS"] == chromID]
            chromQTLs = chromDF[chromDF["QTL"]]
            
            # Iterate through this chromosome to define the range of each QTL
            lastEnd = None
            rows = list(chromQTLs.itertuples())
            for i, row in enumerate(rows):
                # Get the start point of this QTL region
                if i == 0:
                    startIndex = chromDF.iloc[0].name
                else:
                    startIndex = lastEnd
                
                # Get the end point of this QTL
                if (i+1) == len(rows):
                    endIndex = chromDF.iloc[-1].name
                else:
                    endIndex = int((row.Index + rows[i+1].Index) / 2)
                
                # Store and iterate
                self.qtlRanges.append((startIndex, endIndex))
                lastEnd = endIndex
        
        # Sort qtlRanges to match up with self.genomeMap
        "self.genomeMap.chromIDs is a set, and hence the ordering of chromosomes is not guaranteed"
        self.qtlRanges.sort(key = lambda x: x[0])
    
    def run(self, configuration, threads):
        '''
        Parameters:
            configuration -- a Configuration object recording the simulation variable combinations
            threads -- an integer giving the number of parallel processes to run where possible
        '''
        with ProcessPoolExecutor(max_workers=threads) as executor:
            for (popBalance, phenotypeError), popSizes in configuration:
                if popSizes is None:
                    continue
                
                # Obtain the Spreadsheet this configuration will have results stored within
                spreadsheet = Spreadsheet(self.storageDir, popBalance, phenotypeError, popSizes)
                spreadsheet.load() # runs some internal validations of popBalance, phenotypeError, and popSizes
                
                # See if this configuration has been completely processed
                if hasattr(spreadsheet, "fitted1"): # .fitted1 is set if we have run this at least partially
                    expectedAttr = [ f"fitted{i+1}" for i in range(len(self.qtlRanges)) ]
                    if all([ hasattr(spreadsheet, x) for x in expectedAttr ]):
                        "This check is technically passable with faulty data, but you'd really have to be TRYING to kill popquis..."
                        continue
                
                # If not, iterate through each QTL to generate its results
                for i, (startIndex, endIndex) in enumerate(self.qtlRanges):
                    # Slice the Spreadsheet ED array to get the statistics for this range
                    assert spreadsheet.ed is not None, "sanity check; if we are running Critic, .ed must be set already"
                    qtlED = spreadsheet.ed[:,:,startIndex:endIndex] # TBD: check that this is appropriately inclusive of the QTL range
                    
                    # Assess each replication of this parameter combination
                    futures = []
                    for popSizeArray in qtlED:
                        for replicateArray in popSizeArray:
                            future = executor.submit(Critic.triangle_fit, replicateArray)
                            futures.append(future)
                    
                    # Extract and join resulting arrays
                    if len(futures) != 0:
                        fittedY, r2Values = zip(*[ x.result() for x in futures ])
                        fittedArray = np.stack(fittedY) # shape = (popSize*bootstraps, numVariants)
                        fittedArray = np.stack(np.split(fittedArray, len(popSizes))) # shape = (popsize, bootstraps, numVariants)
                        
                        resultsR2 = np.stack(np.split(np.array(r2Values), len(popSizes))) # shape = (popsize, bootstraps)
                    else:
                        raise Exception("Critic failed as futures list is empty but popSizes is not empty")
                    
                    # Store results in Spreadsheet
                    setattr(spreadsheet, f"fitted{i+1}", fittedArray)
                    setattr(spreadsheet, f"rsq{i+1}", resultsR2)
                
                # Store the results into the Spreadsheet
                spreadsheet.save()
    
    def __repr__(self):
        return "<Critic object;storageDir='{0}';qtlRanges={1}>".format(
            self.storageDir,
            self.qtlRanges
        )
