# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

import numpy as np

from concurrent.futures import ProcessPoolExecutor
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.population import Population
from modules.spreadsheet import Spreadsheet
from modules.statistics import Calculator, RandomNumberGenerator
from modules.template import Template

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
        g1Array = g1Array.transpose(1, 0, 2).reshape(group1.variants, -1) # starts as (individuals, variants, ploidy)
        g2Array = g2Array.transpose(1, 0, 2).reshape(group2.variants, -1) # ends as (variants, individuals * ploidy)
        
        # Calculate Euclidean distance for each genotype and return
        return Calculator.euclidean_distance(g1Array, g2Array, power)
    
    def run(self, configuration, qtlRanges, threads, bootstraps=1000, power=4):
        '''
        Parameters:
            configuration -- a Configuration class object
            qtlRanges -- a list of tuples giving (start, end) indices for each QTL, as
                         derived from the Breeder object
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
                if popSizes is None or len(popSizes) == 0:
                    continue
                
                # Obtain the Spreadsheet this configuration will have results stored within
                spreadsheet = Spreadsheet(self.storageDir, popBalance, phenotypeError, popSizes)
                spreadsheet.load() # runs some internal validations of popBalance, phenotypeError, and popSizes
                
                # See if this configuration has been completely processed
                if hasattr(spreadsheet, "ed1"): # .ed1 is set if we have run this at least partially
                    expectedAttr = [ f"ed{i+1}" for i in range(len(qtlRanges)) ]
                    if expectedAttr == spreadsheet.get_unfixed_attributes("ed"):
                        continue
                
                # Bootstrap replication of each QTL for this parameter combination
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
                    raise Exception("Coordinator failed as futures list is empty; cause is unknown")
                
                # Store results in Spreadsheet
                for i, (startIndex, endIndex) in enumerate(qtlRanges):
                    qtlArray = resultsArray[:,:,startIndex:endIndex+1] # +1 for end inclusive range
                    setattr(spreadsheet, f"ed{i+1}", qtlArray)
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
    Attributes:
        storageDir -- a string as from Locations.storageDir wherein Spreadsheet .npz files
                      can be found
        genomeMap -- a GenomeMap object enabling array indexing for QTL positions
        qtlRanges -- a list of tuples with structure akin to:
                     [
                         (genomeMapIndexStart, genomeMapIndexEnd),
                         ...
                     ]
    Methods:
        run -- pipeline function for multithreaded computation of the R^2 line fitting statistic
               for assessment of whether the Euclidean distance segregation would enable identification
               of a QTL peak
    '''
    WEAK_SCORE = 0.25
    MID_SCORE = 0.50
    STRONG_SCORE = 0.75
    
    def __init__(self, locations):
        self.storageDir = locations.storageDir
        self.isCritic = True # object type validator
    
    @staticmethod
    def penalty(y, globalMin, globalMax, left=True):
        '''
        Provides a scaling factor / correction / penalty to data distributions
        that do not provide a meaningfully visible distinction between the minimum
        and maximum values.
        
        Three components are measured when deriving this penalty:
            1) Is the outside edge of the QTL the minimum value?
            2) Is the inside/centre of the QTL the maximum value?
            3) Is the median closer to the minimum value, than it is to the
               maximum value?
        
        The first and second components measure the standard intuitive metric of a QTL;
        the position of the QTL should have a higher segregation statistic than the
        outer perimeter of the QTL. The third component is less intuitive, but it provides
        a heuristic measurement of the slope ascent; a distinct climb to a peak should
        mean the median is closer to the minimum value, than it is to the maximum value.
        
        Gaussian template fitting is performed assymetrically, so we must differentiate
        between the left and right sides. For the left side of a Template, the left
        edge is the outside of the QTL and the right side is the centre of the QTL.
        
        Parameters:
            y -- a numpy array for the left or right side of a data distribution
            globalMax -- the maximum value for the entire data distribution, not limited
                         to left or right side alone
            left -- a boolean indicating whether this data distribution is the left side
                    (True) of a QTL or the right (False)
        '''
        NOTICEABLE_INCREASE = 1.2
        
        localMin = np.min(y)
        if localMin == 0:
            return 1.0 # any increase over zero is significant
        medianY = np.median(y)
        localMax = np.max(y)
        
        if left:
            troughFactor = localMin / y[0] # is the left edge a trough
            peakFactor = y[-1] / globalMax # is the right edge a global peak
        else:
            troughFactor = localMin / y[-1] # is the right edge a trough
            peakFactor = y[0] / globalMax # is the left edge a global peak
        
        if localMax != localMin:
            medianFraction = (medianY - localMin) / (localMax - localMin) # is the minimum closer to the median than the maximum
            slopeFactor = np.clip(
                2 * (1 - medianFraction), # if median is closer to minimum, 1-medianFraction becomes > 0.5
                0, # this should never activate, clipping is solely to prevent a value > 1
                1 # if median is closer to minimum, we get a factor > 1 which isn't suitable
            )
        else:
            slopeFactor = 0 # a flat line has no slope
        
        magnitudeFactor = max(
            ((localMax - localMin) / (globalMax - globalMin)), # global comparison; handles small min-max differences
            min(localMax / (localMin * NOTICEABLE_INCREASE), 1) # local comparison; handles larger differences
        )
        
        return ( (troughFactor + peakFactor + slopeFactor) / 3 ) * magnitudeFactor
    
    @staticmethod
    def score(y):
        '''
        Evaluate whether the data points match a guassian Template shape, whereby a match
        would indicate a visibly identifiable peak in the segregation statistics at a
        central QTL.
        
        The magnitude of the difference between the central peak and the edge minima is
        factored in to ensure that the difference is biologically meaningful and visually
        identifable if the data were plotted.
        
        Parameters:
            y -- a numpy array of numeric values for the ED^4 segregation of the SNPs
            templates -- a list of one or more Template objects for assessing the shape
                         of the data distribution
        Returns:
            score -- a float ranging from zero (worst) to one (best) measuring a QTL's
                     ability to be identified in the data
            width -- a float ranging from zero (pointy) to 0.5 (broad) measuring the QTL's
                     genomic resolution
        '''
        if np.isclose(np.std(y), 0): # isclose to tolerate floating point inaccuracy
            return 0, None, None # a flat line should have 0 score; also speed up program and avoid divide by zero error later
        
        # Optimise the correlation between a Guassian Template and this ED distribution
        "This models an idealised QTL curve with a maximum peak and a variably-shaped slope to the edge minimum"
        leftCorr, rightCorr, leftWidth, rightWidth = Template.fit_gauss(y)
        
        # Score the focal point of each half of this ED distribition
        "This scores the location of the local maxima scaled by the global maximum"
        leftFocus, rightFocus = Template.fit_focus(y)
        
        # Combine the correlation and focus measurements
        leftScore = np.abs(leftCorr * leftFocus)
        if leftCorr < 0 or leftFocus < 0:
            leftScore = -leftScore
        
        rightScore = np.abs(rightCorr * rightFocus)
        if rightCorr < 0 or rightFocus < 0:
            rightScore = -rightScore
        
        # Penalise a side if it has a low magnitude difference from min->max
        centre = len(y) // 2
        globalMin = np.min(y)
        globalMax = np.max(y)
        
        leftY = y[:centre]
        leftScalingFactor = Critic.penalty(leftY, globalMin, globalMax, left=True)
        
        rightY = y[centre:]
        rightScalingFactor = Critic.penalty(rightY, globalMin, globalMax, left=False)
        
        # Scale and return the scores
        if leftScore > 0: # don't apply scaling to a value that's already negative
            leftScore = leftScore * leftScalingFactor
        
        if rightScore > 0:
            rightScore = rightScore * rightScalingFactor
        
        score = (leftScore + rightScore) / 2
        
        return score, leftWidth, rightWidth
    
    @staticmethod
    def scores_to_strength(scores):
        '''
        Categorises multiple line-fit scores, each ranging from from 0->1, into one of several
        categories for easier assessment of the statistical outcome of a simulated segregation
        experiment.
        
        Parameters:
            scores -- a numpy array with shape (num_pop_sizes, num_bootstraps) containing float
                      values from 0 to 1 (inclusive)
        Returns:
            strengths -- a numpy array with shape (num_pop_sizes, 4) where the bootstrap replicated
                         scores have been summarised into one of four different categories ordered
                         as: [none, weak, moderate, strong]
        '''
        strengths = []
        for popSizeArray in scores:
            noSignal = sum((popSizeArray < Critic.WEAK_SCORE) | (np.isnan(popSizeArray)))
            weakSignal = sum((popSizeArray >= Critic.WEAK_SCORE) & (popSizeArray < Critic.MID_SCORE))
            moderateSignal = sum((popSizeArray >= Critic.MID_SCORE) & (popSizeArray < Critic.STRONG_SCORE))
            strongSignal = sum(popSizeArray >= Critic.STRONG_SCORE)
            strengths.append([noSignal, weakSignal, moderateSignal, strongSignal])
        return np.array(strengths)
    
    def run(self, configuration):
        '''
        Parameters:
            configuration -- a Configuration object recording the simulation variable combinations
        '''
        for (popBalance, phenotypeError), popSizes in configuration:
            if popSizes is None or len(popSizes) == 0:
                continue
            
            # Obtain the Spreadsheet this configuration will have results stored within
            spreadsheet = Spreadsheet(self.storageDir, popBalance, phenotypeError, popSizes)
            spreadsheet.load() # runs some internal validations of popBalance, phenotypeError, and popSizes
            
            # See if this configuration has been completely processed
            if hasattr(spreadsheet, "scores1"): # .scores1 is set if we have run this at least partially
                expectedAttr = [ f"scores{x[2:]}" for x in spreadsheet.get_unfixed_attributes("ed") ]
                assert len(expectedAttr) > 0, "sanity check, if we get here .ed* should be set"
                if expectedAttr == spreadsheet.get_unfixed_attributes("scores"):
                    continue
            
            # If not, iterate through each QTL to generate its results
            for i, qtlED in enumerate(spreadsheet.get_ed()):
                numPopSizes, numBootstraps, numVariants = qtlED.shape
                
                # Assess each replication of this parameter combination
                scores = []
                leftWidths = []
                rightWidths = []
                for popSizeArray in qtlED:
                    for replicateArray in popSizeArray:
                        score, leftWidth, rightWidth = Critic.score(replicateArray)
                        scores.append(score)
                        leftWidths.append(leftWidth)
                        rightWidths.append(rightWidth)
                
                # Reshape scores into an array that matches the QTL shape
                scores = np.stack(np.split(np.array(scores), numPopSizes)) # shape = (popsize, bootstraps)
                leftWidths = np.stack(np.split(np.array(leftWidths), numPopSizes))
                rightWidths = np.stack(np.split(np.array(rightWidths), numPopSizes))
                
                # Also summarise scores as the signal strength
                strengths = Critic.scores_to_strength(scores)
                
                # Store results in Spreadsheet
                setattr(spreadsheet, f"scores{i+1}", scores)
                setattr(spreadsheet, f"leftWidths{i+1}", leftWidths)
                setattr(spreadsheet, f"rightWidths{i+1}", rightWidths)
                setattr(spreadsheet, f"strengths{i+1}", strengths)
            
            # Store the results into the Spreadsheet
            spreadsheet.save()
    
    def __repr__(self):
        return "<Critic object;storageDir='{0}';qtlRanges={1}>".format(
            self.storageDir,
            self.qtlRanges
        )
