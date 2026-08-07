# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

import numpy as np

class RandomNumberGenerator:
    '''
    Wrapper around the numpy.random.default_rng method of generating random numbers.
    
    Args:
        seed -- an integer setting the initial seed for number generation
    Methods:
        generate_random_indices -- produces a numpy array of random integers
                                   without replacement.
    '''
    def __init__(self, seed):
        self.seed = seed
        self._rngenerator = np.random.default_rng(seed=seed)
    
    def generate_random_indices(self, rangeMax, numberOfIndices):
        '''
        Encapsulates np.random.default_rng().choice with replace=False
        
        Parameters:
            rangeMax -- an integer giving the upper boundary of indices to generate (exclusive);
                        in other words, treat this like a normal range(0, upper)
            numberOfIndices -- an integer giving the number of indices to generate
        '''
        return self._rngenerator.choice(rangeMax, size=numberOfIndices, replace=False)
    
    def integers(self, low, high):
        '''
        Encapsulates np.random.default_rng().integers
        
        Parameters:
            low -- minimum integer value to generate (inclusive)
            high -- maximum integer vlaue to generate (exclusive)
        '''
        return self._rngenerator.integers(low, high)
    
    def gamma(self, shape, scale):
        '''
        Encapsulates np.random.default_rng().gamma
        
        Parameters:
            shape -- tbd
            scale -- tbd
        '''
        return self._rngenerator.gamma(shape=shape, scale=scale)

class Calculator:
    '''
    Namespace for calculations needed for evaluating segregation statistics.
    
    Methods:
        evenly_spaced_sampling -- assuming an array of size=n, returns a user-specified
                                  number of indices that are evenly spaced within range(0, n)
        counter -- numpy-style version of collections.Counter
        euclidean_distance -- receives two arrays of alleles and calculates the ED
                              segregation statistic
    '''
    @staticmethod
    def evenly_spaced_sampling(length, resultLength):
        '''
        Receives a length value, which may represent the length of an iterable,
        alongside the intended length of an output iterable. Based on these values,
        an output list will be generated with the index values to obtain from
        the original list to produce an approximately evenly divided output.
        
        Params:
            length -- an integer value, possibly derived from a list length as example.
            resultLength -- an integer value for the desired number of output indices
        Returns:
            indices -- a numpy array of integer values giving the array indices to obtain
                       for an evenly spaced sampling outcome
        '''
        if not isinstance(length, (int, np.integer)):
            raise TypeError(f"Spaced sampling requires an integer-type 'length' value, not '{type(length).__name__}'")
        if not isinstance(resultLength, (int, np.integer)):
            raise TypeError(f"Spaced sampling requires an integer-type 'resultLength' value, not '{type(resultLength).__name__}'")
        if length < resultLength:
            raise ValueError(f"Spaced sampling requires resultLength ({resultLength}) to be <= length ({length})")
        if length == 0:
            raise ValueError(f"Spaced sampling from an iterable with length 0 is nonsensical")
        return np.linspace(0, length-1, resultLength).astype(int) # length-1 to make output a 0-based index
    
    @staticmethod
    def counter(array):
        '''
        Parameters:
            array -- a 1D numpy array of integers
        Returns:
            unique -- a numpy array listing the uniquely occurring np.int64 values
            counts -- a numpy array of np.int64 counts ordered to match the keys
                      listed by 'unique'
        '''
        unique, counts = np.unique(array, return_counts=True)
        return unique.astype(np.int64), counts.astype(np.int64)
    
    @staticmethod
    def euclidean_distance(array1, array2, power=4):
        '''
        Parameters:
            array1 / array2 -- numpy arrays with alleles provided as integers in shape akin
                               to: (num_variants, num_alleles). The calculation allows for
                               num_alleles to differ, but num_variants must be consistent
                               or you should expect a truncated output. For efficiency
                               reasons this function will not attempt to catch any
                               shape incompatibilities.
            power -- an integer giving the power to raise each Euclidean distance value to;
                     default and recommended value is 4
        Returns:
            edist -- a numpy array of float values giving the Euclidean distance of each variant
                     after raising to the power value
        '''
        distances = []
        for alleles1, alleles2 in zip(array1, array2):
            a1Unique, a1Counts = Calculator.counter(alleles1)
            a1Dict = dict(zip(a1Unique, a1Counts))
            a1Num = len(alleles1)
            
            a2Unique, a2Counts = Calculator.counter(alleles2)
            a2Dict = dict(zip(a2Unique, a2Counts))
            a2Num = len(alleles2)
            
            allUnique = np.union1d(a1Unique, a2Unique)
            for key in allUnique:
                if not key in a1Dict:
                    a1Dict[key] = np.int64(0)
                if not key in a2Dict:
                    a2Dict[key] = np.int64(0)
            
            edist = np.sqrt(np.sum([
                ((a1Dict[allele] / a1Num) - (a2Dict[allele] / a2Num))**2
                for allele in allUnique
            ]))
            distances.append(np.power(edist, power))
        
        return np.array(distances)
    
    @staticmethod
    def interpolate(xprev, xnext, yprev, ynext):
        '''
        Calculate the slope between two points (xprev, yprev) and (xnext, ynext)
        and return the interpolated y values spanning the space between.
        
        Parameters:
            xprev -- the x value of the previous point
            xnext -- the x value of the next point
            yprev -- the y value of the previous point
            ynext -- the y value of the next point
        Returns:
            y -- the interpolated y values
        '''
        return np.linspace(yprev, ynext, num=xnext-xprev)
    
    @staticmethod
    def interpolate_popsize_milestones(plotDF, bootstraps):
        '''
        Assesses data being plotted to locate milestones to mark. These milestones correspond to the lowest
        population size necessary to achieve a certain confidence or likelihood that an experiment would have
        at least as much "success" with QTL prediction, as defined by the signal strength.
        
        Interpolation occurs since not all parameter combinations are evenly dispersed over a range, and we would
        like to give the most granular estimate possible.
        
        Returns:
            weak95 -- an integer of the pop. size which first has >=95% samples with at least a weak signal
            mid95 -- an integer of the pop. size which first has >=95% samples with at least a mid/intermediate signal
            strong95 -- an integer of the pop. size which first has >=95% samples with at least a strong signal
        '''
        CUTOFF = 0.95
        
        THRESHOLD_COLUMNS = ["atleast_weak", "atleast_mid", "strong_signal"]
        
        # Figure out the number of replications needed to reach the threshold
        thresholdPoint = np.ceil(bootstraps * CUTOFF)
        
        # Quantify the number of replications that support each strength category
        plotDF["atleast_weak"] = plotDF["weak_signal"] + plotDF["moderate_signal"] + plotDF["strong_signal"]
        plotDF["atleast_mid"] = plotDF["moderate_signal"] + plotDF["strong_signal"]
        "plotDF['strong'] already represents 'atleast_strong'"
        
        # Get interpolated milestone counts
        milestones = { key: None for key in THRESHOLD_COLUMNS }
        x = plotDF["pop_size"].tolist()
        for column in THRESHOLD_COLUMNS:
            y = plotDF[column].tolist()
            for xindex in range(1, len(x)):
                xprev, xnext = x[xindex - 1], x[xindex]
                yprev, ynext = y[xindex - 1], y[xindex]
                interpolatedY = Calculator.interpolate(xprev, xnext, yprev, ynext)
                
                # See if a milestone point occurs within this interpolated region
                condition = interpolatedY >= thresholdPoint
                if np.any(condition):
                    interpolatedIndex = np.argmax(condition)
                    milestonePopSize = xprev + interpolatedIndex
                    milestones[column] = milestonePopSize
                    break
        
        return milestones
