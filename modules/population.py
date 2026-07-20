# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.appendarray import AppendArray

class Population(AppendArray):
    '''
    Implementation of AppendArray to allow for progressive growth of a population of
    simulated individuals without memory limitation and with later memory mapped
    access to the array of individuals.
    
    Args:
        fileName -- a string indicating the location of a pre-existing npy file,
                    or to a location that is writeable to create a new npy file
    Attributes:
        data -- None, or after using self.load(), a memory-mapped npy file handle
    Methods:
        store -- stores an array of an individual's simulated variant genotypes
        load -- memory maps the npy file to enable retrieval of stored arrays
        retrieve -- extracts stored arrays by their index
    '''
    def __init__(self, fileName):
        super().__init__(fileName)
        self.isPopulation = True # object type validator
    
    @property
    def individuals(self):
        if self.data is None:
            return None
        numIndividuals, _, _ = self.shape
        return numIndividuals
    
    @property
    def variants(self):
        if self.data is None:
            return None
        _, numVariants, _ = self.shape
        return numVariants
    
    @property
    def ploidy(self):
        if self.data is None:
            return None
        _, _, ploidy = self.shape
        return ploidy
    
    def retrieve(self, indices):
        '''
        Parameters:
            indices -- an iterable of integers giving the index (i.e., 0-based individual number)
                       of one or more simulatd individual arrays to return
        Returns:
            array -- a stacked numpy array with shape (num_individuals, num_variants, ploidy)
        '''
        if self.data is None:
            raise ValueError("Population must be loaded before .retrieve is functional")
        try:
            for value in indices:
                int(value)
                break
        except:
            raise TypeError("Input to Population.retrieve() must be an iterable of integer-convertible values")
        
        return np.stack(self.data[indices])
    
    def __repr__(self):
        try:
            numIndividuals, numVariants, ploidy = self.shape
        except:
            numIndividuals, numVariants, ploidy = None, None, None
        
        return "<Population object;numIndividuals={0};numVariants={1};ploidy={2}>".format(
            numIndividuals,
            numVariants,
            ploidy
        )
