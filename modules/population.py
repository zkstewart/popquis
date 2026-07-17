# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

import numpy as np

from npy_append_array import NpyAppendArray

class Population:
    '''
    Wrapper around the npy-append-array functionalities to allow for progressive growth
    of a population of simulated individuals without memory limitation and with later
    memory mapped access to individuals.
    
    Attributes:
        data -- None, or after using self.load(), a memory-mapped npy file handle
    Methods:
        store -- stores an array
        load -- memory maps the npy file to enable retrieval of stored arrays
        retrieve -- extracts stored arrays by their index
    '''
    def __init__(self, fileName):
        self.fileName = fileName
        self.data = None
        self.isPopulation = True # object type validator
    
    @property
    def fileName(self):
        return self._fileName
    
    @fileName.setter
    def fileName(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Population object expects fileName to be a str, not '{type(value).__name__}'")
        
        value = os.path.abspath(value)
        if os.path.isfile(value):
            pass
        elif os.path.exists(value):
            raise FileNotFoundError(f"Population object was given the fileName '{value}' which appears to " +
                                    "exist but not be a file? Move or rename whatever resides here and try again.")
        else:
            parentDir = os.path.dirname(value)
            if os.path.isdir(parentDir):
                pass
                #open(value, "w").close() # touch file
            elif not os.path.exists(parentDir):
                raise NotADirectoryError(f"Population object cannot create fileName '{value}' since its parent " +
                                         f"location '{parentDir}' does not exist. Create this location first.")
            else:
                raise NotADirectoryError(f"Population object cannot create fileName '{value}' since its parent " +
                                         f"location '{parentDir}' is not a directory? Move or rename whatever " +
                                         "resides here, create the parent location as a directory, then try again.")
        
        self._fileName = value
    
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
    
    @property
    def shape(self):
        if self.data is None:
            return None
        return self.data.shape
    
    def add(self, array):
        '''
        Parameters:
            array -- an array representing one simulated individual
        '''
        ## TBD: need to validate that input is a single individual? or just trust?
        with NpyAppendArray(self.fileName, delete_if_exists=False) as npaa:
            npaa.append(array)
    
    def load(self):
        if os.path.isfile(self.fileName):
            self.data = np.load(self.fileName, mmap_mode="r")
    
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
        if not (isinstance(indices, list) or isinstance(indices, tuple)):
            raise TypeError(f"Population can only receive a list or tuple, not '{type(indices).__name__}'")
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
