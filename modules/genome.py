# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

class Genome:
    '''
    Properties:
        chromIDs -- a set of strings indicating which Chromosome objects this Class has ingested.
        ploidy -- an integer indicating the number of chromosome copy numbers.
        numVariants -- an integer indicating how many variants/genotypes/alleles/SNPs this genome represents.
        array -- a numpy array with shape (1, numVariants, ploidy) which is the fundamental data structure
                 this Class encapsulates.
    Methods:
        add -- ingests a Chromosome object and creates this Genome object out of its composition
    '''
    def __init__(self):
        self.chromIDs = set()
        self._ploidy = None
        self.array = None
        self.numVariants = 0
        self.isGenome = True # object type validator
    
    @property
    def ploidy(self):
        return self._ploidy
    
    @ploidy.setter
    def ploidy(self, value):
        if not isinstance(value, int):
            raise TypeError(f"ploidy must be an int, not '{type(value).__name__}'")
        if self._ploidy is None:
            self._ploidy = value
        elif value != self._ploidy:
            raise ValueError(f"Genome object can only receive a Chromosome if it has ploidy={self._ploidy}")
    
    def add(self, chromosomeObj):
        # Validate object type
        if not hasattr(chromosomeObj, "isChromosome"):
            raise TypeError(f"Genome object can only receive a Chromosome object, not '{type(chromosomeObj).__name__}'")
        
        # Set/validate ploidy compatibility
        self.ploidy = chromosomeObj.ploidy
        
        # Make sure we haven't ingested this chromosome already
        if chromosomeObj.chromID in self.chromIDs:
            raise ValueError(f"Genome object has already had '{chromosomeObj.chromID}' added into it.")
        
        # Store the information
        if self.array is None:
            self.array = chromosomeObj.array.copy()
        else:
            self.array = np.hstack((self.array, chromosomeObj.array))
        
        # Increment supplemental information
        self.numVariants += chromosomeObj.numVariants
        self.chromIDs.add(chromosomeObj.chromID)
    
    def __repr__(self):
        return "<Genome object;chromIDs={0};ploidy={1};numVariants={2}>".format(
            self.chromIDs,
            self.ploidy,
            self.numVariants
        )
