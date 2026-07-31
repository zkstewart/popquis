# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

class Genome:
    '''
    Properties:
        chromIDs -- a list of strings indicating which Chromosome objects this Class has ingested.
        ploidy -- an integer indicating the number of chromosome copy numbers.
        variants -- an integer indicating how many variants/genotypes/alleles/SNPs this genome represents.
        array -- a numpy array with shape (variants, ploidy) concatenating all of the
                 Chromosome.array objects that this one has ingested.
    '''
    def __init__(self):
        self.chromosomes = {}
        self._ploidy = None
        self._array = None
        self._wasUpdated = False
        self.isGenome = True # object type validator
    
    @property
    def chromIDs(self):
        return list(self.chromosomes.keys())
    
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
    
    @property
    def variants(self):
        if self.array is not None:
            return len(self.array)
        else:
            return None
    
    @property
    def array(self):
        # Create the chromosome-spanning array on demand
        if self._array is None or self._wasUpdated:
            _array = None
            for chromID, chromosome in self.chromosomes.items():
                if _array is None:
                    _array = chromosome.array.copy()
                else:
                    _array = np.vstack((_array, chromosome.array.copy()))
            self._array = _array
            self._wasUpdated = False
        return self._array
    
    def __setitem__(self, key, value):
        # Validate object type
        if not hasattr(value, "isChromosome"):
            raise TypeError(f"Genome object can only receive a Chromosome object, not '{type(chromosomeObj).__name__}'")
        
        # Set/validate ploidy compatibility
        self.ploidy = value.ploidy
        
        # Make sure we haven't ingested this chromosome already
        if key in self.chromosomes:
            raise KeyError(f"'{key}' already stored in this Genome")
        
        self.chromosomes[key] = value
        self._wasUpdated = True
    
    def __getitem__(self, key):
        if key not in self.chromosomes:
            raise KeyError(f"'{key}' not in this Genome")
        return self.chromosomes[key]
    
    def __iter__(self):
        for value in self.chromosomes.values():
            yield value
    
    def __repr__(self):
        return "<Genome object;chromIDs={0};variants={1};ploidy={2}>".format(
            self.chromIDs,
            self.variants,
            self.ploidy
        )
