# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

class Chromosome:
    '''
    Args:
        chromID -- a string identifying the chromosome.
        positions -- a list of integers indicating the location of each QTL in basepairs.
        genotypes -- a list of Genotype objects ordered to correspond to the given positions.
        chromMap -- a ChromosomeMap object corresponding to this chromosome.
    Properties:
        variants -- an integer indicating how many variants/genotypes/alleles/SNPs this chromosome represents.
        ploidy -- an integer indicating the number of chromosome copies.
        array -- a numpy array with shape (variants, ploidy)
    '''
    def __init__(self, chromID, positions, genotypes, chromMap):
        self.chromID = chromID
        self.chromMap = chromMap
        self._generate(positions, genotypes)
        self.isChromosome = True # object type validator
    
    @property
    def chromID(self):
        return self._chromID
    
    @chromID.setter
    def chromID(self, value):
        if not isinstance(value, str):
            raise TypeError(f"chromID must be a str, not '{type(value).__name__}'")
        if value == "":
            raise ValueError("chromID must not be blank")
        self._chromID = value
    
    @property
    def variants(self):
        variants, ploidy = self.array.shape
        return variants
    
    @property
    def ploidy(self):
        variants, ploidy = self.array.shape
        return ploidy
    
    @property
    def shape(self):
        return self.array.shape
    
    def _generate(self, positions, genotypes):
        # Validate compatibility of positions and genotypes
        if len(positions) != len(genotypes):
            raise ValueError("Cannot create Chromosome from mismatching positions and genotypes lists")
        
        # Assign each genotype to its closest physical position
        distances = [ [ abs(x - y) for y in positions ] for x in self.chromMap.df["bp"] ]
        closest = [ x.index(min(x)) for x in distances ]
        self.array = np.array([ genotypes[genotypeIndex].alleles for genotypeIndex in closest ])
    
    def __len__(self):
        return len(self.array)
    
    def __repr__(self):
        return "<Chromosome object;chromID='{0}';variants={1};ploidy={2}>".format(
            self.chromID,
            self.variants,
            self.ploidy
        )
