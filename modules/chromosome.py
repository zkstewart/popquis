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
        ploidy -- an integer indicating the number of chromosome copies.
        numVariants -- an integer indicating how many variants/genotypes/alleles/SNPs this chromosome represents.
        array -- a numpy array with shape (1, numVariants, ploidy) which is the fundamental data structure
                 this Class encapsulates.
    Methods:
        generate -- uses parameter values to set self.array
    '''
    def __init__(self, chromID, positions, genotypes, chromMap):
        self.chromID = chromID
        self.positions = positions
        self.genotypes = genotypes
        self.chromMap = chromMap
        self.strands = None
        self.generate()
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
    def positions(self):
        return self._positions
    
    @positions.setter
    def positions(self, value):
        if not (isinstance(value, list) or isinstance(value, tuple)):
            raise TypeError(f"positions must be a list, not '{type(value).__name__}'")
        if any([ not isinstance(x, int) for x in value ]):
            raise TypeError(f"All values in positions list must be an integer")
        # there are more validations we could do, but the inputs to this Class object should be highly validated already
        self._positions = value
    
    @property
    def genotypes(self):
        return self._genotypes
    
    @genotypes.setter
    def genotypes(self, value):
        if not (isinstance(value, list) or isinstance(value, tuple)):
            raise TypeError(f"genotypes must be a list, not '{type(value).__name__}'")
        if any([ not hasattr(x, "isGenotype") for x in value ]):
            raise TypeError(f"All values in genotypes list must be a Genotype object")
        self._genotypes = value
    
    def generate(self):
        # Validate compatibility of positions and genotypes
        if len(self.positions) != len(self.genotypes):
            raise ValueError("Chromosome object has differing lengths in the positions and genotypes lists")
        
        # Assign each genotype to its closest physical position
        self.ploidy = self.genotypes[0].ploidy
        distances = [ [ abs(x - y) for y in self.positions ] for x in self.chromMap.df["bp"] ]
        closest = [ x.index(min(x)) for x in distances ]
        self.array = np.array([ self.genotypes[genotypeIndex].alleles for genotypeIndex in closest ]).reshape(1, len(closest), self.ploidy)
        self.numVariants = len(closest)
    
    def __repr__(self):
        return "<Chromosome object;chromID='{0}';ploidy={1};numVariants={2}>".format(
            self.chromID,
            self.ploidy,
            self.numVariants
        )
