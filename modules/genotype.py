import os
import sys

from itertools import combinations

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from errors import InvalidGenotypeError, IncompatibleGenotypeError

class Genotype:
    '''
    Properties:
        raw -- the original string used to instantiate a Genotype object
        alleles -- a list of integers representing the alleles that compose this genotype
        unique -- returns set(alleles) for easier handling of the unique alleles that compose this genotype
        ploidy -- returns len(alleles) which indicates the ploidy of this genotype
    Methods:
        cross -- produces a list of potential Genotype instance outputs that could be obtained by crossing
                 two parental Genotypes together
    '''
    def __init__(self, value):
        self.raw = value
        self.alleles = value
        self.isGenotype = True # object type validator
    
    @property
    def alleles(self):
        return self._alleles
    
    @alleles.setter
    def alleles(self, value):
        # Validate type
        if not isinstance(value, str):
            raise ValueError("Genotype class expects a string input")
        
        # Validate delimiter
        if "|" in value:
            raise InvalidGenotypeError(f"Genotype '{value}' has a '|' delimiter to indicate phased alleles, " +
                                       "but phasing is not handled by popquis. Explicitly use the '/' unphased " +
                                       "delimiter instead")
        if not "/" in value:
            raise InvalidGenotypeError(f"Genotype '{value}' lacks a '/' delimiter which indicates an invalid "
                                       "genotype format. If you intended to specify a haploid genotype, note " +
                                       "that popquis does not handle haploidy.")
        
        # Validate allele
        splitValues = value.split("/")
        accepted = []
        for svalue in splitValues:
            try:
                intvalue = int(svalue)
            except ValueError:
                raise InvalidGenotypeError(f"Genotype '{value}' has an allele '{svalue}' which does not " +
                                           "appear to be a number. Make sure this is a plain integer.")
            if len(str(intvalue)) != len(svalue):
                raise InvalidGenotypeError(f"Genotype '{value}' has an allele '{svalue}' which looks " +
                                           f"different after being converted to the integer '{intvalue}'; " +
                                           "this likely means the number has a zero prefix which has been " +
                                           "ignored during integer conversion, and suggests that your input " +
                                           "may be incorrectly specified.")
            accepted.append(intvalue)
        
        # Store validated alleles
        self._alleles = accepted
    
    @property
    def unique(self):
        return set(self.alleles)
    
    @property
    def ploidy(self):
        return len(self._alleles)
    
    def cross(self, otherGenotype):
        '''
        Cross this Genotype (parent 1) with another Genotype (parent 2) to generate
        Genotype possibilities of offspring.
        
        Parameters:
            otherGenotype -- another Genotype object
        Returns:
            offspringGenotypes -- a non-redundant list of Genotype objects representing potential
                                  offspring genotypes
        '''
        # Validate type
        if not hasattr(otherGenotype, "isGenotype") and otherGenotype.isGenotype:
            raise TypeError("Input object is not Genotype; cannot cross")
        
        # Validate object compatibility
        if self.ploidy != otherGenotype.ploidy:
            raise IncompatibleGenotypeError(f"Cannot cross '{self.raw}' with '{otherGenotype.raw}' as their ploidies do not match")
        if self.ploidy % 2 != 0:
            raise IncompatibleGenotypeError(f"Cannot cross '{self.raw}' with any other genotype as it has odd-numbered ploidy")
        
        # Identify non-redundant genotype combinations of parents
        numAllelesFromParent = int(self.ploidy/2) # each parent passes along half of its alleles
        possibleGTs = set()
        for alleles1 in combinations(self.alleles, numAllelesFromParent):
            for alleles2 in combinations(otherGenotype.alleles, numAllelesFromParent):
                possibleGTs.add(alleles1 + alleles2)
        
        # Emit Genotype instances for each possible genotype
        return [ Genotype("/".join(map(str, x))) for x in possibleGTs ]
    
    def __eq__(self, other):
        if not hasattr(other, "isGenotype") and other.isGenotype:
            return False
        
        return (self.ploidy == other.ploidy) and (self.unique == other.unique)
    
    def __repr__(self):
        return "<Genotype object;raw='{0}';alleles={1};ploidy={2}>".format(
            self.raw,
            self.alleles,
            self.ploidy
        )
