# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.chromosome import Chromosome
from modules.chromosomemap import ChromosomeMap
from modules.genome import Genome
from modules.genomemap import GenomeMap
from modules.population import Population
from modules.simulator import ChromaxSimulator

class Breeder:
    '''
    Consolidates the GenomeMap and Genome classes with the QTL positions in order to
    enable subsequent breeding simulation to occur.
    
    Attributes:
        genomeMap -- a GenomeMap object which is a composition of ChromosomeMap objects
        parent1 -- a Genome object which is a composition of one or more Chromosome objects,
                   providing an array of parent1's simulated genotype
        parent2 -- as above, but for parent2
        markerIndices -- a list of integers giving the index (row) of the marker (QTL) with
                         respect to the GenomeMap being used for simulation
        markerAlleles -- a list with sublists matched up with markerIndices, where sublists
                         contain integers giving the alleles (genotype) associated with each
                         marker
    Methods:
        establish -- receives the necessary input data to construct the GenomeMap and Genome
                     of the two parents which will be crossed to generate segregating progeny
        produce_progeny -- makes use of the GenomeMap and Genome of each parent to produce
                           two segregating Populations of progeny
    '''
    def __init__(self):
        self.parent1 = None
        self.parent2 = None
        self.isBreeder = True # object type validator
    
    def establish(self, positions, genotypes, cmMbp, snpMbp=1000, edgeBp=5000000):
        '''
        Creates a genetic map for each chromosome alongside modelling the parental genomes
        for those chromosomes according to the proposed QTL positions and genotypes.
        
        Parameters:
            positions -- a list of tuples as derived from modules.parsing.parse_linkage() indicating
                         the location of each QTL as: ('chromID', positionInBp)
            genotypes -- a list of lists with structure like:
                         [
                             [Genotype1, Genotype2, Genotype3], # parent1, parent2, offspring
                             [Genotype4, Genotype5, Genotype6],
                         ]; this should match up with the positions list
            cmMbp -- a float value giving the centiMorgan per Mbp.
            snpMbp -- an integer giving the approximate number of SNPs to be spaced evenly
                      across each Mbp of genome length; default is 1000.
            edgeBp -- an integer giving the length of genome sequence (in bp) that should
                      exist as a buffer/edge around each chromosome.
        '''
        # Relate QTL positions and genotypes by chromosome
        chromosomeData = {}
        for (chromID, positionInBp), genotype in zip(positions, genotypes):
            chromosomeData.setdefault(chromID, [])
            chromosomeData[chromID].append((positionInBp, genotype))
        
        # Iterate through data for each chromosome to establish the initial breeding set-up
        self.genomeMap = GenomeMap()
        self.parent1 = Genome()
        self.parent2 = Genome()
        offspringGenotypes = []
        for chromID, value in chromosomeData.items():
            # Establish marker details
            chromPositions, chromGenotypes = zip(*value)
            chromLength = (edgeBp * 2) + (chromPositions[-1] + 1) # positions are ordered so get the last; 0-based so plus one
            chromPositions = [ x + edgeBp for x in chromPositions ] # marker positions need to update w/r/t the edge buffer length
            
            # Establish the genetic map for this chromosome
            chromosomeMap = ChromosomeMap(chromID, chromLength, cmMbp, snpMbp, chromPositions)
            self.genomeMap[chromID] = chromosomeMap
            
            # Establish the parental genomes for this chromosome
            parent1Genotypes = [ x[0] for x in chromGenotypes ]
            parent1 = Chromosome(chromID, chromPositions, parent1Genotypes, chromosomeMap)
            self.parent1[chromID] = parent1
            
            parent2Genotypes = [ x[1] for x in chromGenotypes ]
            parent2 = Chromosome(chromID, chromPositions, parent2Genotypes, chromosomeMap)
            self.parent2[chromID] = parent2
            
            # Hold onto the markers for segregating offspring identification
            offspringGenotypes += [ x[2] for x in chromGenotypes ]
        
        # Relate the offspring marker genotypes to the underlying GenomeMap
        "Doing this gives us the indices spanning across all chromosomes that compose the GenomeMap"
        self.markerIndices = []
        self.markerAlleles = []
        for genotype, rowTuple in zip(offspringGenotypes, self.genomeMap.markers.itertuples()):
            self.markerIndices.append(rowTuple.Index)
            self.markerAlleles.append(genotype.alleles)
    
    def produce_progeny(self, locations, combinationEvaluator, batchSize=1000, minimumGroupSize=10000, seed=1234):
        '''
        Parameters:
            locations -- a Locations object with .group1Npy and .group2Npy attributes to indicate the file
                         names for the Population objects that store simulated data for each progeny group
            combinationEvaluator -- a Combination object with the .evaluate() method capable of receiving
                                    a list of marker genotypes and returning a boolean indicating whether
                                    the sample belongs in group1 (True) or group2 (False)
            batchSize -- an integer controlling how many individuals are generated by Chromax at a time.
                         In practice, this serves to reduce the memory consumption of the program by producing
                         progeny iteratively in batches.
            minimumGroupSize -- an integer indicating the population size that must be reached for each of
                                group1 and group2, before simulation can terminate. Once a group meets or
                                exceeds this value, no more individuals will be stored for that group.
            seed -- an integer to specify a random number seed used by Chromax when generating progeny
        '''
        # Create a storage container for each population / load an existing and ongoing population build
        group1 = Population(locations.group1Npy)
        group1.load()
        group2 = Population(locations.group2Npy)
        group2.load()
        
        # Simulate progeny in batches to attain a pre-requisite population size for each group
        simulator = ChromaxSimulator(self.parent1, self.parent2, self.genomeMap)
        while (group1.individuals is None or group1.individuals < minimumGroupSize) or (group2.individuals is None or group2.individuals < minimumGroupSize):
            f1 = simulator.cross(batchSize=batchSize)
            
            # Segregate progeny based on combined QTL inheritance
            numGroup1 = 0
            numGroup2 = 0
            for offspring in f1:
                offspringAlleles = [ offspring[index] for index in self.markerIndices ]
                offspringMarkers = [ np.array_equal(offspringAllele, markerAllele) for offspringAllele, markerAllele in zip(offspringAlleles, self.markerAlleles) ]
                isGroup1 = combinationEvaluator.evaluate(offspringMarkers)
                if isGroup1:
                    numGroup1 += 1
                    if (group1.individuals is None) or (group1.individuals < minimumGroupSize):
                        group1.add(offspring.reshape(1, *offspring.shape)) # must add a dimension so AppendArray handles correctly
                else:
                    numGroup2 += 1
                    if (group2.individuals is None) or (group2.individuals < minimumGroupSize):
                        group2.add(offspring.reshape(1, *offspring.shape))
            
            # Raise an error if we failed to produce progeny for either group
            if numGroup1 == 0 or numGroup2 == 0:
                raise Exception(f"After simulating {batchSize} individuals, {numGroup1} had inherited the " +
                                f"QTL combination '{combinationEvaluator.raw}' and were categorised into group1, with " +
                                f"{numGroup2} lacking this QTL combination and hence being categorised into group2. " + 
                                "Since one of these groups has 0 individuals, the QTL combination for one of these groups " +
                                "may be virtually impossible to occur in a simulated individual. popquis will exit now to " +
                                "prevent what might otherwise become an endless process. You should consider " + 
                                "simplifying the QTL combination to avoid this issue.")
            
            # Provide some logging information
            "This also helps to figure out what the inheritance ratio is for the QTL combination"
            print(f"# Simulated a batch of {batchSize} individuals: {numGroup1} in group1 and {numGroup2} in group2")
            
            # Re-load data to enable checking of group size
            group1.load()
            group2.load()
    
    def __repr__(self):
        return "<Breeder object;genomeMap={0};parent1={1};parent2={2}>".format(
            self.genomeMap,
            self.parent1,
            self.parent2
        )
