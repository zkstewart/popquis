#!/usr/bin/env python3
# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.breeder import Breeder
from modules.experiment import Configuration, Coordinator
from modules.locations import Locations
from modules.parsing import parse_qtl_encoding
from modules.population import Population

testDir = os.path.dirname(os.path.abspath(__file__))
tmpDir = os.path.join(testDir, "tmp")

def cleanup():
    os.makedirs(tmpDir, exist_ok=True)
    locations = Locations(tmpDir)
    if os.path.isfile(locations.group1Npy):
        os.unlink(locations.group1Npy)
    if os.path.isfile(locations.group2Npy):
        os.unlink(locations.group2Npy)

def tally_crossovers(group):
    '''
    Note that this process incorrectly identifies the transition sites between
    QTLs as almost always being a crossover, but it's just the parent genotype
    that changes for a different chromosome. That is why two parts of the array
    jump from the average of ~50-70 up to ~1000.
    '''
    crossovers = []
    for individual in group.data:
        lastVariant = None
        for i, variant in enumerate(individual):
            if lastVariant is not None:
                if lastVariant[0] != variant[0] or lastVariant[1] != variant[1]:
                    crossovers.append(i)
            
            lastVariant = variant
    
    return np.bincount(crossovers)

def chromax_test(genotypes, combinationEvaluator, positions, locations,
                 cmMbp, snpMbp, edgeBp, minimumGroupSize, seed):
    # Produce the Chromax-based breeding population
    breeder = Breeder()
    breeder.establish(positions, genotypes, cmMbp, snpMbp=snpMbp, edgeBp=edgeBp)
    breeder.produce_progeny(locations, combinationEvaluator,
                            minimumGroupSize=minimumGroupSize,
                            seed=1234,
                            quiet=True,
                            simulatorToUse="chromax")
    
    # Survey the crossover points for each group
    group1 = Population(locations.group1Npy)
    group1.load() # data shape == (numIndividuals, numVariants, ploidy)
    group1Crossover = tally_crossovers(group1)
    
    group2 = Population(locations.group2Npy)
    group2.load() # data shape == (numIndividuals, numVariants, ploidy)
    group2Crossover = tally_crossovers(group2)
    
    return group1Crossover, group2Crossover

def meiosis_test(genotypes, combinationEvaluator, positions, locations,
                 cmMbp, snpMbp, edgeBp, minimumGroupSize, seed):
    # Produce the Chromax-based breeding population
    breeder = Breeder()
    breeder.establish(positions, genotypes, cmMbp, snpMbp=snpMbp, edgeBp=edgeBp)
    breeder.produce_progeny(locations, combinationEvaluator,
                            minimumGroupSize=minimumGroupSize,
                            seed=1234,
                            quiet=True,
                            simulatorToUse="popquis")
    
    # Survey the crossover points for each group
    group1 = Population(locations.group1Npy)
    group1.load() # data shape == (numIndividuals, numVariants, ploidy)
    group1Crossover = tally_crossovers(group1)
    
    group2 = Population(locations.group2Npy)
    group2.load() # data shape == (numIndividuals, numVariants, ploidy)
    group2Crossover = tally_crossovers(group2)
    
    return group1Crossover, group2Crossover

def main():
    # Specify data locations
    os.makedirs(tmpDir, exist_ok=True)
    locations = Locations(tmpDir)
    
    # Initial argument values
    cmMbp = 3.0 * 1e6 # excessively high to try to force a high amount of recombination
    minimumGroupSize = 1000 # produce enough offspring to establish the trend
    snpMbp = int(1e6) # one SNP every basepair
    seed = 1234 # can be anything
    
    # Parse with in-line validation of QTL simulation encoding
    genotypes, combinationEvaluator, positions = parse_qtl_encoding(
        ["0/1:0/1:1/1", "0/1:2/3:1/2", "0/1:0/1:0/0"],
        "1 AND 2 AND 3",
        ["none", "weak"],
        100, 1000, 10000
    )
    
    # Run the simulations
    for edgeBp in [55, 60, 70, 80]: # arbitrary buffer to left and right of first and last QTLs
        print(f"# edgeBp == {edgeBp}")
        
        cleanup()
        chromax1, chromax2 = chromax_test(genotypes, combinationEvaluator, positions, locations,
                                          cmMbp, snpMbp, edgeBp, minimumGroupSize, seed)
        
        cleanup()
        meiosis1, meiosis2 = meiosis_test(genotypes, combinationEvaluator, positions, locations,
                                          cmMbp, snpMbp, edgeBp, minimumGroupSize, seed)
        
        # Raise an alert if there are any sites with 0 crossovers
        for binArray, algorithm, groupNum in zip([chromax1, chromax2, meiosis1, meiosis2],
                                                ["chromax", "chromax", "popquis", "popquis"],
                                                ["1", "2", "1", "2"]):
            if np.any(binArray[1:] == 0): # first site always "lacks" a crossover due to simple comparison approach
                numSites = np.sum(binArray[1:] == 0)
                print(f"ZEROS: {algorithm} has {numSites} sites where no crossovers occur in group {groupNum}")
            print(f"END: {algorithm} has {binArray[-1]} crossovers at the final position in group {groupNum}")
        print()
    
    # Clean up
    cleanup()

if __name__ == "__main__":
    main()
