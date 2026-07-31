# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.breeder import Breeder
from modules.experiment import Configuration
from modules.parsing import parse_qtl_encoding
from modules.locations import Locations
from modules.simulator import MeiosisSimulator

def meiosis_crossovers(seed, numReps=100000):
    for cmMbp in [1.0, 2.0, 2.2, 2.8, 10.0]:
        genotypes, combinationEvaluator, positions = parse_qtl_encoding(
            ["0/1:0/1:1/1", "0/1:0/1:1/1", "0/1:0/1:1/1", "0/1:0/1:1/1"],
            "1 AND 2 AND 3 AND 4",
            ["weak", "moderate", "strong"],
            int(100e6), # weak linkage; 100 Mbp
            int(10e6), # moderate distance; 10 Mbp
            int(1e6)) # strong linkage; 1 Mbp
        
        breeder = Breeder()
        breeder.establish(positions, genotypes, cmMbp)
        
        simulator = MeiosisSimulator(breeder.parent1,
                                     breeder.parent2,
                                     breeder.genomeMap,
                                     seed=seed)
        
        for chromosome in breeder.parent1:
            cmPositions = breeder.genomeMap[chromosome.chromID].df["cM"].to_numpy()
            
            # Q: Does the number of crossovers meet expectations?
            numCrosses = 0
            for i in range(numReps):
                crossovers = simulator._generate_crossovers(cmPositions) # cM position where a crossover occurs
                numCrosses += len(crossovers)
            
            expectedCrossAvg = cmPositions[-1] / 100 # 100 cM gives 1 morgan (100% chance of recombination)
            actualCrossAvg = numCrosses / numReps
            
            print(f"# meiosis_crossovers(): with cmMbp=={cmMbp} we expect " + 
                  f"{round(expectedCrossAvg, 2)} crossover events for a genetic map " +
                  f"that is {round(cmPositions[-1], 2)} cM long; found {actualCrossAvg}")

def meiosis_interference(seed, numReps=100000):
    cmMbp = 1.0
    genotypes, combinationEvaluator, positions = parse_qtl_encoding(
        ["0/1:0/1:1/1", "0/1:0/1:1/1"],
        "1 AND 2",
        ["weak"],
        int(100e6),
        int(10e6), 
        int(1e6))
    
    breeder = Breeder()
    breeder.establish(positions, genotypes, cmMbp)
    
    # Q: Does the nu parameter influence interference?
    for nu in [1.0, 2.0, 2.2, 2.5, 2.8, 10.0]:
        simulator = MeiosisSimulator(breeder.parent1,
                                     breeder.parent2,
                                     breeder.genomeMap,
                                     seed=seed,
                                     nu=nu)
        
        for chromosome in breeder.parent1:
            cmPositions = breeder.genomeMap[chromosome.chromID].df["cM"].to_numpy()
            
            spacing = []
            numCrosses = 0
            for i in range(numReps):
                crossovers = simulator._generate_crossovers(cmPositions) # cM position where a crossover occurs
                if len(crossovers) > 1:
                    deltas = np.diff(crossovers, axis=0)
                    spacing += list(deltas)
                numCrosses += len(crossovers)
            avgSpacing = np.mean(spacing)
            crossAvg = numCrosses / numReps
            
            print(f"# meiosis_interference(): when nu=={nu} the average " +
                  f"spacing between crossover sites is {round(avgSpacing, 2)} cM; " +
                  f"average number of crossovers is {round(crossAvg, 2)}")

def meiosis_skew(seed, numReps=100000):
    cmMbp = 1.0
    genotypes, combinationEvaluator, positions = parse_qtl_encoding(
        ["0/1:0/1:1/1", "0/1:0/1:1/1"],
        "1 AND 2",
        ["weak"],
        int(100e6),
        int(10e6), 
        int(1e6))
    
    breeder = Breeder()
    breeder.establish(positions, genotypes, cmMbp)
    
    # Q: Do the crossover points spread out evenly?
    simulator = MeiosisSimulator(breeder.parent1,
                                 breeder.parent2,
                                 breeder.genomeMap,
                                 seed=seed)
    
    for chromosome in breeder.parent1:
        cmPositions = breeder.genomeMap[chromosome.chromID].df["cM"].to_numpy()
        
        points = []
        for i in range(numReps):
            crossovers = simulator._generate_crossovers(cmPositions) # cM position where a crossover occurs
            if len(crossovers) > 0:
                points += list(crossovers)
        
        counts, bin_edges = np.histogram(points, bins=10, range=None, density=None, weights=None)
        
        proportionCounts = (counts / np.sum(counts))
        minmaxCounts = (counts - counts.min()) / (counts.max() - counts.min())
        formatted = [ str(round(x*100, 4)) + "%" for x in proportionCounts ]
        
        print("# meiosis_skew(): forming a histogram with 10 bins for " +
              "the crossover points shows a distribution of where each bin " +
              "proportionally contains this amount of the crossover points " + 
              f"{formatted}")

if __name__ == "__main__":
    for seed in [1111, 2222, 3333, 4444, 5555]:
        print(f"## Validation with seed=={seed}")
        meiosis_crossovers(seed)
        meiosis_interference(seed)
        meiosis_skew(seed)
        print()
