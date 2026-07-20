# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.locations import Locations
from modules.population import Population

def validate_args(args):
    # Validate numeric arguments
    if args.threads < 1:
        raise ValueError("-t must be a positive integer")
    if args.cmMbp <= 0:
        raise ValueError("--centimorgans must greater than zero")
    if args.popSize < 10:
        raise ValueError("--popsize less than 10 is likely to provide meaningless results; provide a higher value")
    if args.bootstraps < 1:
        raise ValueError("--bootstraps must be a positive integer")
    if args.weakDistance < 1:
        raise ValueError("--weak must be a positive integer")
    if args.moderateDistance < 1:
        raise ValueError("--moderate must be a positive integer")
    if args.strongDistance < 1:
        raise ValueError("--strong must be a positive integer")
    
    # Stitch combination nargs into a single string
    args.combination = " ".join(args.combination)
    
    # Create locations object
    args.locations = Locations(args.outputDirectory) # internally validates args.outputDirectory

def validate_breeding_population(locations):
    '''
    This function seeks to ensure that the population has been produced using
    the same variables if the program has been interrupted and resumed.
    '''
    group1 = Population(locations.group1Npy)
    group1.load()
    group2 = Population(locations.group2Npy)
    group2.load()
    
    # Check data shape dimensionality
    group1Shape = group1.data.shape
    group2Shape = group2.data.shape   
    if len(group1Shape) != 3:
        raise ValueError("The data array for the Population of group1 individuals should have a shape with " + 
                         f"3 dimensions, not {len(group1Shape)}. The cause for this issue is unknown, but the " +
                         f"most likely solution is to delete '{locations.group1Npy}' and simulate this Population gain.")
    if len(group2Shape) != 3:
        raise ValueError("The data array for the Population of group2 individuals should have a shape with " + 
                         f"3 dimensions, not {len(group2Shape)}. The cause for this issue is unknown, but the " +
                         f"most likely solution is to delete '{locations.group2Npy}' and simulate this Population gain.")
    
    # Check for consistency of genotypes and ploidy
    g1Individuals, g1Genotypes, g1Ploidy = group1Shape
    g2Individuals, g2Genotypes, g2Ploidy = group2Shape
    if g1Genotypes != g2Genotypes:
        raise ValueError("Populations for group1 and group2 have different numbers of genotypes; " + 
                         f"group1 = {g1Genotypes} and group2 = {g2Genotypes}. The cause for this issue is " + 
                         "possibly related to running popquis multiple times with different parameter values " + 
                         "(e.g., different -q or -l or --centimorgans values). The most likely solution is to " +
                         f"delete the '{locations.group1Npy}' and '{locations.group2Npy}' files and re-run " + 
                         "popquis from scratch.")
    elif g1Ploidy != g2Ploidy:
        raise ValueError("Populations for group1 and group2 have different levels of ploidy; " + 
                         f"group1 = {g1Ploidy} and group2 = {g2Ploidy}. The cause for this issue is " + 
                         "possibly related to running popquis multiple times with different parameter values " + 
                         "for -q. The most likely solution is to delete the " +
                         f"'{locations.group1Npy}' and '{locations.group2Npy}' files and re-run " + 
                         "popquis from scratch.")
