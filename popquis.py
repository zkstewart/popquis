#!/usr/bin/env python3
# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.breeder import Breeder
from modules.locations import Locations
from modules.parsing import parse_qtl_encoding
from modules.reporting import write_report_tsv
from modules.simulation import Configuration, Coordinator, Critic
from modules.validation import validate_args, validate_breeding_population

def main():
    usage = """%(prog)s simulates a biparental segregating population with configurable
    QTL occurrence, to estimate the approximate population size needed for a successful
    study outcome.
    
    An example of the -q 'parent1:parent2:group1' encoding is '0/1:1/1:1/1' such that
    parent1 is a heterozygote, parent2 is a homozygote, and the offspring categorised
    into group1 are homozygotes. More than one QTL can be provided as such.
    
    When providing multiple -q values, you must indicate the QTLs that need to be
    inherited in combination (-c) to manifest the phenotypic trait, and the linkage
    (-l) between adjacent QTLs. When specifying only one -q value, you can omit
    the use of -c and -l as they are irrelevant for the single QTL scenario.
    
    With respect to -c, the combination of QTLs is specified using a logical
    structure akin to pseudocode. You can mix numbers (identifying each QTL),
    parentheses, as well as the AND OR operators.
    
    With respect to -l, linkage strength is a function of distance between QTLs. The
    strength values are shorthand for distances which have default values, but which
    can be overridden using the --weak/moderate/strong arguments.
    """
    # Parse command line arguments
    ## Required
    p = argparse.ArgumentParser(description=usage)
    p.add_argument("-q", dest="qtls",
                   required=True,
                   nargs="+",
                   help="""Indicate one or more QTLs in VCF genotype (GT) encoding
                   of the genotypes for parent1:parent2:group1. If providing
                   multiple QTLs, each must be separated by the intended
                   linkage strength""")
    p.add_argument("-c", dest="combination",
                   required=True,
                   nargs="+",
                   help="""Provide a logical formula expressing the necessary combination
                   of QTLs in an offspring organism for it to be categorised into 'group1'
                   using operators: QTL number [e.g., 1, 2, ...], parentheses, AND, OR.
                   If you provided only one QTL to -q, you should just specify '1' here.""")
    p.add_argument("-l", dest="linkage",
                   required=True,
                   nargs="*",
                   choices=["none", "weak", "moderate", "strong"],
                   help="""If you are modelling multiple QTLs, specify the linkage strength
                   of each QTL to its neighbour; if you provided n QTLs to -q, you should
                   provide n-1 values here""")
    p.add_argument("-t", dest="threads",
                   required=True,
                   type=int,
                   help="Specify the number of threads to use for parallel operations")
    p.add_argument("-o", dest="outputDirectory",
                   required=True,
                   help="Specify the location to write outputs")
    ## Optional species-level parameterisation
    p.add_argument("--centimorgans", dest="cmMbp",
                   required=False,
                   type=float,
                   help="Specify the centiMorgan per megabase (default==3.0)",
                   default=3.0)
    ## Optional behavioural parameterisation
    p.add_argument("--popsize", dest="popSize",
                   required=False,
                   type=int,
                   help="""Optionally, specify the number of simulated individuals
                   to assess (default=1000)""",
                   default=1000)
    p.add_argument("--bootstraps", dest="bootstraps",
                   required=False,
                   type=int,
                   help="""Optionally, specify the number of bootstrap replications
                   to run (default=1000)""",
                   default=1000)
    p.add_argument("--weak", dest="weakDistance",
                   required=False,
                   type=int,
                   help="""Optionally, specify the genomic distance (in bp) that
                   would lead to weak linkage (default=10000000 i.e., 10Mbp)""",
                   default=10000000)
    p.add_argument("--moderate", dest="moderateDistance",
                   required=False,
                   type=int,
                   help="""Optionally, specify the genomic distance (in bp) that
                   would lead to moderate linkage (default=5000000 i.e., 5Mbp)""",
                   default=5000000)
    p.add_argument("--strong", dest="strongDistance",
                   required=False,
                   type=int,
                   help="""Optionally, specify the genomic distance (in bp) that
                   would lead to strong linkage (default=1000000 i.e., 1Mbp)""",
                   default=1000000)
    
    args = p.parse_args()
    validate_args(args) # sets args.locations and creates working directory layout; modifies args.combination to be a string, not list
    
    # Parse with in-line validation of QTL simulation encoding
    genotypes, combinationEvaluator, positions = parse_qtl_encoding(args.qtls, args.combination, args.linkage,
                                                                    args.weakDistance, args.moderateDistance,
                                                                    args.strongDistance)
    
    # Produce the breeding population
    "This populates the locations.group1Npy and locations.group2Npy files with simulated individuals"
    breeder = Breeder()
    breeder.establish(positions, genotypes, args.cmMbp)
    breeder.produce_progeny(args.locations, combinationEvaluator,
                            minimumGroupSize=10000, seed=1234)
    
    # Make sure that the simulated populations are consistent with program parameters
    "This check should catch any unusual occurrence where popquis has been run multiple times with different settings"
    validate_breeding_population(args.locations)
    
    # Establish simulation variable combinations
    configuration = Configuration(args.popSize)
    
    # Compute the ED segregation statistics of each simulated variable combination
    coordinator = Coordinator(args.locations)
    coordinator.run(configuration, args.threads, bootstraps=args.bootstraps)
    
    # Score each simulated variable combination
    critic = Critic(args.locations, breeder)
    critic.run(configuration)
    
    # Produce an output tabular report of the simulation outcomes
    if not (os.path.isfile(args.locations.outputTSV) and os.path.isfile(args.locations.outputTSV + Locations.OKAY_SUFFIX)):
        write_report_tsv(args.locations, configuration, len(args.qtls))
        Locations.touch(args.locations.outputTSV)
    else:
        print(f"# Output report table '{args.locations.outputTSV}' already exists; skipping ...")
    
    # Produce the final stacked barplot visualisation
    ## TBD
    
    print("Program completed successfully!")

if __name__ == "__main__":
    main()
