#!/usr/bin/env python3
# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.breeder import Breeder
from modules.chromosome import Chromosome
from modules.chromosomemap import ChromosomeMap
from modules.combination import Combination
from modules.errors import InvalidGenotypeError, IncompatibleGenotypeError
from modules.experiment import Configuration, Coordinator, Critic
from modules.genome import Genome
from modules.genomemap import GenomeMap
from modules.genotype import Genotype # note that the Genotype class is implicitly tested by the TestParsing class herein
from modules.locations import Locations
from modules.parsing import parse_genotypes, parse_combination, parse_linkage, parse_qtl_encoding
from modules.population import Population
from modules.reporting import _separate_thresholds
from modules.simulator import MeiosisSimulator
from modules.spreadsheet import Spreadsheet
from modules.statistics import RandomNumberGenerator, Calculator
from modules.template import Template

# Specify data locations
testDir = os.path.dirname(os.path.abspath(__file__))
tmpDir = os.path.join(testDir, "tmp")

# Define initial clean up function
def cleanup():
    os.makedirs(tmpDir, exist_ok=True)
    locations = Locations(tmpDir)
    if os.path.isfile(locations.group1Npy):
        os.unlink(locations.group1Npy)
    if os.path.isfile(locations.group2Npy):
        os.unlink(locations.group2Npy)
    
    dummyConfig = Configuration(100)
    for b in dummyConfig.popBalance:
        for p in dummyConfig.phenotypeError:
            npzFile = os.path.join(locations.storageDir, f"{b}_{p}.npz")
            if os.path.isfile(npzFile):
                os.unlink(npzFile)

# Define unit tests
class TestParsing(unittest.TestCase):
    def test_parse_genotypes_when_valid(self):
        # Arrange
        gtString1 = "0/1:1/1:1/1"
        gtString2 = "0/0:0/1:0/1"
        gtString3 = "0/1:1/2:0/2"
        
        # Act
        genotypes1 = parse_genotypes(gtString1)
        genotypes2 = parse_genotypes(gtString2)
        genotypes3 = parse_genotypes(gtString3)
        
        # Assert on equality
        self.assertEqual(genotypes1[0].alleles, [0, 1])
        self.assertEqual(genotypes1[1].alleles, [1, 1])
        self.assertEqual(genotypes1[2].alleles, [1, 1])
        
        self.assertEqual(genotypes2[0].alleles, [0, 0])
        self.assertEqual(genotypes2[1].alleles, [0, 1])
        self.assertEqual(genotypes2[2].alleles, [0, 1])
        
        self.assertEqual(genotypes3[0].alleles, [0, 1])
        self.assertEqual(genotypes3[1].alleles, [1, 2])
        self.assertEqual(genotypes3[2].alleles, [0, 2])
    
    def test_parse_genotypes_when_invalid(self):
        # Arrange
        gtString1 = "0|1:1/1:1/1"
        gtString2 = "00:0/1:0/1"
        gtString3 = "0/1:1/2"
        gtString4 = "0/1:1/2;0/11"
        
        # Act & Assert on errors
        with self.assertRaises(InvalidGenotypeError):
            genotypes1 = parse_genotypes(gtString1)
        with self.assertRaises(InvalidGenotypeError):
            genotypes2 = parse_genotypes(gtString2)
        with self.assertRaises(ValueError):
            genotypes3 = parse_genotypes(gtString3)
        with self.assertRaises(ValueError):
            genotypes4 = parse_genotypes(gtString4)
    
    def test_parse_genotypes_when_impossible(self):
        # Arrange
        gtString1 = "0/1:1/1:0/0"
        gtString2 = "0/0:0/1:1/1"
        gtString3 = "0/1:0/1:0/2"
        
        # Act & Assert on errors
        with self.assertRaises(IncompatibleGenotypeError):
            genotypes1 = parse_genotypes(gtString1)
        with self.assertRaises(IncompatibleGenotypeError):
            genotypes2 = parse_genotypes(gtString2)
        with self.assertRaises(IncompatibleGenotypeError):
            genotypes3 = parse_genotypes(gtString3)
    
    def test_parse_combination_when_valid(self):
        # Arrange
        numQTLs = 3
        combinationStrings = [
            "1 AND 2 OR 3", # all are mandatory
            "1 AND (2 OR 3)", # 1 is mandatory; must come with either 2 or 3
            "(1 AND 2) OR 3", # either 1 and 2 inherit together, or 3 is sufficient on its own
            "(1 OR 2 OR 3)", # parentheses add nothing, but are no harm
            "((1 AND 2) OR 3)" # parentheses add nothing, but are no harm
        ]
        
        # Act & Assert (no error is a pass)
        for combinationString in combinationStrings:
            combination = parse_combination(combinationString, numQTLs)
    
    def test_parse_combination_when_invalid(self):
        # Arrange
        numQTLs = 3
        valueErrorStrings = [
            "1 AND 2", # must mention 3
            "1 AND 2 AND 3 AND 4", # there is no 4th QTL
            "2 AND 3 AND 1", # must be ordered from lowest to highest
            "1 AND 2 AND 4", # must mention 3; there is also no 4th QTL
            "" # cannot be empty
        ]
        syntaxErrorStrings = [
            "(1 AND) 2 or 3", # parentheses must close after a QTL number
            "(1 AND 2 AND 3", # parentheses must be closed
            "1 AND 2 AND 3)", # parentheses must be opened first before closing
            "1 OR OR 2 AND 3", # cannot have multiple operators sequentially
            "1 2 AND 3" # must have 1 operator between each number
        ]
        
        # Act & Assert on errors
        for combinationString in valueErrorStrings:
            with self.assertRaises(ValueError):
                combination = parse_combination(combinationString, numQTLs)
        
        for combinationString in syntaxErrorStrings:
            with self.assertRaises(SyntaxError):
                combination = parse_combination(combinationString, numQTLs)
    
    def test_parse_linkage_when_valid(self):
        # Arrange
        weak = 100
        moderate = 10
        strong = 1
        
        linkageList1 = []
        linkageList2 = ["none"]
        linkageList3 = ["weak"]
        linkageList4 = ["weak", "strong"]
        linkageList5 = ["weak", "none", "strong"]
        
        expected1 = [('chr1', 0)] # single QTL outcome
        expected2 = [('chr1', 0), ('chr2', 0)] # 'none' triggers new chromosome
        expected3 = [('chr1', 0), ('chr1', 100)] # 'weak' adds +100bp
        expected4 = [('chr1', 0), ('chr1', 100), ('chr1', 101)] # +100bp then +1bp
        expected5 = [('chr1', 0), ('chr1', 100), ('chr2', 0), ('chr2', 1)] # +100bp then new chrom then +1bp
        
        # Act
        positions1 = parse_linkage(linkageList1, weak, moderate, strong, 1)
        positions2 = parse_linkage(linkageList2, weak, moderate, strong, 2)
        positions3 = parse_linkage(linkageList3, weak, moderate, strong, 2)
        positions4 = parse_linkage(linkageList4, weak, moderate, strong, 3)
        positions5 = parse_linkage(linkageList5, weak, moderate, strong, 4)
        
        # Assert
        self.assertEqual(positions1, expected1)
        self.assertEqual(positions2, expected2)
        self.assertEqual(positions3, expected3)
        self.assertEqual(positions4, expected4)
        self.assertEqual(positions5, expected5)
    
    def test_parse_linkage_when_invalid(self):
        # Arrange
        weak = 100
        moderate = 10
        strong = 1
        
        linkageList1 = []
        linkageList2 = ["none"]
        
        # Act & Assert on errors
        with self.assertRaises(ValueError):
            positions1 = parse_linkage(linkageList1, weak, moderate, strong, 2)
        with self.assertRaises(ValueError):
            positions2 = parse_linkage(linkageList2, weak, moderate, strong, 1)
    
    def test_parse_qtl_encoding_when_valid(self):
        # Arrange
        qtls1 = ["0/1:1/1:1/1", "0/0:0/1:0/1", "0/1:1/2:0/2"]
        combination1 = "1 AND 2 AND 3"
        linkage1 = ["none", "weak"]
        
        qtls2 = ["0/1:1/1:1/1"]
        combination2 = "1"
        linkage2 = []
        
        weak = 100
        moderate = 10
        strong = 1
        
        # Act & Assert (no error is a pass)
        qtlGenotypes1, combinationEvaluator1, qtlPositions1 = parse_qtl_encoding(
            qtls1, combination1, linkage1, weak, moderate, strong
        )
        qtlGenotypes2, combinationEvaluator2, qtlPositions2 = parse_qtl_encoding(
            qtls2, combination2, linkage2, weak, moderate, strong
        )
    
    def test_parse_qtl_encoding_when_invalid(self):
        # Arrange
        qtls1 = ["0/0:0/0:0/0"] # deterministic outcome
        combination1 = "1"
        linkage1 = []
        weak = 100
        moderate = 10
        strong = 1
        
        # Act & Assert on errors
        with self.assertRaises(ValueError):
            qtlGenotypes1, combinationEvaluator1, qtlPositions1 = parse_qtl_encoding(
                qtls1, combination1, linkage1, weak, moderate, strong
            )

class TestCombinationEvaluation(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        combinationStrs = [
            "1 AND 2 OR 3",
            "1 AND (2 OR 3)",
            "(1 AND 2) OR 3",
            "(1 OR 2 OR 3)",
            "((1 AND 2) OR 3)"
        ]
        evalStrs = [
            "{0} and {1} or {2}",
            "{0} and ({1} or {2})",
            "({0} and {1}) or {2}",
            "({0} or {1} or {2})",
            "(({0} and {1}) or {2})"
        ]
        qtlBoolCombos = [
            (qtl1, qtl2, qtl3)
            for qtl1 in (True, False)
            for qtl2 in (True, False)
            for qtl3 in (True, False)
        ]
        
        # Act & Assert in loop on equality
        for combinationStr, evalStr in zip(combinationStrs, evalStrs):
            for qtlBoolCombo in qtlBoolCombos:
                combination = Combination(combinationStr)
                evaluatorResult = combination.evaluate(qtlBoolCombo)
                evalStrResult = eval(evalStr.format(*qtlBoolCombo))
                assert evaluatorResult == evalStrResult
                
                self.assertEqual(evaluatorResult, evalStrResult, 
                                 f"'{combinationStr}' gives inconsistent output")
    
    def test_when_invalid(self):
        # Arrange
        combinationStrs = [
            "1 AND 2 OR 3",
            "1 AND (2 OR 3)",
            "(1 AND 2) OR 3",
            "(1 OR 2 OR 3)",
            "((1 AND 2) OR 3)"
        ]
        valueErrorLists = [
            [True, True], # too few numbers
            [True, True, True, True], # too many numbers
            [], # empty list
            ["True", True, True] # object type mismatch
        ]
        typeErrorLists = [
            {1: True, 2: True, 3: True} # object type mismatch
        ]
        
        # Act & Assert in loop on errors
        for combinationStr in combinationStrs:
            combination = Combination(combinationStr)
            for variables in valueErrorLists:
                with self.assertRaises(ValueError):
                    evaluatorResult = combination.evaluate(variables)
            for variables in typeErrorLists:
                with self.assertRaises(TypeError):
                    evaluatorResult = combination.evaluate(variables)

class TestChromosome(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        chromID = "chr1"
        positions = [50, 100, 150]
        length = 200
        cmMbp = 1
        snpMbp = int(1e6)
        
        transitionPoints = [(positions[0]+positions[1])//2, (positions[1]+positions[2])//2]
        genotypeTransitions1 = [[0,0], [0,1], [1,1]]
        genotypeTransitions2 = [[0,0,0,0], [0,0,1,1], [1,1,1,1]]
        
        chromMap = ChromosomeMap(chromID, length, cmMbp, snpMbp, positions) # 1 SNP per bp
        
        genotypes1 = [Genotype("0/0"), Genotype("0/1"), Genotype("1/1")]
        ploidy1 = 2
        
        genotypes2 = [Genotype("0/0/0/0"), Genotype("0/0/1/1"), Genotype("1/1/1/1")]
        ploidy2 = 4
        
        # Act
        chromosome1 = Chromosome(chromID, positions, genotypes1, chromMap)
        chromosome2 = Chromosome(chromID, positions, genotypes2, chromMap)
        
        # Assert
        self.assertEqual((length, ploidy1), chromosome1.array.shape)
        for i, gt in enumerate(chromosome1.array):
            if i <= transitionPoints[0]:
                self.assertTrue(np.array_equal(gt, genotypeTransitions1[0]))
            elif i <= transitionPoints[1]:
                self.assertTrue(np.array_equal(gt, genotypeTransitions1[1]))
            else:
                self.assertTrue(np.array_equal(gt, genotypeTransitions1[2]))
        
        self.assertEqual((length, ploidy2), chromosome2.array.shape)
        for i, gt in enumerate(chromosome2.array):
            if i <= transitionPoints[0]:
                self.assertTrue(np.array_equal(gt, genotypeTransitions2[0]))
            elif i <= transitionPoints[1]:
                self.assertTrue(np.array_equal(gt, genotypeTransitions2[1]))
            else:
                self.assertTrue(np.array_equal(gt, genotypeTransitions2[2]))

class TestGenome(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        chromID1 = "chr1"
        chromID2 = "chr2"
        positions = [50, 100, 150]
        length = 200
        cmMbp = 1
        snpMbp = int(1e6)
        
        chromMap1 = ChromosomeMap(chromID1, length, cmMbp, snpMbp, positions) # 1 SNP per bp
        chromMap2 = ChromosomeMap(chromID2, length, cmMbp, snpMbp, positions) # 1 SNP per bp
        genotypes1 = [Genotype("0/0"), Genotype("0/1"), Genotype("1/1")]
        genotypes2 = [Genotype("1/2"), Genotype("2/2"), Genotype("2/3")]
        
        # Act
        chromosome1 = Chromosome(chromID1, positions, genotypes1, chromMap1)
        chromosome2 = Chromosome(chromID2, positions, genotypes2, chromMap2)
        
        genome = Genome()
        genome[chromID1] = chromosome1
        genome[chromID2] = chromosome2
        
        # Assert
        self.assertEqual(chromosome1.array.shape[0] + chromosome2.array.shape[0], genome.array.shape[0])
        ongoingCount = 0
        for chromosome in (chromosome1, chromosome2):
            for gt in chromosome.array:
                genomeGt = genome.array[ongoingCount]
                self.assertTrue(np.array_equal(gt, genomeGt))
                ongoingCount += 1
    
    def test_when_invalid(self):
        # Arrange
        chromID1 = "chr1"
        chromID2 = "chr2"
        positions = [50, 100, 150]
        length = 200
        cmMbp = 1
        snpMbp = int(1e6)
        
        transitionPoints = [(positions[0]+positions[1])//2, (positions[1]+positions[2])//2]
        genotypeTransitions1 = [[0,0], [0,1], [1,1]]
        genotypeTransitions2 = [[0,0,0,0], [0,0,1,1], [1,1,1,1]]
        
        chromMap = ChromosomeMap(chromID1, length, cmMbp, snpMbp, positions) # 1 SNP per bp
        genotypes1 = [Genotype("0/0"), Genotype("0/1"), Genotype("1/1")]
        genotypes2 = [Genotype("0/0/0/0"), Genotype("0/0/1/1"), Genotype("1/1/1/1")]
        
        # Act
        chromosome1 = Chromosome(chromID1, positions, genotypes1, chromMap)
        chromosome2 = Chromosome(chromID1, positions, genotypes2, chromMap)
        genome = Genome()
        genome[chromID1] = chromosome1
        with self.assertRaises(KeyError): # cannot store the same chromID twice
            genome[chromID1] = chromosome1
        with self.assertRaises(ValueError): # ploidy incompatibility
            genome[chromID2] = chromosome2

class TestChromosomeMap(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        chromID = "chr1"
        cmMbp = 3
        positions = [0] # just make sure every map has a marker at index == 0
        
        length1 = 100000
        snpMbp1 = 1000 # should equate to 1 SNP every 1000 bp
        
        length2 = 100
        snpMbp2 = int(1e6) # should equate to 1 SNP every 1 bp
        
        length3 = int(1e6)
        snpMbp3 = 1 # should equate to 1 SNP every 1 Mbp
        
        # Act
        chromMap1 = ChromosomeMap(chromID, length1, cmMbp, snpMbp1, positions)
        nrow1 = len(chromMap1.df)
        expectedNrow1 = length1 // (int(1e6)//snpMbp1)
        
        chromMap2 = ChromosomeMap(chromID, length2, cmMbp, snpMbp2, positions)
        nrow2 = len(chromMap2.df)
        expectedNrow2 = length2 // (int(1e6)//snpMbp2)
        
        chromMap3 = ChromosomeMap(chromID, length3, cmMbp, snpMbp3, positions)
        nrow3 = len(chromMap3.df)
        expectedNrow3 = length3 // (int(1e6)//snpMbp3)
        
        # Assert
        self.assertEqual(expectedNrow1, nrow1)
        self.assertEqual(expectedNrow2, nrow2)
        self.assertEqual(expectedNrow3, nrow3)
    
    def test_when_invalid(self):
        # Arrange
        chromID = "chr1"
        length = 1
        cmMbp = 3
        
        snpMbp1 = 1
        snpMbp2 = int(1e6)
        positions1 = [0]
        positions2 = [100]
        
        # Act & Assert on error
        with self.assertRaises(ValueError): # snpMbp is too sparse for the short length
            chromMap = ChromosomeMap(chromID, length, cmMbp, snpMbp1, positions1)
        with self.assertRaises(ValueError): # positions doesn't lead to a marker placement
            chromMap = ChromosomeMap(chromID, length, cmMbp, snpMbp2, positions2)

class TestGenomeMap(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        chromID1 = "chr1"
        chromID2 = "chr2"
        chromID3 = "chr3"
        cmMbp = 3
        positions = [0]
        
        length1 = 100000
        snpMbp1 = 1000 # should equate to 1 SNP every 1000 bp
        
        length2 = 100
        snpMbp2 = int(1e6) # should equate to 1 SNP every 1 bp
        
        length3 = int(1e6)
        snpMbp3 = 1 # should equate to 1 SNP every 1 Mbp
        
        # Act
        chromMap1 = ChromosomeMap(chromID1, length1, cmMbp, snpMbp1, positions)
        chromMap2 = ChromosomeMap(chromID2, length2, cmMbp, snpMbp2, positions)
        chromMap3 = ChromosomeMap(chromID3, length3, cmMbp, snpMbp3, positions)
        
        genomeMap = GenomeMap()
        genomeMap[chromID1] = chromMap1
        genomeMap[chromID2] = chromMap2
        genomeMap[chromID3] = chromMap3
        
        # Assert
        self.assertEqual(len(chromMap1.df) + len(chromMap2.df) + len(chromMap3.df), len(genomeMap.df))
        ongoingCount = 0
        for chromMap in (chromMap1, chromMap2, chromMap3):
            for _, chromRow in chromMap.df.iterrows():
                genomeRow = genomeMap.df.iloc[ongoingCount]
                assert genomeRow.equals(chromRow)
                ongoingCount += 1

class TestPopulation(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        tmpFile = os.path.join(tmpDir, "tmp.npy")
        
        individual1 = np.array([0, 1, 0, 1, 0, 1]).reshape(1, 3, 2) # 3 heterozygous genotypes
        individual2 = np.array([1, 2, 1, 2, 1, 2]).reshape(1, 3, 2) # 3 heterozygous genotypes
        
        # Act
        pop = Population(tmpFile)
        pop.add(individual1)
        pop.add(individual2)
        pop.load()
        
        retrieved1 = pop.retrieve([0])
        retrieved2 = pop.retrieve([1])
        
        # Assert
        self.assertTrue(np.array_equal(individual1, retrieved1))
        self.assertTrue(np.array_equal(individual2, retrieved2))
        
        # Clean up
        os.unlink(tmpFile)
    
    def test_when_invalid(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        tmpFile = os.path.join(tmpDir, "tmp.npy")
        
        # Act & Assert on error
        open(tmpFile, "w").close()
        pop = Population(tmpFile)
        with self.assertRaises(EOFError):
            pop.load()
        
        # Clean up
        os.unlink(tmpFile)

class TestBreeder(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        cmMbp = 3.0
        genotype1 = [Genotype("0/0"), Genotype("0/1"), Genotype("0/1")] # parent1, parent2, offspring
        positions = [0]
        edgeBp = 50
        positions = [ ("chr1", x) for x in positions ]
        combinationEvaluator = Combination("1")
        
        # Act
        breeder = Breeder()
        breeder.establish(positions, [genotype1], cmMbp,
                          snpMbp=int(1e6), edgeBp=edgeBp)
        
        # Assert
        self.assertEqual(len(breeder.genomeMap.df), (edgeBp*2) + (positions[-1][-1] + 1))
        
        markerRow = breeder.genomeMap.markers
        self.assertEqual(len(markerRow), 1) # should be one marker row
        
        breeder.produce_progeny(locations, combinationEvaluator, batchSize=100, minimumGroupSize=1, seed=1234) # no error is a pass
        
        # Clean up
        cleanup()
    
    def test_qtlRanges(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        chroms = ["chr1", "chr2"]
        cmMbp = 3.0
        genotype1 = [Genotype("0/0"), Genotype("0/1"), Genotype("0/1")] # parent1, parent2, offspring
        positions = [0]
        edgeBp = 50
        positions = [ (chrom, x) for x in positions for chrom in chroms ]
        combinationEvaluator = Combination("1 AND 2")
        
        # Act
        breeder = Breeder()
        breeder.establish(positions, [genotype1, genotype1], cmMbp,
                          snpMbp=int(1e6), edgeBp=edgeBp)
        
        # Assert
        expectedLength = sum([
            (edgeBp*2) + pos[-1] + 1
            for pos in positions # two chroms so have to do edgeBp and such for each
        ])
        
        self.assertEqual(len(breeder.genomeMap.df), expectedLength)
        
        markerRow = breeder.genomeMap.markers
        self.assertEqual(len(markerRow), 2) # should be two marker rows
        
        self.assertEqual(breeder.qtlRanges, [(0, 100), (101, 201)])
        
        # Clean up
        cleanup()
    
    def test_when_invalid(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        cmMbp = 3.0
        genotype1 = [Genotype("0/0"), Genotype("1/1"), Genotype("0/1")] # parent1, parent2, offspring
        positions = [0]
        edgeBp = 50
        positions = [ ("chr1", x) for x in positions ]
        combinationEvaluator = Combination("1")
        
        # Act
        breeder = Breeder()
        breeder.establish(positions, [genotype1], cmMbp,
                          snpMbp=int(1e6), edgeBp=edgeBp)
        
        # Assert
        with self.assertRaises(Exception): # all offspring end up in group1 with none in group2
            breeder.produce_progeny(locations, combinationEvaluator, batchSize=10, minimumGroupSize=1, seed=1234)

class TestConfiguration(unittest.TestCase):
    def test_when_valid(self):
        "It's kind of hard to test this Class meaningfully"
        # Arrange
        popSize1 = 100
        popSize2 = 1000
        popSize3 = 19 # small-size prime number
        popSize4 = 127 # medium-size prime number
        popSize5 = 953 # large-size prime number
        popSize6 = 7919 # very large-size prime number
        
        # Act
        config1 = Configuration(popSize1)
        config2 = Configuration(popSize2)
        config3 = Configuration(popSize3)
        config4 = Configuration(popSize4)
        config5 = Configuration(popSize5)
        config6 = Configuration(popSize6)
        
        expectedKeys = set([
            (b, p)
            for b in config1.popBalance
            for p in config1.phenotypeError
        ])
        
        # Assert
        self.assertEqual(set(config1.combos.keys()), expectedKeys)
        self.assertEqual(set(config2.combos.keys()), expectedKeys)
        self.assertEqual(set(config3.combos.keys()), expectedKeys)
        self.assertEqual(set(config4.combos.keys()), expectedKeys)
        self.assertEqual(set(config5.combos.keys()), expectedKeys)
        self.assertEqual(set(config6.combos.keys()), expectedKeys)
    
    def test_when_invalid(self):
        # Arrange
        popSize1 = 104.5
        popSize2 = 4
        
        # Act & Assert on errors
        with self.assertRaises(TypeError): # value is not an integer
            config1 = Configuration(popSize1)
        with self.assertRaises(ValueError): # value is less than 10
            config2 = Configuration(popSize2)

class TestCoordinator(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        open(locations.group1Npy, "w").close()
        open(locations.group2Npy, "w").close()
        
        # Act & Assert (no error is a pass)
        coordinator = Coordinator(locations)
    
    def test_array_shape1(self):
        "The Coordinator class forms a combined array of many results; understanding its shape is necessary to parse results back out of the array"
        # Arrange
        numVariants = 15
        array1 = np.array([1] * numVariants)
        array2 = np.array([2] * numVariants)
        array3 = np.array([3] * numVariants)
        array4 = np.array([4] * numVariants)
        bootstraps = 10
        
        expectedShape = (4, bootstraps, numVariants) # (popsize, bootstraps, numVariants)
        
        # Act
        futures = []
        for array in [array1, array2, array3, array4]:
            for replication in range(bootstraps):
                futures.append(array)
        
        resultsArray = np.stack(futures)
        resultsArray = np.stack(np.split(resultsArray, 4)) # len(popSizes) == 4 as each popSize has its own array
        
        # Assert
        self.assertEqual(resultsArray.shape, expectedShape)
        self.assertEqual(np.sum(resultsArray[0]), bootstraps * numVariants * 1)
        self.assertEqual(np.sum(resultsArray[1]), bootstraps * numVariants * 2)
        self.assertEqual(np.sum(resultsArray[2]), bootstraps * numVariants * 3)
        self.assertEqual(np.sum(resultsArray[3]), bootstraps * numVariants * 4)
    
    def test_array_shape2(self):
        "The first shape test failed to detect an actual problem with how the arrays were being reshaped"
        # Arrange
        numVariants = 15
        groupSize = 10
        ploidy = 2
        array1 = np.array([[[0,1]]*numVariants]*groupSize)
        
        expectedStartShape = (groupSize, numVariants, ploidy)
        expectedEndShape = (numVariants, groupSize*ploidy) # groupSize*ploidy gives numAlleles
        
        # Act
        reshaped1 = np.reshape(np.moveaxis(array1, 0, 2), (numVariants, -1))
        
        # Assert
        self.assertEqual(array1.shape, expectedStartShape)
        self.assertEqual(reshaped1.shape, expectedEndShape)
    
    def test_array_shape3(self):
        """In the second shape test, the arrays were being dimensionally shaped correctly,
        but it was unclear if the data was correct. This test shows two different ways at
        ariving at the same shape with equality of data contents, which appears to suggest
        that either approach is suitable."""
        # Arrange
        numVariants = 15
        groupSize = 10
        ploidy = 2
        array1 = np.array([[[0,1]]*numVariants]*groupSize)
        
        expectedStartShape = (groupSize, numVariants, ploidy)
        expectedEndShape = (numVariants, groupSize*ploidy) # groupSize*ploidy gives numAlleles
        
        # Act
        reshaped1 = np.reshape(np.moveaxis(array1, 0, 2), (numVariants, -1))
        reshaped2 = array1.transpose(1, 0, 2).reshape(numVariants, -1) 
        
        # Assert
        self.assertEqual(array1.shape, expectedStartShape)
        self.assertEqual(reshaped1.shape, expectedEndShape)
        self.assertEqual(reshaped1.shape, reshaped2.shape)
        self.assertFalse(np.array_equal(reshaped1, reshaped2))
    
    def test_phenotype_error_groups(self):
        "This test will model how the phenotype error is manifested in a group array"
        # Arrange
        numVariants = 15
        groupSize = 10
        ploidy = 2
        numGroup1Correct = 7
        numGroup1Errors = 3
        numGroup2Correct = 7
        numGroup2Errors = 3
        
        array1 = np.array([[[0,0]]*numVariants]*groupSize)
        array2 = np.array([[[1,1]]*numVariants]*groupSize)
        
        expectedShape = (groupSize, numVariants, ploidy)
        
        # Act
        g1Array = np.vstack((array1[0:numGroup1Correct], array2[numGroup2Correct:])) # good + bad
        g2Array = np.vstack((array2[0:numGroup2Correct], array1[numGroup1Correct:])) # good + bad
        
        # Assert
        self.assertEqual(g1Array.shape, expectedShape)
        self.assertEqual(g2Array.shape, expectedShape)
        
        for i in range(groupSize):
            g1Alleles = g1Array[:,i]
            g2Alleles = g2Array[:,i]
            if i < numGroup1Correct:
                self.assertEqual(int(np.sum(g1Alleles == 0)), numGroup1Correct*ploidy)
                self.assertEqual(int(np.sum(g2Alleles == 1)), numGroup2Correct*ploidy)
            else:
                self.assertEqual(int(np.sum(g1Alleles == 1)), numGroup1Errors*ploidy)
                self.assertEqual(int(np.sum(g2Alleles == 0)), numGroup2Errors*ploidy)
    
    def test_run(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        chromID = "chr1"
        positions = [(chromID, 50), (chromID, 100), (chromID, 150)]
        cmMbp = 1
        snpMbp = int(1e6)
        edgeBp = 50
        popSize = 50
        genotypes = [[Genotype("0/1"), Genotype("0/1"), Genotype("1/1")]]*3
        combinationEvaluator = Combination("1 AND 2 AND 3")
        
        breeder = Breeder()
        breeder.establish(positions, genotypes, cmMbp, snpMbp=snpMbp, edgeBp=edgeBp)
        breeder.produce_progeny(locations, combinationEvaluator,
                                batchSize=100,
                                minimumGroupSize=50, # chosenPopsize below goes up to 50
                                seed=1234)
        
        configuration = Configuration(popSize)
        chosenBalance, chosenError = (0.5, 0.0)
        chosenPopsize = configuration.combos[(chosenBalance, chosenError)]
        
        threads = 1
        bootstraps = 10
        
        # Act
        coordinator = Coordinator(locations)
        coordinator.run(configuration, breeder.qtlRanges, threads, bootstraps=bootstraps)
        chosenSpreadsheet = Spreadsheet(locations.storageDir, chosenBalance, chosenError, chosenPopsize)
        chosenSpreadsheet.load()
        
        # Assert
        self.assertTrue(hasattr(chosenSpreadsheet, "ed1"))
        self.assertTrue(hasattr(chosenSpreadsheet, "ed2"))
        self.assertTrue(hasattr(chosenSpreadsheet, "ed3"))
        
        expectedShape1 = (
            len(chosenPopsize),
            bootstraps,
            edgeBp + (positions[0][1]) + (((positions[1][1]+positions[0][1])/2) - positions[0][1]) + 1
        )
        self.assertEqual(chosenSpreadsheet.ed1.shape, expectedShape1)
        
        expectedShape2 = (
            len(chosenPopsize),
            bootstraps,
            ((positions[2][1]+positions[1][1])/2) - ((positions[1][1]+positions[0][1])/2)
        )
        self.assertEqual(chosenSpreadsheet.ed2.shape, expectedShape2)
        
        expectedShape3 = (
            len(chosenPopsize),
            bootstraps,
            edgeBp + positions[2][1] - ((positions[2][1]+positions[1][1])/2)
        )
        self.assertEqual(chosenSpreadsheet.ed3.shape, expectedShape3)
        
        # Clean up
        os.unlink(locations.group1Npy)
        os.unlink(locations.group2Npy)
    
    def test_when_invalid(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        # Act & Assert on error
        with self.assertRaises(FileNotFoundError): # npy files don't exist
            coordinator = Coordinator(locations)

class TestSpreadsheet(unittest.TestCase):
    def test_unfixed_variable_setting(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        popSize = 50
        configuration = Configuration(popSize)
        chosenBalance, chosenError = (0.5, 0.0)
        chosenPopsize = configuration.combos[(chosenBalance, chosenError)]
        
        scores0 = np.array([0, 0, 0])
        scores1 = np.array([1, 1, 1])
        _ignored = np.array([2, 2, 2])
        
        # Act
        spreadsheet = Spreadsheet(locations.storageDir, chosenBalance, chosenError, chosenPopsize)
        spreadsheet.scores0 = scores0
        spreadsheet.scores1 = scores1
        spreadsheet._ignored = _ignored
        
        spreadsheet.save()
        
        spreadsheet = Spreadsheet(locations.storageDir, chosenBalance, chosenError, chosenPopsize)
        spreadsheet.load()
        
        # Assert
        self.assertTrue(hasattr(spreadsheet, "scores0"))
        self.assertTrue(np.array_equal(spreadsheet.scores0, scores0))
        self.assertTrue(hasattr(spreadsheet, "scores1"))
        self.assertTrue(np.array_equal(spreadsheet.scores1, scores1))
        self.assertFalse(hasattr(spreadsheet, "_ignored"))

class TestRandomNumberGenerator(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        seed1 = 1234
        seed2 = 1122
        
        rangeMax = 100
        numberOfIndices = 10
        
        gen1 = RandomNumberGenerator(seed1)
        gen2 = RandomNumberGenerator(seed1)
        gen3 = RandomNumberGenerator(seed2)
        
        # Act
        indices1 = gen1.generate_random_indices(rangeMax, numberOfIndices)
        indices2 = gen2.generate_random_indices(rangeMax, numberOfIndices) # RNG with same seed should be same output
        indices3 = gen3.generate_random_indices(rangeMax, numberOfIndices)
        
        # Assert
        self.assertTrue(np.array_equal(indices1, indices2))
        self.assertFalse(np.array_equal(indices2, indices3))
    
    def test_when_invalid(self):
        # Arrange
        seed1 = 1234
        gen1 = RandomNumberGenerator(seed1)
        
        # Act & Assert on error
        with self.assertRaises(ValueError):
            indices1 = gen1.generate_random_indices(10, 100) # without replacement we can't take 100 out of 10

class TestCalculatorSpacedSampling(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        length1, resultLength1 = 100, 10
        length2, resultLength2 = 10, 10
        length3, resultLength3 = 1, 1
        length4, resultLength4 = 2, 2
        length5, resultLength5 = 73, 17 # prime numbers
        length6, resultLength6 = 79, 78 # close to a prime number
        
        # Act
        indices1 = Calculator.evenly_spaced_sampling(length1, resultLength1)
        indices2 = Calculator.evenly_spaced_sampling(length2, resultLength2)
        indices3 = Calculator.evenly_spaced_sampling(length3, resultLength3)
        indices4 = Calculator.evenly_spaced_sampling(length4, resultLength4)
        indices5 = Calculator.evenly_spaced_sampling(length5, resultLength5)
        indices6 = Calculator.evenly_spaced_sampling(length6, resultLength6)
        
        # Assert
        self.assertEqual(len(indices1), resultLength1)
        self.assertEqual(len(indices2), resultLength2)
        self.assertEqual(len(indices3), resultLength3)
        self.assertEqual(len(indices4), resultLength4)
        self.assertEqual(len(indices5), resultLength5)
    
    def test_when_invalid(self):
        # Arrange
        length1, resultLength1 = "100", 10
        length2, resultLength2 = 10, 1e6
        length3, resultLength3 = 1, 2
        length4, resultLength4 = 0, 0
        
        # Act & Assert on errors
        with self.assertRaises(TypeError): # value is not an integer
            indices1 = Calculator.evenly_spaced_sampling(length1, resultLength1)
        with self.assertRaises(TypeError): # value is not an integer
            indices2 = Calculator.evenly_spaced_sampling(length2, resultLength2)
        with self.assertRaises(ValueError): # resultLength >= length
            indices3 = Calculator.evenly_spaced_sampling(length3, resultLength3)
        with self.assertRaises(ValueError): # length of 0 is nonsensical
            indices4 = Calculator.evenly_spaced_sampling(length4, resultLength4)
        pass

class TestCalculatorCounter(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        values1 = [0, 1, 2, 3]
        values2 = [0, 0, 1, 1, 2]
        values3 = [0, "0", 0, 1]
        
        expectedUnique1 = values1
        expectedCounts1 = [1, 1, 1, 1]
        
        expectedUnique2 = [0, 1, 2]
        expectedCounts2 = [2, 2, 1]
        
        expectedUnique3 = [0, 1]
        expectedCounts3 = [3, 1]
        
        # Act
        unique1, counts1 = Calculator.counter(values1)
        unique2, counts2 = Calculator.counter(values2)
        unique3, counts3 = Calculator.counter(values3) # numpy is happy casting "0" to np.int(0)
        
        # Assert
        self.assertTrue(np.array_equal(unique1, expectedUnique1))
        self.assertTrue(np.array_equal(unique2, expectedUnique2))
        self.assertTrue(np.array_equal(unique3, expectedUnique3))
        
        self.assertTrue(np.array_equal(counts1, expectedCounts1))
        self.assertTrue(np.array_equal(counts2, expectedCounts2))
        self.assertTrue(np.array_equal(counts3, expectedCounts3))

class TestCalculatorEuclideanDistance(unittest.TestCase):
    def test_original_ed_validity(self):
        "Compare a manually performed calculation to the original implementation"
        # Arrange
        alleles1 = np.array([
            [0,0,0,0],
            [0,0,0,1],
            [0,0,1,1],
            [0,1,1,1],
            [1,1,1,1]
        ])
        alleles2 = np.array([
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1]
        ])
        numAlleles = 4
        
        # Act
        value1 = np.sqrt(np.sum([
            ((4 / 4) - (0 / 4))**2, # allele '0' in alleles1 vs alleles2
            ((0 / 4) - (4 / 4))**2, # allele '1' in alleles1 vs alleles2
        ]))**4
        value2 = np.sqrt(np.sum([
            ((3 / 4) - (0 / 4))**2,
            ((1 / 4) - (4 / 4))**2,
        ]))**4
        value3 = np.sqrt(np.sum([
            ((2 / 4) - (0 / 4))**2,
            ((2 / 4) - (4 / 4))**2,
        ]))**4
        value4 = np.sqrt(np.sum([
            ((1 / 4) - (0 / 4))**2,
            ((3 / 4) - (4 / 4))**2,
        ]))**4
        value5 = np.sqrt(np.sum([
            ((0 / 4) - (0 / 4))**2,
            ((4 / 4) - (4 / 4))**2,
        ]))**4
        expectedValues = np.array([value1, value2, value3, value4, value5])
        
        edist = Calculator.euclidean_distance(alleles1, alleles2, power=4)
        
        # Assert
        self.assertTrue(np.array_equal(edist, expectedValues))
    
    def test_fast_ed_validity(self):
        "Tests an optimised ED implementation against the well-tested original version"
        rng = RandomNumberGenerator(1234)
        numReps = 1000
        for _ in range(numReps):
            # Arrange
            alleles1 = np.array([
                [ rng.integers(0, 4) for _ in range(4) ],
                [ rng.integers(0, 4) for _ in range(4) ],
                [ rng.integers(0, 4) for _ in range(4) ],
                [ rng.integers(0, 4) for _ in range(4) ],
                [ rng.integers(0, 4) for _ in range(4) ]
            ])
            alleles2 = np.array([
                [ rng.integers(0, 4) for _ in range(4) ],
                [ rng.integers(0, 4) for _ in range(4) ],
                [ rng.integers(0, 4) for _ in range(4) ],
                [ rng.integers(0, 4) for _ in range(4) ],
                [ rng.integers(0, 4) for _ in range(4) ]
            ])
            
            # Act
            orig = Calculator.euclidean_distance(alleles1, alleles2, power=4)
            new = Calculator.fast_euclidean_distance(alleles1, alleles2)
            
            # Assert
            self.assertTrue(np.array_equal(orig, new))

class TestCritic(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        cmMbp = 3.0
        edgeBp = 50
        
        genotypes1 = [[Genotype("0/0"), Genotype("0/1"), Genotype("0/1")]] # parent1, parent2, offspring
        positions1 = [0]
        positions1 = [ ("chr1", x) for x in positions1 ]
        
        genotypes2 = [[Genotype("0/0"), Genotype("0/1"), Genotype("0/1")],
                      [Genotype("0/0"), Genotype("0/1"), Genotype("0/1")]]
        positions2 = [0, 50]
        positions2 = [ ("chr1", x) for x in positions2 ]
        
        expectedQtlRanges1 = [(0, 100)] # region is 100bp long, and 0-based
        expectedQtlRanges2 = [(0, 75), (76, 150)] # region is 150bp long, and 0-based
        
        # Act
        # critic1 = Critic(locations)
        # critic2 = Critic(locations)
        
        breeder1 = Breeder()
        breeder1.establish(positions1, genotypes1, cmMbp,
                          snpMbp=int(1e6), edgeBp=edgeBp)
        
        breeder2 = Breeder()
        breeder2.establish(positions2, genotypes2, cmMbp,
                          snpMbp=int(1e6), edgeBp=edgeBp)
        
        # Assert
        self.assertEqual(breeder1.qtlRanges, expectedQtlRanges1)
        self.assertEqual(breeder2.qtlRanges, expectedQtlRanges2)
    
    def test_score(self):
        "See TestTemplate.test_when_valid for the comparison being made in this test"
        # Arrange
        y4 = np.array([
            *list(np.ones(19)),
            1.01,
            1.01,
            *list(np.ones(19))
        ])
        
        # Act
        leftCorr, rightCorr, leftWidth, rightWidth = Template.fit_gauss(y4) # near perfect
        criticScore, leftWidth, rightWidth = Critic.score(y4) # np.float64(0.02000000000000001)
        
        # Assert
        self.assertTrue(leftCorr > criticScore) # Critic penalises the lack of magnitude/prominence
        self.assertAlmostEqual(leftCorr, 1) # without Critic penalisation the correlation is very good
    
    def test_strengths(self):
        # Arrange
        scores = np.array([
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ])
        
        # Act
        strengths = Critic.scores_to_strength(scores)
        
        # Assert
        self.assertEqual(scores.shape[0], strengths.shape[0])
        for popSizeStrengths in strengths:
            self.assertTrue(np.array_equal(popSizeStrengths, [3, 2, 3, 3]))
    
    def test_run(self):
        # Arrange
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        groupSize = 15
        g1Array = np.array(
            [
                [
                    [0,0],
                    [0,0],
                    [0,0],
                    [1,1],
                    [0,0],
                    [0,0],
                    [0,0]
                ]
            ] * groupSize
        )
        g2Array = np.array(
            [
                [
                    [0,0],
                    [0,0],
                    [0,0],
                    [0,0],
                    [0,0],
                    [0,0],
                    [0,0]
                ]
            ] * groupSize
        )
        numVariants = 7
        ploidy = 2
        power = 4
        numSizes = 3
        bootstraps = 11
        
        # Arrange the Breeder object
        cmMbp = 3.0
        genotype1 = [Genotype("0/1"), Genotype("0/1"), Genotype("1/1")]
        positions = [0]
        edgeBp = 3
        positions = [ ("chr1", x) for x in positions ]
        
        breeder = Breeder()
        breeder.establish(positions, [genotype1], cmMbp,
                          snpMbp=int(1e6), edgeBp=edgeBp)
        
        # Act to emulate the Coordinator process
        g1Array = g1Array.transpose(1, 0, 2).reshape(g1Array.shape[1], -1)
        g2Array = g2Array.transpose(1, 0, 2).reshape(g2Array.shape[1], -1)
        
        futures = []
        for _i in range(numSizes):
            for _x in range(bootstraps):
                edist = Calculator.euclidean_distance(g1Array, g2Array, power)
                futures.append(edist)
        
        resultsArray = np.stack([ x for x in futures ]) # shape = (numSizes*bootstraps, numVariants)
        resultsArray = np.stack(np.split(resultsArray, numSizes)) # shape = (numSizes, bootstraps, numVariants)
        
        # Act to emulate the Critic process
        critic = Critic(locations)
        
        #startIndex, endIndex = critic.qtlRanges[0]
        startIndex, endIndex = breeder.qtlRanges[0]
        qtlED = resultsArray[:,:,startIndex:endIndex+1]
        numPopSizes, numBootstraps, numVariants = qtlED.shape
        
        scores = []
        for popSizeArray in qtlED:
            for replicateArray in popSizeArray:
                score, leftWidth, rightWidth = Critic.score(replicateArray)
                scores.append(score)
        
        scores = np.stack(np.split(np.array(scores), numPopSizes)) # shape = (popsize, bootstraps)
        strengths = Critic.scores_to_strength(scores) # shape = (popSize, 4)
        
        # Assert
        self.assertEqual(scores.shape, (numSizes, bootstraps))
        self.assertEqual(strengths.shape, (numSizes, 4))
        
        # Clean up
        cleanup()

def _generate_template_y_arrays():
    # A peak (^) shape
    y1 = np.array([
        *list(np.linspace(0, 4, 20)),
        *list(np.linspace(4, 0, 20)),
    ])
    # A min-peak-min (_^_) shape
    y2 = np.array([
        *list(np.zeros(10)),
        *list(np.linspace(0, 4, 10)),
        *list(np.linspace(4, 0, 10)),
        *list(np.zeros(10))
    ])
    # A medium-peak-min (-^_) shape
    y3 = np.array([
        *list(np.ones(10)),
        *list(np.linspace(0, 4, 10)),
        *list(np.linspace(4, 0, 10)),
        *list(np.zeros(10))
    ])
    # A small peak (--'--) shape
    y4 = np.array([
        *list(np.ones(19)),
        1.01,
        1.01,
        *list(np.ones(19))
    ])
    # A completely flat (--) shape
    y5 = np.ones(40)
    # A diagonal (\) shape
    y6 = np.linspace(4, 0, 40)
    
    return y1, y2, y3, y4, y5, y6

class TestTemplate(unittest.TestCase):
    def test_gauss(self):
        """This test used to make more sense before refactoring the Template
        to work asymmetrically. Now, it just proves that the asymmetry handling
        works as expected. The asymmetric distributions show a left and right
        score that is approximately equal to each other."""
        # Arrange
        y1, y2, y3, y4, y5, y6 = _generate_template_y_arrays()
        
        # Act
        leftCorr1, rightCorr1, leftWidth1, rightWidth1 = Template.fit_gauss(y1)
        leftCorr2, rightCorr2, leftWidth2, rightWidth2 = Template.fit_gauss(y2)
        leftCorr3, rightCorr3, leftWidth3, rightWidth3 = Template.fit_gauss(y3)
        leftCorr4, rightCorr4, leftWidth4, rightWidth4 = Template.fit_gauss(y4)
        leftCorr5, rightCorr5, leftWidth5, rightWidth5 = Template.fit_gauss(y5)
        leftCorr6, rightCorr6, leftWidth6, rightWidth6 = Template.fit_gauss(y6)
        
        # Assert
        self.assertAlmostEqual(leftCorr1, rightCorr1, places=2)
        self.assertAlmostEqual(leftCorr1, 0.99, places=1)
        self.assertAlmostEqual(leftCorr2, rightCorr2, places=2)
        self.assertAlmostEqual(leftCorr2, 0.99, places=1)
        self.assertAlmostEqual(leftCorr3, rightCorr3, places=1) # the left is a bit worse than the right
        self.assertAlmostEqual(leftCorr3, 0.99, places=1)
        self.assertAlmostEqual(leftCorr4, rightCorr4, places=2)
        self.assertAlmostEqual(leftCorr4, 0.99, places=1)
        self.assertAlmostEqual(leftCorr5, rightCorr5, places=2)
        self.assertAlmostEqual(leftCorr5, 0, places=1)
        self.assertTrue(leftCorr6 < 0)
        self.assertAlmostEqual(rightCorr6, 1, places=1)
    
    def test_focus(self):
        # Arrange
        y1, y2, y3, y4, y5, y6 = _generate_template_y_arrays()
        
        # Act
        leftFocus1, rightFocus1 = Template.fit_focus(y1)
        leftFocus2, rightFocus2 = Template.fit_focus(y2)
        leftFocus3, rightFocus3 = Template.fit_focus(y3)
        leftFocus4, rightFocus4 = Template.fit_focus(y4)
        leftFocus5, rightFocus5 = Template.fit_focus(y5)
        leftFocus6, rightFocus6 = Template.fit_focus(y6)
        
        # Assert
        self.assertEqual(leftFocus1, rightFocus1)
        self.assertEqual(leftFocus1, 1) # focus point is at the exact centre
        self.assertEqual(leftFocus2, rightFocus2)
        self.assertEqual(leftFocus2, 1)
        self.assertEqual(leftFocus3, rightFocus3)
        self.assertEqual(leftFocus3, 1)
        self.assertEqual(leftFocus4, rightFocus4)
        self.assertEqual(leftFocus4, 1)
        self.assertEqual(leftFocus5, rightFocus5)
        self.assertAlmostEqual(leftFocus5, 0, places=1) # the Template is not perfectly balanced
        self.assertEqual(leftFocus6, -1)
        self.assertAlmostEqual(rightFocus6, np.max(y6[len(y6)//2:]) / np.max(y6), places=2)

class TestMeiosisSimulator(unittest.TestCase):
    def test_basic(self):
        """Most simple test to assert that recombination is being modelled in some way,
        and that the data structures like Genome and GenomeMap are properly working
        with the composition of underlying Chromosome and ChromosomeMap classes"""
        cleanup()
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        # Arrange
        chromID1 = "chr1"
        chromID2 = "chr2"
        positions = [50, 100, 150]
        length = 200
        cmMbp = 3 * 1000000
        snpMbp = int(1e6)
        genotypeHomRef = Genotype("0/0")
        genotypeHet = Genotype("0/1")
        genotypeHomAlt = Genotype("1/1")
        
        chromMap1 = ChromosomeMap(chromID1, length, cmMbp, snpMbp, positions) # 1 SNP per bp
        chromMap2 = ChromosomeMap(chromID2, length, cmMbp, snpMbp, positions) # 1 SNP per bp
        chromosome1 = Chromosome(chromID1, positions, [genotypeHet]*len(positions), chromMap1)
        chromosome2 = Chromosome(chromID2, positions, [genotypeHet]*len(positions), chromMap2)
        
        genomeMap = GenomeMap()
        genomeMap[chromID1] = chromMap1
        genomeMap[chromID2] = chromMap2
        
        parent1 = Genome()
        parent1[chromID1] = chromosome1
        parent1[chromID2] = chromosome2
        
        parent2 = Genome()
        parent2[chromID1] = chromosome1
        parent2[chromID2] = chromosome2
        
        batchSize = 1
        ploidy = 2
        expectedShape = (batchSize, length*2, ploidy)
        
        # Act
        simulator = MeiosisSimulator(parent1, parent2, genomeMap)
        offspring = simulator.cross(batchSize=1)
        
        # Assert
        self.assertEqual(offspring.shape, expectedShape)
        self.assertTrue(not np.array_equal(offspring[0,:, 0], offspring[0,:, 1])) # strands differ
        
        uniqueAlleles = set([ tuple(x) for x in offspring[0]])
        self.assertTrue(len(uniqueAlleles) != 1) # recombination is occurring

class TestReporting(unittest.TestCase):
    def test_separate_thresholds(self):
        # Arrange
        delta = 5
        
        thresholds1 = np.array([
            ["1", 205, "1"],
            ["2", 205, "2"],
            ["3", 205, "3"]
        ], dtype=object)
        thresholds2 = np.array([
            ("1", 200, "1"),
            ("2", 205, "2"),
            ("3", 210, "3")
        ], dtype=object)
        thresholds3 = np.array([
            ("1", 200, "1"),
            ("2", 204, "2"),
            ("3", 208, "3")
        ], dtype=object)
        thresholds4 = np.array([
            ("1", 200, "1"),
            ("2", 204, "2"),
            ("3", 209, "3")
        ], dtype=object)
        thresholds5 = np.array([
            ("1", None, "1"),
            ("2", None, "2"),
            ("3", None, "3")
        ], dtype=object)
        
        expectedLabels = ["1", "2", "3"]
        
        # Act
        result1 = _separate_thresholds(thresholds1, delta)
        result2 = _separate_thresholds(thresholds2, delta)
        result3 = _separate_thresholds(thresholds3, delta)
        result4 = _separate_thresholds(thresholds4, delta)
        result5 = _separate_thresholds(thresholds5, delta)
        
        # Assert
        self.assertEqual([ x[1] for x in result1 ], [200, 205, 210])
        self.assertEqual([ x[1] for x in result2 ], [200, 205, 210])
        self.assertEqual([ x[1] for x in result3 ], [199, 204, 209])
        self.assertEqual([ int(x[1]) for x in result4 ], [199, 204, 209]) # there is some fractional 0.3333
        self.assertEqual(result5, [])

if __name__ == '__main__':
    cleanup()
    unittest.main()
