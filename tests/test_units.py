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
from modules.genome import Genome
from modules.genomemap import GenomeMap
from modules.genotype import Genotype # note that the Genotype class is implicitly tested by the TestParsing class herein
from modules.locations import Locations
from modules.parsing import parse_genotypes, parse_combination, parse_linkage, parse_qtl_encoding
from modules.population import Population

# Specify data locations
testDir = os.path.dirname(os.path.abspath(__file__))
tmpDir = os.path.join(testDir, "tmp")

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
        
        chromMap = ChromosomeMap(chromID, length, cmMbp, snpMbp) # 1 SNP per bp
        
        genotypes1 = [Genotype("0/0"), Genotype("0/1"), Genotype("1/1")]
        ploidy1 = 2
        
        genotypes2 = [Genotype("0/0/0/0"), Genotype("0/0/1/1"), Genotype("1/1/1/1")]
        ploidy2 = 4
        
        # Act
        chromosome1 = Chromosome(chromID, positions, genotypes1, chromMap)
        chromosome2 = Chromosome(chromID, positions, genotypes2, chromMap)
        
        # Assert
        self.assertEqual((1, length, ploidy1), chromosome1.array.shape)
        for i, gt in enumerate(chromosome1.array[0]):
            if i <= transitionPoints[0]:
                self.assertTrue(np.array_equal(gt, genotypeTransitions1[0]))
            elif i <= transitionPoints[1]:
                self.assertTrue(np.array_equal(gt, genotypeTransitions1[1]))
            else:
                self.assertTrue(np.array_equal(gt, genotypeTransitions1[2]))
        
        self.assertEqual((1, length, ploidy2), chromosome2.array.shape)
        for i, gt in enumerate(chromosome2.array[0]):
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
        
        chromMap1 = ChromosomeMap(chromID1, length, cmMbp, snpMbp) # 1 SNP per bp
        chromMap2 = ChromosomeMap(chromID2, length, cmMbp, snpMbp) # 1 SNP per bp
        genotypes1 = [Genotype("0/0"), Genotype("0/1"), Genotype("1/1")]
        genotypes2 = [Genotype("1/2"), Genotype("2/2"), Genotype("2/3")]
        
        # Act
        chromosome1 = Chromosome(chromID1, positions, genotypes1, chromMap1)
        chromosome2 = Chromosome(chromID2, positions, genotypes2, chromMap2)
        
        genome = Genome()
        genome.add(chromosome1)
        genome.add(chromosome2)
        
        # Assert
        self.assertEqual(chromosome1.array.shape[1] + chromosome2.array.shape[1], genome.array.shape[1])
        ongoingCount = 0
        for chromosome in (chromosome1, chromosome2):
            for gt in chromosome.array[0]:
                genomeGt = genome.array[0][ongoingCount]
                self.assertTrue(np.array_equal(gt, genomeGt))
                ongoingCount += 1
    
    def test_when_invalid(self):
        # Arrange
        chromID = "chr1"
        positions = [50, 100, 150]
        length = 200
        cmMbp = 1
        snpMbp = int(1e6)
        
        transitionPoints = [(positions[0]+positions[1])//2, (positions[1]+positions[2])//2]
        genotypeTransitions1 = [[0,0], [0,1], [1,1]]
        genotypeTransitions2 = [[0,0,0,0], [0,0,1,1], [1,1,1,1]]
        
        chromMap = ChromosomeMap(chromID, length, cmMbp, snpMbp) # 1 SNP per bp
        genotypes1 = [Genotype("0/0"), Genotype("0/1"), Genotype("1/1")]
        genotypes2 = [Genotype("0/0/0/0"), Genotype("0/0/1/1"), Genotype("1/1/1/1")]
        
        # Act
        chromosome1 = Chromosome(chromID, positions, genotypes1, chromMap)
        chromosome2 = Chromosome(chromID, positions, genotypes2, chromMap)
        genome = Genome()
        genome.add(chromosome1)
        with self.assertRaises(ValueError): # ploidy incompatibility
            genome.add(chromosome2)

class TestChromosomeMap(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        chromID = "chr1"
        cmMbp = 3
        length1 = 100000
        snpMbp1 = 1000 # should equate to 1 SNP every 1000 bp
        
        length2 = 100
        snpMbp2 = int(1e6) # should equate to 1 SNP every 1 bp
        
        length3 = int(1e6)
        snpMbp3 = 1 # should equate to 1 SNP every 1 Mbp
        
        # Act
        chromMap1 = ChromosomeMap(chromID, length1, cmMbp, snpMbp1)
        nrow1 = len(chromMap1.df)
        expectedNrow1 = length1 // (int(1e6)//snpMbp1)
        
        chromMap2 = ChromosomeMap(chromID, length2, cmMbp, snpMbp2)
        nrow2 = len(chromMap2.df)
        expectedNrow2 = length2 // (int(1e6)//snpMbp2)
        
        chromMap3 = ChromosomeMap(chromID, length3, cmMbp, snpMbp3)
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
        snpMbp = 1
        
        # Act & Assert on error
        with self.assertRaises(ValueError): # snpMbp is too sparse for the short length
            chromMap = ChromosomeMap(chromID, length, cmMbp, snpMbp)

class TestGenomeMap(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        chromID1 = "chr1"
        chromID2 = "chr2"
        chromID3 = "chr3"
        cmMbp = 3
        length1 = 100000
        snpMbp1 = 1000 # should equate to 1 SNP every 1000 bp
        
        length2 = 100
        snpMbp2 = int(1e6) # should equate to 1 SNP every 1 bp
        
        length3 = int(1e6)
        snpMbp3 = 1 # should equate to 1 SNP every 1 Mbp
        
        # Act
        chromMap1 = ChromosomeMap(chromID1, length1, cmMbp, snpMbp1)
        chromMap2 = ChromosomeMap(chromID2, length2, cmMbp, snpMbp2)
        chromMap3 = ChromosomeMap(chromID3, length3, cmMbp, snpMbp3)
        
        genomeMap = GenomeMap()
        genomeMap.add(chromMap1)
        genomeMap.add(chromMap2)
        genomeMap.add(chromMap3)
        
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
        individual1 = np.array([0, 1, 0, 1, 0, 1]).reshape(1, 3, 2) # 3 heterozygous genotypes
        individual2 = np.array([1, 2, 1, 2, 1, 2]).reshape(1, 3, 2) # 3 heterozygous genotypes
        
        os.makedirs(tmpDir, exist_ok=True)
        tmpFile = os.path.join(tmpDir, "tmp.npy")
        
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

class TestBreeder(unittest.TestCase):
    def test_when_valid(self):
        # Arrange
        cmMbp = 3.0
        genotype1 = [Genotype("0/0"), Genotype("0/1"), Genotype("0/1")] # parent1, parent2, offspring
        positions = [0]
        edgeBp = 50
        positions = [ ("chr1", x + edgeBp) for x in positions ]
        combinationEvaluator = Combination("1")
        
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        # Act & Assert (no error is a pass)
        breeder = Breeder()
        breeder.establish(positions, [genotype1], cmMbp,
                          snpMbp=int(1e6), edgeBp=edgeBp)
        breeder.produce_progeny(locations, combinationEvaluator, batchSize=100, minimumGroupSize=1, seed=1234)
        
        # Clean up
        os.unlink(locations.group1Npy)
        os.unlink(locations.group2Npy)
    
    def test_when_invalid(self):
        # Arrange
        cmMbp = 3.0
        genotype1 = [Genotype("0/0"), Genotype("1/1"), Genotype("0/1")] # parent1, parent2, offspring
        positions = [0]
        edgeBp = 50
        positions = [ ("chr1", x + edgeBp) for x in positions ]
        combinationEvaluator = Combination("1")
        
        os.makedirs(tmpDir, exist_ok=True)
        locations = Locations(tmpDir)
        
        # Act
        breeder = Breeder()
        breeder.establish(positions, [genotype1], cmMbp,
                          snpMbp=int(1e6), edgeBp=edgeBp)
        
        # Assert
        with self.assertRaises(Exception): # all offspring end up in group1 with none in group2
            breeder.produce_progeny(locations, combinationEvaluator, batchSize=10, minimumGroupSize=1, seed=1234)

if __name__ == '__main__':
    unittest.main()
