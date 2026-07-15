#!/usr/bin/env python3
# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.parsing import parse_genotypes, parse_combination, parse_linkage
from modules.errors import InvalidGenotypeError, IncompatibleGenotypeError
from modules.combination import Combination

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
        
        # Assert
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
        
        # Act & Assert on anticipated errors
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
        
        # Act & Assert on anticipated errors
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

class TestCombinationEvaluation(unittest.TestCase):
    def test_evaluation_when_valid(self):
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
        
        # Act & Assert in loop
        for combinationStr, evalStr in zip(combinationStrs, evalStrs):
            for qtlBoolCombo in qtlBoolCombos:
                combination = Combination(combinationStr)
                evaluatorResult = combination.evaluate({
                    i+1:qtlBoolCombo[i] for i in range(len(qtlBoolCombo))
                })
                evalStrResult = eval(evalStr.format(*qtlBoolCombo))
                self.assertEqual(evaluatorResult, evalStrResult, 
                                 f"'{combinationStr}' gives inconsistent output")
    
    def test_evaluation_when_invalid(self):
        # Arrange
        combinationStrs = [
            "1 AND 2 OR 3",
            "1 AND (2 OR 3)",
            "(1 AND 2) OR 3",
            "(1 OR 2 OR 3)",
            "((1 AND 2) OR 3)"
        ]
        valueErrorDicts = [
            {1: True, 2: True}, # too few numbers
            {1: True, 2: True, 3: True, 4: True}, # too many numbers
            {}, # empty dict
            {2: True, 3: True, 4: True}, # mismatching numbers
            {"1": True, "2": True, "3": True} # key type mismatch
        ]
        typeErrorDicts = [
            [True, True, True] # object type mismatch
        ]
        
        # Act & Assert in loop on errors
        for combinationStr in combinationStrs:
            combination = Combination(combinationStr)
            for variableDict in valueErrorDicts:
                with self.assertRaises(ValueError):
                    evaluatorResult = combination.evaluate(variableDict)
            for variableDict in typeErrorDicts:
                with self.assertRaises(TypeError):
                    evaluatorResult = combination.evaluate(variableDict)

if __name__ == '__main__':
    unittest.main()
