# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ast

from copy import deepcopy

class Combination:
    '''
    Attributes:
        raw -- the original string used to instantiate a Combination object
        interpreted -- a string representation of how ast interpreted the original raw string
        numbers -- a list of integers representing the QTL positions that can be substituted
                   when evaluating a combination of booleans
        isCombination -- a boolean flag for object type validation
    Methods:
        evaluate -- 
    '''
    VALID_OPERATORS = ["and", "or"]
    
    def __init__(self, value):
        self.raw = value
        self.isCombination = True # object type validator
        
        # Immediately reject empty values
        if value.strip() == "":
            raise ValueError(f"Combination string is empty and cannot be interpreted.")
        
        # Make input lowercase for compatibility with ast library interpretation
        value = value.lower()
        
        # Ensure that combination components (e.g., operators) are compatible with popquis
        components = [ x for x in value.replace("(", "( ").replace(")", " )").split() ] # separate parentheses from numbers
        numbers = []
        for component in components:
            if component == "": # blanks are stripped by ast internally
                continue
            elif component == "(" or component == ")": # don't need to validate
                continue
            elif component.isdigit(): # numbers should refer to QTL index; will be validated later
                try:
                    intcomponent = int(component)
                except ValueError:
                    raise InvalidGenotypeError(f"Combination string '{value}' has a component '{intcomponent}' which " +
                                               "was expected to be a number, but cannot be interpreted as an integer. " +
                                               "Make sure this is a plain integer.")
                numbers.append(intcomponent)
            elif not component in Combination.VALID_OPERATORS:
                raise ValueError(f"The component '{component}' of '{value}' is not recognised. " +
                                 "Make sure you only use QTL numbers and operators including: " +
                                 ", ".join(Combination.VALID_OPERATORS))
        
        # Check that the QTL numbers are provided in the expected sequential ordering
        expectedNumbers = list(range(1, len(numbers)+1))
        if numbers != expectedNumbers:
            # Error type 1: numbers aren't ordered in ascending +1 fashion
            if any([ numbers[i] != (numbers[i+1]-1) for i in range(0, len(numbers)-1)]):
                raise ValueError(f"QTL numbers in the combination string '{value}' must be sequential and in ascending order!")
            
            # Error type 2: numbers don't start at 1
            elif numbers[0] != expectedNumbers[0]:
                raise ValueError(f"QTL numbers in the combination string '{value}' should start at '{expectedNumbers[0]}' " + 
                                 f"but instead start at '{numbers[0]}'")
            
            # Error type 3: something else
            else:
                raise ValueError(f"QTL numbers in the combination string '{value}' should be akin to {expectedNumbers} " + 
                                 f"when parsed by popquis, but instead appear as {numbers}")
        self.numbers = numbers # expose this value for callers to easily tell what drop-in variables exist
        
        # Convert numbers into variable names
        substitution = []
        for component in components:
            if component.isdigit():
                substitution.append("qtl" + component)
            else:
                substitution.append(component)
        
        # Load combination with ast library interpretation
        try:
            tree = ast.parse(" ".join(substitution), mode="eval")
        except:
            raise SyntaxError(f"Syntax of combination string '{value}' was not understood by the Python ast library. " +
                              "Double check that your input is formatted according to popquis expectations.")
        
        # Unparse the interpreted expression
        self.interpreted = ast.unparse(tree)
        
        # Convert expression into a format string for easier variable substitution
        self._formatString = self.interpreted
        for number in self.numbers[::-1]:
            self._formatString = self._formatString.replace(f"qtl{number}", "{}")
    
    def evaluate(self, variables):
        # Validate variables type
        if not (isinstance(variables, list) or isinstance(variables, tuple)):
            raise TypeError("Combination.evaluate must receive a list or tuple, not " + 
                            f"'{type(variables).__name__}'")
        
        # Validate variables compatibility
        if len(variables) == 0:
            raise ValueError("The list provided for evaluation appears to be empty.")
        elif len(variables) != len(self.numbers):
            raise ValueError("The list provided for evaluation has a different number of QTLs " + 
                             f"({len(variables)}) to what is expected ({len(self.numbers)})")
        if not all([ isinstance(x, bool) for x in variables ]):
            raise ValueError("The list provided for evaluation must contain bool objects")
        
        # Format and evaluate the expression
        expressionString = self._formatString.format(*variables)
        return eval(expressionString)
    
    def __repr__(self):
        return "<Combination object;raw='{0}';interpreted='{1}';numbers={2}>".format(
            self.raw,
            self.interpreted,
            self.numbers
        )
