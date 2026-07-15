# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ast

class Combination:
    '''
    Properties:
        raw -- the original string used to instantiate a Combination object
        interpreted -- a string representation of how ast interpreted the original raw string
        tree -- an ast object obtained by ast.parse(raw, mode='eval')
        numbers -- a list of integers representing the node labels that can be substituted
                   when evaluating a combination of booleans
        isCombination -- a boolean flag for object type validation
    Methods:
        evaluate -- 
    '''
    VALID_OPERATORS = ["and", "or"]
    
    def __init__(self, value):
        self.raw = value
        self.tree = value
        self.isCombination = True # object type validator
    
    @property
    def tree(self):
        return self._tree
    
    @tree.setter
    def tree(self, value):
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
            self._tree = ast.parse(" ".join(substitution), mode="eval")
        except:
            raise SyntaxError(f"Syntax of combination string '{value}' was not understood by the Python ast library. " +
                              "Double check that your input is formatted according to popquis expectations.")
    
    @property
    def numbers(self):
        return self._numbers
    
    @numbers.setter
    def numbers(self, value):
        self._numbers = value
    
    @property
    def interpreted(self):
        return ast.unparse(self.tree)
    
    def evaluate(self, variableDict):
        # Validate variableDict type
        if not isinstance(variableDict, dict):
            raise TypeError("Combination.evaluate must receive a dict, not " + 
                            f"'{type(variableDict).__name__}'")
        
        # Validate variableDict compatibility
        if len(variableDict) == 0:
            raise ValueError("The dictionary provided for evaluation appears to be empty.")
        if not all([ x in variableDict for x in self.numbers ]):
            raise ValueError(f"All variable numbers in this Combination object ({self.numbers}) " +
                             f"must be found within the variableDict provided for evaluation " + 
                             f"({variableDict.keys()})")
        if any([ x not in self.numbers for x in variableDict.keys() ]):
            raise ValueError(f"There is a mismatch between the variable keys found within the dictionary " +
                             f"provided for evaluation ({variableDict.keys()}) and the Combination " + 
                             f"object ({self.numbers})")
        
        # Substitute variableDict number keys with qtl variable identifiers
        qtlVariableDict = { f"qtl{key}":value for key, value in variableDict.items() }
        
        # Perform variable substitution and return the combination evaluation
        expressionTree = CombinationEvaluator(qtlVariableDict)
        expressionString = ast.unparse(expressionTree.visit(self.tree))
        return eval(expressionString)
    
    def __repr__(self):
        return "<Combination object;raw='{0}';interpreted='{1}';numbers={2}>".format(
            self.raw,
            self.interpreted,
            self.numbers
        )

class CombinationEvaluator(ast.NodeTransformer):
    def __init__(self, variableDict):
        self.variableDict = variableDict
        super().__init__()
    
    def visit_Name(self, node):
        if node.id in self.variableDict:
            return ast.Constant(value=self.variableDict[node.id])
        return node
    
    def __repr__(self):
        return "<CombinationEvaluator object;variableDict='{0}'>".format(
            self.variableDict
        )
