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
        # Make input lowercase for compatibility with ast library interpretation
        value = value.lower()
        
        # Ensure that combination components (e.g., operators) are compatible with popquis
        components = [ x.strip("()") for x in value.split() ]
        numbers = []
        for component in components:
            if component == "": # blanks are stripped by ast internally
                continue
            if component.isdigit(): # numbers should refer to QTL index; will be validated later
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
        
        # Load combination with ast library interpretation
        try:
            self._tree = ast.parse(value, mode="eval")
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
    
    def __repr__(self):
        return "<Combination object;raw='{0}';interpreted='{1}';numbers={2}>".format(
            self.raw,
            self.interpreted,
            self.numbers
        )
