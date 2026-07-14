import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from combination import Combination
from genotype import Genotype
from errors import IncompatibleGenotypeError

def parse_genotypes(gtString):
    '''
    Parameters:
        gtString -- a string with format akin to '0/0:1/1:0/1' representing
                    'parent1:parent2:group1'
    Returns:
        parent1, parent2, offspring -- a Genotype object for each genotype ordered as per
                                       the original gtString layout
    '''
    # Break string apart into list
    sampleGenotypes = gtString.split(":")
    if len(sampleGenotypes) != 3:
        raise ValueError(f"'{gtString}' value is expected to have two colons separating three VCF-encoded genotypes")
    
    # Parse each sample genotype into a Genotype object instance
    genotypes = parent1, parent2, offspring = [ Genotype(x) for x in sampleGenotypes ]
    
    # Validate each Genotype
    lastPloidy = None
    for gt in genotypes:
        if gt.ploidy % 2 != 0:
            raise ValueError(f"The ploidy of genotype '{genotype}' is odd; only even ploidy numbers " +
                             "(e.g., 2, 4, ...) can be handled by popquis")
        if lastPloidy is not None:
            if gt.ploidy != lastPloidy:
                raise ValueError(f"Genotype' {genotype}' in '{genotypes}' has a different ploidy than " +
                                 f"expected; all genotypes must have the same ploidy i.e., {lastPloidy} alleles")
        lastPloidy = gt.ploidy
    
    # Validate that offspring allele is a possible combination of parental alleles
    possibleOffspring = parent1.cross(parent2)
    if offspring not in possibleOffspring:
        raise IncompatibleGenotypeError(f"Genotype '{offspring.raw}' is not a possible combination of the " +
                                        f"parental genotypes '{parent1.raw}' and '{parent2.raw}")
    
    # All genotypes are validated; return
    return parent1, parent2, offspring

def parse_combination(combinationString, numQTLs):
    '''
    Parameters:
        combinationString -- a string with format akin to '1 AND (2 OR 3)' providing a logical
                             framework for handling QTL combined inheritance.
        numQTLs -- an integer indicating how many QTLs are being combined, for validation purpose
    Returns:
        combinationEvaluator -- an ast (Abstract Syntax Tree) object capable of determining whether
                                the QTLs inherited by a simulated individual would categorise it into
                                group1 or group2. 
    '''
    combinationEvaluator = Combination(combinationString)
    numFoundQTLs = len(combinationEvaluator.numbers)
    if numFoundQTLs != numQTLs:
        raise ValueError(f"The -c combination string '{combinationString}' should indicate {numQTLs} QTLs, " +
                         f"but instead refers to {numFoundQTLs} QTLs.")
    return combinationEvaluator

def parse_linkage(linkageList, weakDistance, moderateDistance, strongDistance, numQTLs):
    '''
    Parameters:
        linkage -- a list of strings with possible options being ["none", "weak", "moderate", "strong"]
        weak/moderate/strongDistance -- integer values providing the basepairs distance that each
                                        linkage value should correspond to; 'none' signals a new
                                        chromosome and does not need a basepairs value.
    '''
    if numQTLs > 1:
        if len(linkageList) != numQTLs-1:
            raise ValueError(f"{numQTLs} values were given to -q, which means we expect {numQTLs-1} " +
                             f"values to be given to -l; you instead gave {len(linkageList)}")
    
    # Seed first QTL position
    chromNumber = 1
    nucPosition = 0
    qtlPositions = [(f"chr{chromNumber}", nucPosition)]
    
    # Add each sequential QTL position
    for linkage in linkageList:
        if linkage == "none":
            chromNumber += 1
            nucPosition = 0
            qtlPositions.append((f"chr{chromNumber}", nucPosition))
        elif linkage == "weak":
            nucPosition += weakDistance
            qtlPositions.append((f"chr{chromNumber}", nucPosition))
        elif linkage == "moderate":
            nucPosition += moderateDistance
            qtlPositions.append((f"chr{chromNumber}", nucPosition))
        elif linkage == "strong":
            nucPosition += strongDistance
            qtlPositions.append((f"chr{chromNumber}", nucPosition))
        else:
            raise ValueError(f"parse_linkage() does not recognise '{linkage}' as a valid linkage option")
    
    return qtlPositions

def parse_qtl_encoding(qtls, combination, linkage, weakDistance, moderateDistance, strongDistance):
    '''
    Parameters:
        qtls -- a list as obtained from popquis -q, with each value being a string with 
                format akin to: 'parent1:parent2:group1'
        combination -- a string as obtained from popquis -c with format akin to:
                       '1 AND (2 OR 3)'
        linkage -- a list as obtained from popquis -l, with each value being a string with
                   possible options being: ["none", "weak", "moderate", "strong"]
        weak/moderate/strongDistance -- integer values providing the basepairs distance that each
                                        linkage value should correspond to
    Returns:
        qtlGenotypes -- a list of Genotype objects resulting from parsing of the input qtls
        combinationEvaluator -- a Combination object able to segregate simulated samples according
                                to their genotype inheritance
        qtlPositions -- a list of tuples with format akin to: ('chrom1', basepair_position)
    '''
    numQTLs = len(qtls)
    
    # Handle qtls
    lastPloidy = None
    qtlGenotypes = []
    for i in range(len(qtls)):
        thisQTL = qtls[i]
        genotypes = parent1, parent2, group1 = parse_genotypes(thisQTL)
        if lastPloidy is not None:
            if parent1.ploidy != lastPloidy:
                raise ValueError("QTLs provided to popquis -q must all be of the same ploidy level; " + 
                                 f"{thisQTL} differs from {qtls[i-1]}")
        qtlGenotypes.append(genotypes)
        lastPloidy = parent1.ploidy
    
    # Handle combination
    combinationEvaluator = parse_combination(combination, numQTLs)
    
    # Handle linkage
    qtlPositions = parse_linkage(linkageList, weakDistance, moderateDistance, strongDistance, numQTLs)
    
    return qtlGenotypes, combinationEvaluator, qtlPositions
