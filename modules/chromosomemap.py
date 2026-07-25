# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd

class ChromosomeMap:
    '''
    Args:
        chromID -- a string identifying the chromosome.
        length -- an integer value giving the chromosome length in bp.
        cmMbp -- a float value giving the centiMorgan per Mbp.
        snpMbp -- an integer giving the approximate number of SNPs to be spaced evenly
                 across each Mbp of genome length; default is 1000.
        markerLocations -- a list or set of integers representing the bp position of a
                           marker in this chromosome
    Properties:
        df -- a pandas DataFrame representing a genetic map with physical and centimorgans
              distances as well as indication of marker locations
        markers -- a convenience function to return the rows of self.df that are
                   a marker
    '''
    def __init__(self, chromID, length, cmMbp, snpMbp, markerLocations):
        self.chromID = chromID
        self.length = length
        self.cmMbp = cmMbp
        self.snpMbp = snpMbp
        self._create_df(set(markerLocations)) # sets self.df
        self.isChromosomeMap = True # object type validator
    
    @property
    def chromID(self):
        return self._chromID
    
    @chromID.setter
    def chromID(self, value):
        if not isinstance(value, str):
            raise TypeError(f"chromID must be a str, not '{type(value).__name__}'")
        if value == "":
            raise ValueError("chromID must not be blank")
        self._chromID = value
    
    @property
    def length(self):
        return self._length
    
    @length.setter
    def length(self, value):
        if not isinstance(value, int):
            raise TypeError(f"length must be an int, not '{type(value).__name__}'")
        if value <= 0:
            raise ValueError("length must be >= zero")
        self._length = value
    
    @property
    def cmMbp(self):
        return self._cmMbp
    
    @cmMbp.setter
    def cmMbp(self, value):
        if not (isinstance(value, int) or isinstance(value, float)):
            raise TypeError(f"cmMbp must be an int or float, not '{type(value).__name__}'")
        if value <= 0:
            raise ValueError("cmMbp must be >= zero")
        self._cmMbp = value
    
    @property
    def snpMbp(self):
        return self._snpMbp
    
    @snpMbp.setter
    def snpMbp(self, value):
        if not isinstance(value, int):
            raise TypeError(f"snpMbp must be an int, not '{type(value).__name__}'")
        if value <= 0:
            raise ValueError("snpMbp must be >= zero")
        self._snpMbp = value
    
    @property
    def markers(self):
        return self.df[self.df["Marker"]]
    
    def _create_df(self, markerLocations):
        # Validate markerLocations
        if not (isinstance(markerLocations, list) or isinstance(markerLocations, set)):
            raise TypeError(f"markerLocations must be a list or set, not '{type(markerLocations).__name__}'")
        if len(markerLocations) == 0:
            raise ValueError("markerLocations must contain at least one marker position")
        for _v in markerLocations:
            if not isinstance(_v, int):
                raise TypeError(f"markerLocations must contain int types, not '{type(_v).__name__}'")
        
        # Make sure our length and snpMbp parameters lead to an interpretable result
        stepSize = int(1e6) // self.snpMbp
        if stepSize > self.length:
            raise ValueError(f"Genetic map with '{self.snpMbp}' SNPs per Mbp is too sparse for a " +
                             f"chromosome of {self.length} bp in length (i.e., we do not find any " +
                             "SNPs within a section of genome this short)")
        
        foundMarker = False
        mapList = [["CHR.PHYS", "bp", "cM", "Trait", "Marker"]]
        for bp in range(0, self.length, stepSize):
            cM = (bp / int(1e6)) * self.cmMbp
            isMarker = bp in markerLocations
            mapList.append([self.chromID, bp, cM, 0.01, isMarker])
            if isMarker:
                foundMarker = True
        
        if not foundMarker:
            raise ValueError(f"Did not find a marker when creating ChromosomeMap for '{self.chromID}'")
        self.df = pd.DataFrame(mapList[1:], columns=mapList[0])
    
    def __repr__(self):
        lengthRepr = f"{(self.length / 1e6)} Mbp" if self.length >= 1e6 else self.length
        return "<ChromosomeMap object;chromID='{0}';length={1};cmMbp={2};snpMbp={3}>".format(
            self.chromID,
            lengthRepr,
            self.cmMbp,
            self.snpMbp
        )
