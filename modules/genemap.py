# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd

class Genemap:
    '''
    Properties:
        chromID -- a string identifying the chromosome.
        length -- an integer value giving the chromosome length in bp.
        cmMbp -- a float value giving the centiMorgan per Mbp.
        snpMbp -- an integer giving the approximate number of SNPs to be spaced evenly
                 across each Mbp of genome length; default is 1000.
        genemap -- a pandas DataFrame suitable for chromax handling.
    Methods:
        generate -- uses parameter values to produce the genemap DataFrame.
    '''
    def __init__(self, chromID, length, cmMbp, snpMbp):
        self.chromID = chromID
        self.length = length
        self.cmMbp = cmMbp
        self.snpMbp = snpMbp
        self.generate()
        self.isGenemap = True # object type validator
    
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
    
    def generate(self):
        mapList = [["CHR.PHYS", "cM", "Trait"]]
        for i in range(0, self.length // self.snpMbp):
            physicalPosition = i * self.snpMbp
            cMPosition = (physicalPosition / 1000000) * self.cmMbp
            mapList.append([self.chromID, cMPosition, 0.01])
        self.genemap = pd.DataFrame(mapList[1:], columns=mapList[0])
    
    def __repr__(self):
        lengthRepr = f"{(self.length / 1e6)} Mbp" if self.length >= 1e6 else self.length
        return "<Genemap object;chromID={0};length={1};cmMbp={2};snpMbp={3}>".format(
            self.chromID,
            lengthRepr,
            self.cmMbp,
            self.snpMbp
        )
