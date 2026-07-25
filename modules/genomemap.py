# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd

class GenomeMap:
    '''
    Properties:
        chromIDs -- a list of strings indicating which ChromosomeMap objects this Class has ingested.
        df -- a pandas DataFrame suitable for chromax handling.
        markers -- a convenience function to return the rows of self.df that are
                   a marker
    '''
    def __init__(self):
        self.chromosomes = {}
        self._df = None
        self._wasUpdated = False
        self.isGenomeMap = True # object type validator
    
    @property
    def chromIDs(self):
        return list(self.chromosomes.keys())
    
    @property
    def df(self):
        # Create the chromosome-spanning df on demand
        if self._df is None or self._wasUpdated:
            _df = None
            for chromID, chromosomeMap in self.chromosomes.items():
                if _df is None:
                    _df = chromosomeMap.df.copy()
                else:
                    _df = pd.concat((_df, chromosomeMap.df))
                    _df.reset_index(drop=True, inplace=True)
            self._df = _df
            self._wasUpdated = False
        return self._df
    
    @property
    def markers(self):
        if self.df is not None:
            return self.df[self.df["Marker"]]
        else:
            return None
    
    def __setitem__(self, key, value):
        # Validate object type
        if not hasattr(value, "isChromosomeMap"):
            raise TypeError(f"GenomeMap object can only receive a ChromosomeMap object, not '{type(chromMapObj).__name__}'")
        
        # Make sure we haven't ingested this ChromosomeMap already
        if key in self.chromosomes:
            raise KeyError(f"'{key}' already stored in this GenomeMap")
        self.chromosomes[key] = value
        self._wasUpdated = True
    
    def __getitem__(self, key):
        if key not in self.chromosomes:
            return KeyError(f"'{key}' not in this GenomeMap")
        return self.chromosomes[key]
    
    def __iter__(self):
        for value in self.chromosomes.values():
            yield value
    
    def __repr__(self):
        return "<GenomeMap object;chromIDs={0}>".format(
            self.chromIDs
        )
