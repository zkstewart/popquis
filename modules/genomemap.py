# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd

class GenomeMap:
    '''
    Properties:
        chromIDs -- a set of strings indicating which ChromosomeMap objects this Class has ingested.
        df -- a pandas DataFrame suitable for chromax handling.
    Methods:
        add -- ingests a ChromosomeMap object and creates this GenomeMap object out of its composition
    '''
    def __init__(self):
        self.chromIDs = set()
        self.df = None
        self.isGenomeMap = True # object type validator
    
    def add(self, chromMapObj):
        # Validate object type
        if not hasattr(chromMapObj, "isChromosomeMap"):
            raise TypeError(f"GenomeMap object can only receive a ChromosomeMap object, not '{type(chromMapObj).__name__}'")
        
        # Make sure we haven't ingested this ChromosomeMap already
        if chromMapObj.chromID in self.chromIDs:
            raise ValueError(f"GenomeMap object has already had '{chromMapObj.chromID}' added into it.")
        
        # Store the information
        if self.df is None:
            self.df = chromMapObj.df.copy()
        else:
            self.df = pd.concat((self.df, chromMapObj.df))
        self.chromIDs.add(chromMapObj.chromID)
    
    def __repr__(self):
        return "<GenomeMap object;chromIDs={0}>".format(
            self.chromIDs
        )
