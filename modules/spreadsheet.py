# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys

import numpy as np

class Spreadsheet:
    '''
    Wrapper around a static .npy file to allow for storage of a numpy array and associated
    metadata resulting from statistical calculations.
    
    Args:
        storageDir -- a string indicating the location of pre-existing .npz files,
                      and/or to a location that is writeable when creating new npz files
        popBalance -- a float giving the balance between group1:group2
        phenotypeError -- a float giving the proportion of each group1 which is incorrectly assigned
        popSizes -- an iterable containing integers giving population size increments
    Attributes:
        fileName -- a string indicating the location of the .npz storing the data of this Spreadsheet
        ed -- None, or after using self.load(), it may instead be a numpy array
        r2 -- None, or after using self.load(), it may instead be a numpy array
    Methods:
        save -- save the data in self to self.fileName
        load -- loads data out of the self.fileName file into self.array and also cross-checks the
                existing popBalance and phenotypeError and popSizes values for compatibility
    '''
    def __init__(self, storageDir, popBalance, phenotypeError, popSizes):
        self.storageDir = storageDir
        self.popBalance = popBalance
        self.phenotypeError = phenotypeError
        self.popSizes = popSizes
        self.ed = None
        self.r2 = None
        self.isSpreadsheet = True # object type validator
    
    @property
    def storageDir(self):
        return self._storageDir
    
    @storageDir.setter
    def storageDir(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Spreadsheet object expects storageDir to be a str, not '{type(value).__name__}'")
        
        value = os.path.abspath(value)
        if not os.path.isdir(value):
            raise FileNotFoundError(f"Spreadsheet object was given the storageDir '{value}' which appears to " +
                                    "either not exist, or not be a directory.")
        
        self._storageDir = value
    
    @property
    def fileName(self):
        return os.path.join(self.storageDir, f"{self.popBalance}_{self.phenotypeError}.npz")
    
    @property
    def shape(self):
        if self.ed is None: # ed is always set before r2
            return None
        return self.ed.shape # r2 has the same shape as ed
    
    def save(self):
        np.savez(self.fileName, **{key:value for key, value in self.__dict__.items() if not key.startswith("_")})
    
    def load(self):
        if os.path.isfile(self.fileName):
            with np.load(self.fileName, allow_pickle=True) as data:
                # Load unfixed variables
                EXPECTED_UNFIXED = ["fitted", "rsq"]
                for key in data.files:
                    if any([ key.startswith(prefix) for prefix in EXPECTED_UNFIXED ]):
                        setattr(self, key, data[key])
                
                # Load static expected variables
                _ed = data["ed"]
                if _ed.ndim == 0:
                    _ed = None
                self.ed = _ed
                
                _popBalance = float(data["popBalance"])
                if _popBalance != self.popBalance:
                    raise ValueError(f"Spreadsheet file at '{self.fileName}' should have " + 
                                     f"popBalance=={self.popBalance} but instead is =={_popBalance}")
                
                _phenotypeError = float(data["phenotypeError"])
                if _phenotypeError != self.phenotypeError:
                    raise ValueError(f"Spreadsheet file at '{self.fileName}' should have " + 
                                     f"phenotypeError=={self.phenotypeError} but instead is =={_phenotypeError}")
                
                _popSizes = data["popSizes"]
                if not np.array_equal(_popSizes, self.popSizes):
                    raise ValueError(f"Spreadsheet file at '{self.fileName}' should have " + 
                                     f"popSizes=={self.popSizes} but instead is =={_popSizes}")
    
    def __repr__(self):
        return "<Spreadsheet object;storageDir='{0}';popBalance={1};phenotypeError={2};hasED={3};hasR2={4}>".format(
            self.storageDir,
            self.popBalance,
            self.phenotypeError,
            self.ed is not None,
            self.r2 is not None
        )
