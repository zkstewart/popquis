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
        array -- None, or after using self.load(), a numpy array
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
        self.array = None
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
        if self.array is None:
            return None
        return self.array.shape
    
    def save(self):
        np.savez(self.fileName, array=self.array, popBalance=self.popBalance,
                 phenotypeError=self.phenotypeError, popSizes=self.popSizes)
    
    def load(self):
        if os.path.isfile(self.fileName):
            with np.load(self.fileName, allow_pickle=True) as data:
                _array = data["array"]
                if _array.ndim == 0:
                    _array = None
                self.array = _array
                
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
        return "<Spreadsheet object;storageDir={0};popBalance={1};phenotypeError={2}>".format(
            self.storageDir,
            self.popBalance,
            self.phenotypeError
        )
