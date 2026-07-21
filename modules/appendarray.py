# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

import numpy as np

from npy_append_array import NpyAppendArray

class AppendArray:
    '''
    Wrapper around the npy-append-array functionalities to allow for progressive growth
    of a numpy array.
    
    Args:
        fileName -- a string indicating the location of a pre-existing npy file,
                    or to a location that is writeable to create a new npy file
    Attributes:
        data -- None, or after using self.load(), a memory-mapped npy file handle
        shape -- None, or after using self.load(), the underlying numpy array's shape
    Methods:
        store -- stores an array
        load -- memory maps the npy file to enable retrieval of stored arrays
    '''
    def __init__(self, fileName):
        self.fileName = fileName
        self.data = None
        self.isAppendArray = True # object type validator
    
    @property
    def fileName(self):
        return self._fileName
    
    @fileName.setter
    def fileName(self, value):
        if not isinstance(value, str):
            raise TypeError(f"AppendArray object expects fileName to be a str, not '{type(value).__name__}'")
        
        value = os.path.abspath(value)
        if os.path.isfile(value):
            pass
        elif os.path.exists(value):
            raise FileNotFoundError(f"AppendArray object was given the fileName '{value}' which appears to " +
                                    "exist but not be a file? Move or rename whatever resides here and try again.")
        else:
            parentDir = os.path.dirname(value)
            if os.path.isdir(parentDir):
                pass
            elif not os.path.exists(parentDir):
                raise NotADirectoryError(f"AppendArray object cannot create fileName '{value}' since its parent " +
                                         f"location '{parentDir}' does not exist. Create this location first.")
            else:
                raise NotADirectoryError(f"AppendArray object cannot create fileName '{value}' since its parent " +
                                         f"location '{parentDir}' is not a directory? Move or rename whatever " +
                                         "resides here, create the parent location as a directory, then try again.")
        
        self._fileName = value
    
    @property
    def shape(self):
        if self.data is None:
            return None
        return self.data.shape
    
    def add(self, array):
        '''
        Parameters:
            array -- a numpy array
        '''
        with NpyAppendArray(self.fileName, delete_if_exists=False) as npaa:
            npaa.append(array)
    
    def load(self):
        try:
            if os.path.isfile(self.fileName):
                self.data = np.load(self.fileName, mmap_mode="r")
        except EOFError:
            raise EOFError(f"Could not load '{self.fileName}'; the file may be empty and needs to be deleted")
