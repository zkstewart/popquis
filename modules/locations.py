# Copyright (C) 2026 Zachary Kenneth Stewart
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

class Locations:
    '''
    Args:
        workingDirectory -- a string indicating the location where popquis temporary and output
                            files are to be written
    Attributes:
        storageDir -- a string indicating the subdirectory where data storage files e.g., .npz and
                      .npy files, are to be written
        group1Npy -- a string giving the full location of the group1.npy file
        group2Npy -- as above but for group2
    Methods:
        init_dirs -- pipeline function which calls make_workdir() and make_subdir() in sequence to
                     set up a popquis analysis folder
        make_workdir -- creates the workingDirectory folder with appropriate validations
        make_subdir -- creates a subdirectory under workingDirectory based on an attribute stored
                       by this Location object
    '''
    OKAY_SUFFIX = ".ok"
    
    @staticmethod
    def touch(fileName):
        parentDir = os.path.dirname(os.path.abspath(fileName))
        if os.path.isdir(parentDir):
            open(fileName + Locations.OKAY_SUFFIX, "w").close()
    
    def __init__(self, workingDirectory, quiet=True):
        self.workingDirectory = workingDirectory
        self.init_dirs(quiet=quiet)
    
    @property
    def workingDirectory(self):
        return self._workingDirectory
    
    @workingDirectory.setter
    def workingDirectory(self, value):
        value = os.path.abspath(value)
        self._workingDirectory = value
    
    # Naive directory properties
    @property
    def storageDir(self):
        return os.path.join(self.workingDirectory, "data_store")
    
    @property
    def qcPlotsDir(self):
        return os.path.join(self.workingDirectory, "qc_plots")
    
    # Naive file properties
    @property
    def group1Npy(self):
        return os.path.join(self.storageDir, "group1.npy")
    
    @property
    def group2Npy(self):
        return os.path.join(self.storageDir, "group2.npy")
    
    @property
    def rawTSV(self):
        return os.path.join(self.workingDirectory, "raw_results.tsv")
    
    @property
    def thresholdsTSV(self):
        return os.path.join(self.workingDirectory, "thresholds.tsv")
    
    # Attributes with value input
    def outputPNG(self, value):
        return os.path.join(self.workingDirectory, f"stacked_barplot.{value}.png")
    
    def outputPDF(self, value):
        return os.path.join(self.workingDirectory, f"stacked_barplot.{value}.pdf")
    
    # Methods
    def init_dirs(self, quiet=True):
        self.make_workdir(quiet=quiet)
        self.make_subdir("storageDir")
        self.make_subdir("qcPlotsDir")
    
    def make_workdir(self, quiet=True):
        if os.path.isdir(self.workingDirectory):
            if not quiet:
                print(f"# -o location already exists; will attempt to resume a previous run")
        elif not os.path.exists(self.workingDirectory):
            parentDir = os.path.dirname(self.workingDirectory)
            if not os.path.isdir(parentDir):
                raise NotADirectoryError(f"Cannot create the -o '{self.workingDirectory}' directory as its parent " +
                                         f"location '{parentDir}' is not a directory or does not exist.")
            else:
                os.mkdir(self.workingDirectory)
                if not quiet:
                    print(f"# Created output directory '{self.workingDirectory}' as part of argument validation")
        else:
            raise NotADirectoryError(f"-o location already exists, but is not a directory. Try to " +
                                     "specify a different location instead")
    
    def make_subdir(self, attribute):
        if not hasattr(self, attribute):
            raise ValueError(f"Locations object does not recognise '{attribute}' as a location.")
        
        targetDir = getattr(self, attribute)
        targetBase = os.path.basename(targetDir)
        if os.path.exists(targetDir):
            if not os.path.isdir(targetDir):
                raise NotADirectoryError(f"Tried to create the '{targetBase}' folder within the " +
                                         f"output folder '{self.workingDirectory}'. However, I found that " +
                                         "it already exists but is not a folder? To resolve this issue, try " + 
                                         "specifying a different output folder when running popquis")
        else:
            assert os.path.isdir(self.workingDirectory), \
                "sanity check, this should not occur unless a folder was deleted mid-operation"
            os.mkdir(targetDir) # silent operation, no need to announce
