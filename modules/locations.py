import os

class Locations:
    OKAY_SUFFIX = ".ok"
    
    def __init__(self, workingDirectory):
        self.workingDirectory = workingDirectory
        self.init_dirs()
    
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
    
    # Naive file properties
    @property
    def parentsNpy(self):
        return os.path.join(self.storageDir, "parents.npy")
    
    @property
    def group1Npy(self):
        return os.path.join(self.storageDir, "group1.npy")
    
    @property
    def group2Npy(self):
        return os.path.join(self.storageDir, "group2.npy")
    
    # Methods
    def init_dirs(self):
        self.make_workdir()
        self.make_subdir("storageDir")
    
    def make_workdir(self):
        if os.path.isdir(self.workingDirectory):
            print(f"# -o location already exists; will attempt to resume a previous run")
        elif not os.path.exists(self.workingDirectory):
            parentDir = os.path.dirname(self.workingDirectory)
            if not os.path.isdir(parentDir):
                raise NotADirectoryError(f"Cannot create the -o '{self.workingDirectory}' directory as its parent " +
                                         f"location '{parentDir}' is not a directory or does not exist.")
            else:
                os.mkdir(self.workingDirectory)
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
