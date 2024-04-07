from ROOT import pythonization
from pyamptools.utility.general import raiseError


############## PYTHONIZATION ##############
def _parMgr_returnParPtr_if_keyExists(self, key):
    if key in self.getParMap():
        return self.getParMap()[key]  # getParMap() returns std::map<std::string, MinuitParameter*>
    raiseError(KeyError, f"Production (or Amplitude) Parameter {key} does not exist in ParameterManager")


def _parMgr_setValue(self, key, value):
    assert key in self.getParMap(), f"Production (or Amplitude) Parameter {key} does not exist in ParameterManager"
    assert not self.getDoCovarianceUpdate(), "doCovarianceUpdate must be set to False before setting a new value!"
    self.getParMap()[key].setValue(value)  # getParMap() returns std::map<std::string, MinuitParameter*>


# Pythonize requires v6.26 or later
@pythonization("ParameterManager")
def pythonize_parMgr(klass):
    klass.__repr__ = lambda self: "\n".join([f"{k}: {_parMgr_returnParPtr_if_keyExists(self,k).value()}" for k in self.getParMap()])
    klass.__len__ = lambda self: self.getParMap().size()
    klass.__getitem__ = lambda self, key: _parMgr_returnParPtr_if_keyExists(self, key).value()
    klass.__setitem__ = lambda self, key, value: _parMgr_setValue(self, key, value)
    klass.__contains__ = lambda self, key: key in self.getParMap()
