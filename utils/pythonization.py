from ROOT import pythonization
from utils import raiseError

############## PYTHONIZATION ##############
def _parMgr_returnParPtr_if_keyExists(self,key):
    if self.findParameter(key): return self.findParameter(key) # findParameter returns ComplexParameter*
    if self.findAmpParameter(key): return self.findAmpParameter(key) # findAmpParameter returns MinuitParameter*
    raiseError(KeyError, f'Production (or Amplitude) Parameter {key} does not exist in ParameterManager')

def _parMgr_setValue(self, key, value):
    if self.findParameter(key): # IF Production Parameter
        parameter = self.findParameter(key) # ComplexParameter*
        if parameter.isFixed():
            raiseError(ValueError, f'Invalid request to set a new value for a fixed production parameter {key}')
        parameter.setValue(value) # if value is complex and parameter is real, imaginary part is ignored
        return
    if self.findAmpParameter(key): # IF Amplitude Parameter
        parameter = self.findAmpParameter(key) # MinuitParameter*
        if not parameter.floating():
            raiseError(ValueError, f'Invalid request to set a new value for a non-floating amplitude parameter {key}')
        parameter.setValue(value, False) # do not notify here as we wish to skip updateParCovariance()
        self.update(parameter, True) # manually update and set skipCovarianceUpdate=True
        return
    raiseError(KeyError, f'Production (or Amplitude) Parameter {key} does not exist in ParameterManager')

# Pythonize requires v6.26 or later
@pythonization("ParameterManager")
def pythonize_parMgr(klass):
    klass.__repr__ = lambda self: '\n'.join([f'{k}: {_parMgr_returnParPtr_if_keyExists(self,k).value()}' for k in self.getParametersList()])
    klass.__len__  = lambda self: self.getParametersList().size()
    klass.__getitem__ = lambda self, key: _parMgr_returnParPtr_if_keyExists(self,key).value()
    klass.__setitem__ = lambda self, key, value: _parMgr_setValue(self,key,value)
    klass.__contains__ = lambda self, key: key in self.getParametersList()
