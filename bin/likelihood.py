
import logging as log
import os
import pickle as pkl
from functools import partial

import jax
import jax.numpy as jnp

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
import time

import numpy as np
from omegaconf import OmegaConf

log.basicConfig(level=log.INFO, format='%(asctime)s| %(message)s', datefmt='%H:%M:%S')

# Mark wi, numTerms as static so jax does not trace them
# @partial(jax.pmap) # , static_broadcasted_argnums=(1, 2, 3))
# @partial(jax.jit) # , static_argnums=(0, 1, 2))

############# THINGS TO DO #############
# Code to always over reaction so likelihood(mass,t,cV)
# (mass, t, reaction, terms, events) ampvecs

#### normint.pkl ~ (reaction * nmbMasses * nmbTprimes, terms, terms)
# have to rerun to cluster reactions together so we can accept the same cV for all reactions in kinematic bin

#### amps.pkl   ~ (reaction * nmbMasses * nmbTprimes, terms, events)
# We have the same number of reactions and terms per kinematic bin
# number of events can differ but they are all concatenated together
# have to ensure number of len(starts) == reaction * nmbMasses * nmbTprimes. Perhaps off by 1 since we prepend 0
# nreactions

def intensity_calc(array, wi, nTerms):

    """
    Design:
    1) create a bunch of jitted intensity calculators for each kinematic bin
        - cons: introduces lots of compilation overhead
    2) histogram number of events across all data sources. Find several groupings that minimize number of jit recompiles 
        and minimizes the total number of zero paddings. This can be effective if compiling ends up to take a long time
    3) create a single jitted intensity calculator that stores the entire dataset and
        will take start and stop indicies. This is fine if we return a single value like the 2LnLik.
        This will not allow us to output intensities per event since the output shape will be different
    """
    
    @jax.jit
    def _intensity_calc(cTmp):

        intensities = jnp.zeros(array.shape[-1])

        cVRe = cTmp[0::2]
        cVIm = cTmp[1::2]

        for i in range(nTerms):
            for j in range(i + 1):

                # (a + ib) * (c - id) = ac + bd + i(bc - ad)
                cViVjRe = cVRe[i] * cVRe[j] + cVIm[i] * cVIm[j]
                cViVjIm = cVIm[i] * cVRe[j] - cVRe[i] * cVIm[j]

                if i != j:
                    _cViVjRe = 2 * cViVjRe
                    _cViVjIm = 2 * cViVjIm
                else:
                    _cViVjRe = cViVjRe
                    _cViVjIm = cViVjIm

                cAiAjRe =  array[2*i] * array[2*j] + array[2*i+1] * array[2*j+1]
                cAiAjIm = -array[2*i] * array[2*j+1] + array[2*i+1] * array[2*j]

                intensities += (_cViVjRe * cAiAjRe - _cViVjIm * cAiAjIm)

        intensities = intensities * array[wi] / jnp.sum(array[wi], axis=-1)

        return intensities

    return _intensity_calc

def normint_calc(nTerms):

    """
    Similar pattern to intensity_calc but since our static arrays (normalization integrals)
    are of fixed shape and we can directly pass them in as arguments
    Jax will not need to retrace the function i think. Only one jit compiled function will be needed
    """

    @jax.jit
    def _normInt_calc(cTmp, ampInts, normInts):

        normInt = 0

        cVRe = cTmp[0::2]
        cVIm = cTmp[1::2]

        for i in range(nTerms):
            for j in range(i + 1):

                # (a + ib) * (c - id) = ac + bd + i(bc - ad)
                cViVjRe = cVRe[i] * cVRe[j] + cVIm[i] * cVIm[j]
                cViVjIm = cVIm[i] * cVRe[j] - cVRe[i] * cVIm[j]

                if i != j:
                    _cViVjRe = 2 * cViVjRe
                    _cViVjIm = 2 * cViVjIm
                else:
                    _cViVjRe = cViVjRe
                    _cViVjIm = cViVjIm

                reNI = normInts[i, j].real
                imNI = normInts[i, j].imag

                normInt += (_cViVjRe * reNI - _cViVjIm * imNI) 

        return normInt

    return _normInt_calc

# Should already be binned
class Likelihood:

    def __init__(self, yaml_file):

        yaml_file = OmegaConf.load(yaml_file)
        self.base_directory = yaml_file["base_directory"]
        self.output_directory = yaml_file["amptools"]["output_directory"]
        self.nmbMasses = yaml_file["n_mass_bins"]
        self.nmbTprimes = yaml_file["n_t_bins"]
        self.reactions = yaml_file["polarizations"]
        self.nmbReactions = len(self.reactions)
        self.yaml_file = yaml_file

        with open(f"{self.base_directory}/amps.pkl", "rb") as f:
            amp_src = pkl.load(f)
        with open(f"{self.base_directory}/normint.pkl", "rb") as f:
            normint_src = pkl.load(f)
        self.amp_src = amp_src
        self.normint_src = normint_src

        self.starts = amp_src["starts"]
        self.stops = amp_src["stops"]
        self.keys = amp_src["termNames"] # ~ (2*nmbAmps + 2) real and imaginary parts of complex terms + weight + bin
        self.wi = self.keys.index("weight") # wi, bi, nTerms should never change. if using in jax, no need to re-jit
        self.bi = self.keys.index("bin")
        self.nTerms = len(self.keys) // 2 - 1 

        self.data = amp_src["data"]
        self.accmc = amp_src["accmc"]
        self.genmc = amp_src["genmc"]
        self.m_sumDataWeights = jnp.sum(self.data[self.wi])
        self.m_sumAccWeights = jnp.sum(self.accmc[self.wi])
        self.m_sumGenWeights = jnp.sum(self.genmc[self.wi])
        self._check_data_shape()

        self.bkgnd = amp_src.get("bkgnd") # get is load key safe
        self.bkgndexist = self.bkgnd is not None
        self.m_sumBkgWeights = jnp.sum(self.bkgnd[self.wi]) if self.bkgndexist else 0

        self.reactions = normint_src["reactions"] # ~ (nmbMasses, nmbTprimes) List of strings containing reaction names
        self.nGens = normint_src["nGens"] # ~ (nmbMasses, nmbTprimes) # Number of genmc events
        self.nAccs = normint_src["nAccs"] # ~ (nmbMasses, nmbTprimes) Number of accmc weighted events
        self.nTerms = normint_src["nTerms"] # ~ (nmbReactions) length of termss
        self.terms = normint_src["terms"] # ~ (nmbAmps) List of strings containing term names (i.e. the Zlm partial waves amplitudes)
        self.ampIntss = normint_src["ampIntss"] # ~ (nmbMasses, nmbTprimes, nmbAmps, nmbAmps) Amplitude integrals
        self.normIntss = normint_src["normIntss"] # ~ (nmbMasses, nmbTprimes, nmbAmps, nmbAmps) Normalized integrals
        self.normint_files = normint_src["normint_files"] # ~ (nmbMasses, nmbTprimes). List of strings containing normint file names

        # nifty production coefficient
        self.cV = jnp.full( (self.nmbMasses, self.nmbTprimes, self.nTerms*2), 1 ) # real/imag parts interleaved as all 1s

        ##########################################################

        self.fDIntens, self.fBIntens = [], []
        self.fDStarts, self.fBStarts = [], []
        self.fDStops,  self.fBStops  = [], []
        Dbins = []
        for tbin in range(self.nmbTprimes):
            for mbin in range(self.nmbMasses):
                for rbin in range(len(self.reactions)):
                    Dbins.append((rbin, mbin, tbin))
                    start = self.starts["data"][rbin][mbin][tbin]
                    stop = self.stops["data"][rbin][mbin][tbin]
                    subArray = self.data[:, start:stop]
                    self.fDStarts.append(start)
                    self.fDStops.append(stop)
                    self.fDIntens.append(intensity_calc(subArray, self.wi, self.nTerms)) 
                    start = self.starts["bkgnd"][rbin][mbin][tbin]
                    stop = self.stops["bkgnd"][rbin][mbin][tbin]
                    subArray = self.bkgnd[:, start:stop]
                    self.fBStarts.append(start)
                    self.fBStops.append(stop)
                    self.fBIntens.append(intensity_calc(subArray, self.wi, self.nTerms))
        self.fNormInt = normint_calc(self.nTerms)

        # print(self.fDStarts)
        # print(self.fDStops)
        # print(self.data.shape)
        # print(Dbins)
        # exit()
    
    def likelihood(self, params):

        (mbin, tbin), cV = params[0], params[1]
        k = tbin * self.nmbMasses + mbin

        ############################ DATA TERM ############################
        # NOTE: We divide out the weight we multiplied by in the intensity function similar to AmpTools likelihood
        # Skip this step if we end up with no use for intensity function

        sumLnI_data = 0
        sumLnI_bkgnd = 0

        fDIntens = self.fDIntens[k*self.nmbReactions:(k+1)*self.nmbReactions]
        fDStarts = self.fDStarts[k*self.nmbReactions:(k+1)*self.nmbReactions]
        fDStops = self.fDStops[k*self.nmbReactions:(k+1)*self.nmbReactions]

        for start, stop, fIntens in zip(fDStarts, fDStops, fDIntens):
            thisTerm = jnp.sum ( self.data[self.wi, start:stop] * jnp.log(fIntens(cV) / self.data[self.wi, start:stop]) )
            sumLnI_data += thisTerm
            log.info(f"sumLnI_data += {thisTerm}")
        
        if self.bkgndexist:
            fBIntens = self.fBIntens[k*self.nmbReactions:(k+1)*self.nmbReactions]
            fBStarts = self.fBStarts[k*self.nmbReactions:(k+1)*self.nmbReactions]
            fBStops = self.fBStops[k*self.nmbReactions:(k+1)*self.nmbReactions]
            for start, stop, fIntens in zip(fBStarts, fBStops, fBIntens):
                thisTerm = jnp.sum ( self.bkgnd[self.wi, start:stop] * jnp.log(fIntens(cV) / self.bkgnd[self.wi, start:stop]) )
                sumLnI_bkgnd += thisTerm
                log.info(f"sumLnI_bkgnd += {thisTerm}")

        dataTerm = sumLnI_data - sumLnI_bkgnd
        log.info(f"dataTerm: {dataTerm}")
        ########################## END DATA TERM ###########################

        ########################## NORM INT TERM ###########################
        normTerm = 0
        for rbin in range(self.nmbReactions):
            thisTerm = self.fNormInt(cV, self.ampIntss[rbin, mbin, tbin], self.normIntss[rbin, mbin, tbin])
            if self.bkgndexist:
                nPred = thisTerm
                thisTerm = (self.m_sumDataWeights - self.m_sumBkgWeights) * jnp.log(thisTerm)
                thisTerm += nPred - (self.m_sumDataWeights * jnp.log(nPred))
            normTerm += thisTerm
            log.info(f"normTerm += {thisTerm}")
        log.info(f"normTerm: {normTerm}")
        ######################## END NORM INT TERM #########################

        nll = -2 * (dataTerm - normTerm)
        return nll

    def nBar(self):
        # return nBar, nBarAccCorr
        pass

    def _check_data_shape(self):
        log.info("------ Checking data shapes ------")
        log.info(f"nTerms: {self.nTerms}")
        log.info(f"nmbReactions: {len(self.reactions)}")
        log.info(f"nmbMasses: {self.nmbMasses}")
        log.info(f"nmbTprimes: {self.nmbTprimes}")
        log.info('- - - - - - - - - - - - - - - - - - ')
        for k, v in self.amp_src.items():
            if isinstance(v, np.ndarray):
                log.info(f"Shape of {k}: {v.shape} ~ (2nTerms+[weight, bin], nEvents)")
        for k, v in self.normint_src.items():
            if isinstance(v, np.ndarray):
                if len(v.shape) == 3:
                    log.info(f"Shape of {k}: {v.shape} ~ (nmbReactions, nmbMasses, nmbTprimes)")
                elif len(v.shape) == 5:
                    log.info(f"Shape of {k}: {v.shape} ~ (nmbReactions, nmbMasses, nmbTprimes, nTerms, nTerms)")
        log.info("---- Done checking data shapes ----\n")
        

# yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/tests_ift/pa.yaml"
yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0/minor_changes_primary.yaml"

likMan = Likelihood(yaml_file)

params = []
for tbin in range(likMan.nmbTprimes):
    for mbin in range(likMan.nmbMasses):
        params.append( ((mbin, tbin), jnp.full(2*likMan.nTerms, 1)) )

intensity = likMan.likelihood(params[0])
log.info(intensity)
# 
# log.info(likelihood.data[-1])

