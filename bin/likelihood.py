
import logging
import os
import pickle as pkl
import time

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

# ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
level = logging.INFO
logger = logging.getLogger('pwa_manager')
logger.setLevel(level)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s| %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(handler)

jax.config.update("jax_enable_x64", True)

# Mark wi, numTerms as static so jax does not trace them
# @partial(jax.pmap) # , static_broadcasted_argnums=(1, 2, 3))
# @partial(jax.jit) # , static_argnums=(0, 1, 2))

def generate_pairs(m_sumCoherently):
    """ coherence matrix is symmetric """
    if len(m_sumCoherently.shape) != 2:
        raise ValueError("Coherence matrix is not 2D")
    if m_sumCoherently.shape[0] != m_sumCoherently.shape[1]:
        raise ValueError("Coherence matrix is not square")
    nTerms = m_sumCoherently.shape[0]
    triplet = []
    for i in range(nTerms):
        for j in range(i + 1):
            if m_sumCoherently[i][j] == 1:
                sym_factor = 2 if i != j else 1
                triplet.append((i, j, sym_factor))
    return jnp.array(triplet)

def intensity_calc(array, wi, nTerms, coh_pairs):

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

    array = jnp.array(array) # For some reason converting it here is faster?
    m_iNTrueEvents = array.shape[-1]
    
    # NOTE: jax.lax.scan makes looping better. Can parallelize better and stores intermediate values
    # Jax unrolls loops adding them to the computational graph. Inherently sequential
    # For small number of loopings, scanning can have more overhead
    @jax.jit
    def __intensity_calc(args, triple):

        i, j, sym_factor = triple
        cVRe, cVIm = args

        # V * conj(V) = (a + ib) * (c - id) = ac + bd + i(bc - ad)
        cViVjRe = cVRe[i] * cVRe[j] + cVIm[i] * cVIm[j]
        cViVjIm = cVIm[i] * cVRe[j] - cVRe[i] * cVIm[j]

        cAiAjRe =  array[2*i] * array[2*j]   + array[2*i+1] * array[2*j+1]
        cAiAjIm = -array[2*i] * array[2*j+1] + array[2*i+1] * array[2*j]

        # Dividing by m_iNTrueEvents
        # See AmpTools source code for more details. Not necessary but shifts minimum
        intensity = (cViVjRe * cAiAjRe - cViVjIm * cAiAjIm) * sym_factor / m_iNTrueEvents

        return args, intensity

    @jax.jit
    def _intensity_calc(cTmp):

        cVRe = cTmp[0::2]
        cVIm = cTmp[1::2]
        args = (cVRe, cVIm)
        _, intensities = jax.lax.scan(__intensity_calc, args, coh_pairs)

        intensities = jnp.sum(intensities, axis=0) * array[wi]

        return intensities

    return _intensity_calc

def normint_calc(nTerms, coh_pairs):

    """
    Similar pattern to intensity_calc but since our static arrays (normalization integrals)
    are of fixed shape and we can directly pass them in as arguments
    Jax will not need to retrace the function i think. Only one jit compiled function will be needed
    """

    @jax.jit
    def __normInt_calc(carry, triple):

        i, j, sym_factor = triple
        cVRe, cVIm, ampInts, normInts = carry

        # V * conj(V) = (a + ib) * (c - id) = ac + bd + i(bc - ad)
        cViVjRe = cVRe[i] * cVRe[j] + cVIm[i] * cVIm[j]
        cViVjIm = cVIm[i] * cVRe[j] - cVRe[i] * cVIm[j]

        cViVjRe = sym_factor * cViVjRe
        cViVjIm = sym_factor * cViVjIm

        reNI = normInts[i, j].real
        imNI = normInts[i, j].imag

        thisTerm = sym_factor * (cViVjRe * reNI - cViVjIm * imNI) 
        
        return carry, thisTerm

    @jax.jit
    def _normInt_calc(cTmp, ampInts, normInts):

        cVRe = cTmp[0::2]
        cVIm = cTmp[1::2]

        carry = (cVRe, cVIm, ampInts, normInts)

        _, terms = jax.lax.scan(__normInt_calc, carry, coh_pairs)

        return jnp.sum(terms)

    return _normInt_calc

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
        self.keys = amp_src["partNames"] # ~ (2*nmbAmps + 2) real and imaginary parts of complex terms + weight + bin
        self.wi = self.keys.index("weight") # wi, bi, nTerms should never change. if using in jax, no need to re-jit
        self.bi = self.keys.index("bin")
        self.nTerms = len(self.keys) // 2 - 1 

        self.data = amp_src["data"]
        self.accmc = amp_src["accmc"]
        self.genmc = amp_src["genmc"]
        self._check_data_shape()

        self.bkgnd = amp_src.get("bkgnd") # get is load key safe
        self.bkgndexist = self.bkgnd is not None

        self.reactions = normint_src["reactions"] # ~ (nmbMasses, nmbTprimes) List of strings containing reaction names
        self.nGens = normint_src["nGens"] # ~ (nmbMasses, nmbTprimes) # Number of genmc events
        self.nAccs = normint_src["nAccs"] # ~ (nmbMasses, nmbTprimes) Number of accmc weighted events
        self.nTerms = normint_src["nTerms"] # ~ (nmbReactions) length of termss
        self.normint_terms = normint_src["terms"] # ~ (nmbAmps) List of strings containing term names (i.e. the Zlm partial waves amplitudes)
        self.m_sumCoherently = jnp.array(normint_src["m_sumCoherently"]) # ~ (nmbMasses, nmbTprimes, nTerms, nTerms) Coherent sum of terms
        self.coh_pairs = generate_pairs(self.m_sumCoherently) # ~ (nmbPairs, 2) Pairs of coherent terms
        self.ampIntss = normint_src["ampIntss"] # ~ (nmbReactions, nmbMasses, nmbTprimes, nmbAmps, nmbAmps) Amplitude integrals
        self.normIntss = normint_src["normIntss"] # ~ (nmbReactions, nmbMasses, nmbTprimes, nmbAmps, nmbAmps) Normalized integrals
        self.normint_files = normint_src["normint_files"] # ~ (nmbMasses, nmbTprimes). List of strings containing normint file names

        # Ensure the ording of the terms are the same
        self.ampvec_terms = [v[:-3] for v in self.keys if v[-3:] in ["_re", "_im"]]
        self.ampvec_terms = list(dict.fromkeys(self.ampvec_terms)) # remove duplicates
        self.normint_terms = [".".join(v.split(".")[1:]) for v in self.normint_terms]
        if self.ampvec_terms != self.normint_terms:
            log.error("Ampvec terms and normint terms do not match!")
            exit(1)
        self.terms = self.normint_terms

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
                    self.fDIntens.append(intensity_calc(subArray, self.wi, self.nTerms, self.coh_pairs))
                    if self.bkgndexist:
                        start = self.starts["bkgnd"][rbin][mbin][tbin]
                        stop = self.stops["bkgnd"][rbin][mbin][tbin]
                        subArray = self.bkgnd[:, start:stop]
                        self.fBStarts.append(start)
                        self.fBStops.append(stop)
                        self.fBIntens.append(intensity_calc(subArray, self.wi, self.nTerms, self.coh_pairs))
        self.fNormInt = normint_calc(self.nTerms, self.coh_pairs)

        self.fDStarts = jnp.array(self.fDStarts)
        self.fDStops = jnp.array(self.fDStops)
        self.fBStarts = jnp.array(self.fBStarts)
        self.fBStops = jnp.array(self.fBStops)
    
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
            intens = fIntens(cV)
            thisTerm = jnp.sum ( self.data[self.wi, start:stop] * jnp.log(intens / self.data[self.wi, start:stop]) )
            sumLnI_data += thisTerm
            logger.debug(f"sumLnI_data += {thisTerm} at {start}:{stop}")
        
        if self.bkgndexist:
            fBIntens = self.fBIntens[k*self.nmbReactions:(k+1)*self.nmbReactions]
            fBStarts = self.fBStarts[k*self.nmbReactions:(k+1)*self.nmbReactions]
            fBStops = self.fBStops[k*self.nmbReactions:(k+1)*self.nmbReactions]
            for start, stop, fIntens in zip(fBStarts, fBStops, fBIntens):
                thisTerm = jnp.sum ( self.bkgnd[self.wi, start:stop] * jnp.log(fIntens(cV) / self.bkgnd[self.wi, start:stop]) )
                sumLnI_bkgnd += thisTerm
                logger.debug(f"sumLnI_bkgnd += {thisTerm} at {start}:{stop}")

        dataTerm = sumLnI_data - sumLnI_bkgnd
        logger.debug(f"dataTerm: {dataTerm}")
        ########################## END DATA TERM ###########################

        ########################## NORM INT TERM ###########################
        normTerm = 0
        bStarts = self.fBStarts[k*self.nmbReactions:(k+1)*self.nmbReactions]
        bStops = self.fBStops[k*self.nmbReactions:(k+1)*self.nmbReactions]
        dStarts = self.fDStarts[k*self.nmbReactions:(k+1)*self.nmbReactions]
        dStops = self.fDStops[k*self.nmbReactions:(k+1)*self.nmbReactions]
        for rbin, bStart, bStop, dStart, dStop in zip(range(self.nmbReactions), bStarts, bStops, dStarts, dStops):
            thisTerm = self.fNormInt(cV, self.ampIntss[rbin, mbin, tbin], self.normIntss[rbin, mbin, tbin])
            if self.bkgndexist:
                m_sumBkgWeights = jnp.sum(self.bkgnd[self.wi, bStart:bStop])
                m_sumDataWeights = jnp.sum(self.data[self.wi, dStart:dStop])
                nPred = thisTerm + m_sumBkgWeights
                thisTerm = (m_sumDataWeights - m_sumBkgWeights) * jnp.log(thisTerm)
                thisTerm += nPred - (m_sumDataWeights * jnp.log(nPred))
            normTerm += thisTerm
            logger.debug(f"normTerm += {thisTerm} at bkgnd[{bStart}:{bStop}] and data[{dStart}:{dStop}]")
        logger.debug(f"normTerm: {normTerm}")
        ######################## END NORM INT TERM #########################

        nll = -2 * (dataTerm - normTerm)
        return nll

    def nBar(self):
        # return nBar, nBarAccCorr
        pass

    def _check_data_shape(self):
        logger.info("------ Checking data shapes ------")
        logger.info(f"nTerms: {self.nTerms}")
        logger.info(f"nmbReactions: {len(self.reactions)}")
        logger.info(f"nmbMasses: {self.nmbMasses}")
        logger.info(f"nmbTprimes: {self.nmbTprimes}")
        logger.info('- - - - - - - - - - - - - - - - - - ')
        for k, v in self.amp_src.items():
            if isinstance(v, np.ndarray):
                logger.info(f"Shape of {k}: {v.shape} ~ (2nTerms+[weight, bin], nEvents)")
        for k, v in self.normint_src.items():
            if isinstance(v, np.ndarray):
                if len(v.shape) == 3:
                    logger.info(f"Shape of {k}: {v.shape} ~ (nmbReactions, nmbMasses, nmbTprimes)")
                elif len(v.shape) == 5:
                    logger.info(f"Shape of {k}: {v.shape} ~ (nmbReactions, nmbMasses, nmbTprimes, nTerms, nTerms)")
        logger.info("---- Done checking data shapes ----\n")
        

yaml_file = "/w/halld-scshelf2101/lng/WORK/PyAmpTools9/OTHER_CHANNELS/ETAPI0/minor_changes_primary.yaml"

likMan = Likelihood(yaml_file)

params = []
for tbin in range(likMan.nmbTprimes):
    for mbin in range(likMan.nmbMasses):
        params.append( ((mbin, tbin), jnp.full(2*likMan.nTerms, 1)) )

start_time = time.time()
for i in range(6):
    nll = likMan.likelihood(params[i])
    # print(nll)
logger.info(f"Time taken: {time.time() - start_time}")

start_time = time.time()
for i in range(6):
    nll = likMan.likelihood(params[i])
    # print(nll)
logger.info(f"Time taken: {time.time() - start_time}")
start_time = time.time()
for i in range(6):
    nll = likMan.likelihood(params[i])
    print(nll)
logger.info(f"Time taken: {time.time() - start_time}")
# log.info(nll)