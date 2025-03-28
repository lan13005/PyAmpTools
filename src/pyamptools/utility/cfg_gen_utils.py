import random
from pyamptools import atiSetup
from pyamptools.utility.general import vps_amp_name, zlm_amp_name

help_header = """#####################################
####	THIS IS A CONFIG FILE	 ####
#####################################
##
##  Blank lines or lines beginning with a "#" are ignored.
##
##  Double colons (::) are treated like a space.
##     This is sometimes useful for grouping (for example,
##     grouping strings like "reaction::sum::amplitudeName")
##
##  All non-comment lines must begin with one of the following keywords.
##
##  (note:  <word> means necessary
##	    (word) means optional)
##
##  include	  <file>
##  define	  <word> (defn1) (defn2) (defn3) ...
##  fit 	  <fitname>
##  keyword	  <keyword> <min arguments> <max arguments>
##  reaction	  <reaction> <particle1> <particle2> (particle3) ...
##  data	  <reaction> <class> (arg1) (arg2) (arg3) ...
##  genmc	  <reaction> <class> (arg1) (arg2) (arg3) ...
##  accmc	  <reaction> <class> (arg1) (arg2) (arg3) ...
##  normintfile   <reaction> <file>
##  sum 	  <reaction> <sum> (sum2) (sum3) ...
##  amplitude	  <reaction> <sum> <amp> <class> (arg1) (arg2) ([par]) ...
##  initialize    <reaction> <sum> <amp> <"events"/"polar"/"cartesian">
##		    <value1> <value2> ("fixed"/"real")
##  scale	  <reaction> <sum> <amp> <value or [parameter]>
##  constrain	  <reaction1> <sum1> <amp1> <reaction2> <sum2> <amp2> ...
##  permute	  <reaction> <sum> <amp> <index1> <index2> ...
##  parameter	  <par> <value> ("fixed"/"bounded"/"gaussian")
##		    (lower/central) (upper/error)
##    DEPRECATED:
##  datafile	  <reaction> <file> (file2) (file3) ...
##  genmcfile	  <reaction> <file> (file2) (file3) ...
##  accmcfile	  <reaction> <file> (file2) (file3) ...
##
#####################################\n\n
"""

amptools_zlm_ampName = "Zlm"
amptools_vps_ampName = "Vec_ps_refl"

def generate_amptools_cfg(
    quantum_numbers,
    polAngles,
    polMags,
    polFixedScales,
    datas,
    gens,
    accs,
    bkgnds,
    realAmps,
    fixedAmps,
    fitName,
    cfgFileOutputName,
    basereactName,
    particles,
    header=help_header,
    initialization=None,
    datareader="ROOTDataReader",
    add_amp_factor='',
    append_to_cfg='',
    append_to_decay='',
    exclude_sums_zeroed_by_polmag_value=True,
):
    """
    Generate an AmpTools configuration file for a Zlm fit

    Args:
        quantum_numbers (list): List of lists of quantum numbers. For Zlm then [Reflectivity, spin, spin-projection]
        polAngles (list): List of polarization polAngles in degrees
        polMags (list): List of polarization magnitudes
        polFixedScales (list): List of polarization fixed scales
        datas (list): List of data files
        gens (list): List of gen files
        accs (list): List of acc files
        bkgnds (list): List of bkgnd files
        realAmps (list): List of amplitude names that are real
        fixedAmps (list): List of amplitude names that are fixed
        fitName (str): A FitResults (.fit) file will be created with this name prefixed
        cfgFileOutputName (str): Name of output configuration file
        basereactName (str): Base name of reaction
        particles (list): List of particles in reaction
        header (str): Header to append to the top of the file
        datareader (str): Data reader to use
        add_amp_factor (str): Additional factor to add to the amplitude
        append_to_cfg (str): Additional string args to append to the end of the cfg file
        append_to_decay (str): Additional string args to append to the decay factor
        initialization (complex, None): All amplitudes will be set to this value if initialization is a complex number
        exclude_sums_zeroed_by_polmag_value (bool): If True, sums that are zeroed with polMag=1 will be excluded

    Returns:
        None, writes a file to cfgFileOutputName
    """

    ############## SET ENVIRONMENT VARIABLES ##############
    atiSetup.setup(globals())
    #######################################################

    cfgInfo = ConfigurationInfo(fitName)

    # Dict[str, List[cppyy.gbl.AmplitudeInfo]] where AmplitudeInfo is an AmpTools object
    # Example: constraintMap: {'Sp0+': [<cppyy.gbl.AmplitudeInfo object at 0xcebb720>, <cppyy.gbl.AmplitudeInfo object at 0xcf3a1a0>]}
    #   This means that there are 2 amplitudes for the Sp0+ partial wave
    #   The first amplitude is the default one, and the second is the one that will be used for the constraint
    constraintMap = {}
    
    # The intensity formula for the Zlm / vec_ps_refl amplitudes consists of 4 independent coherent sums
    #    Each sum is designated by the sign in the prefactor (1 +/- P_gamma)
    #    and the (real/imag) part taken on the Zlm / vec_ps_refl amplitude
    #    See GlueX Document 4094-v3 for more details
    #    TIP: Positive reflectivity amplitude consists of two terms: RealPos and ImagNeg
    refl_sum_numpair_map = {
         1: (["RealPos", "ImagNeg"], ["+1 +1", "-1 -1"]),
        -1: (["RealNeg", "ImagPos"], ["+1 -1", "-1 +1"])
    }

    for i in range(len(polAngles)):
        ####################################
        #### Create ReactionInfo object ####
        ####################################

        polAngle = polAngles[i]
        polMag = polMags[i]

        reactName = f"{basereactName}_{polAngle:0>3}"
        scaleName = f"parScale{polAngle:0>3}"
        parScale = cfgInfo.createParameter(scaleName, 1.0)
        if polFixedScales[i]:
            parScale.setFixed(True)

        reactionInfo = cfgInfo.createReaction(reactName, particles)  # ReactionInfo*
        # reactionInfo.setNormIntFile(f"{reactName}.ni", False)

        ###################################
        #### SET DATA, GEN, ACC, BKGND ####
        ###################################

        required_srcs = set(["data", "genmc", "accmc"]) # amptools keywords
        if len(bkgnds) > 0: 
            required_srcs.add("bkgnd") # amptools keyword
        reader_names = {}
        reader_argss = {}
        if isinstance(datareader, str):
            reader_parts = datareader.split(" ")
            reader_name = reader_parts[0]
            reader_args = reader_parts[1:] if len(reader_parts) > 1 else []
            for required_src in required_srcs:
                reader_names[required_src] = reader_name
                reader_argss[required_src] = reader_args
        if isinstance(datareader, dict):
            assert set(datareader.keys()) == required_srcs, f"Datareader keys must be {required_srcs} instead of {set(datareader.keys())}"
            for required_src, reader_parts in datareader.items():
                reader_parts = reader_parts.split(" ")
                reader_name = reader_parts[0]
                reader_args = reader_parts[1:] if len(reader_parts) > 1 else []
                reader_names[required_src] = reader_name
                reader_argss[required_src] = reader_args

        # A dataset is not always needed, i.e. if you are generating simulations
        # If you are doing a fit, you must have at least one (data, gens, accs) bkgnds is optional
        if len(datas) > 0:
            data = datas[i]
            data_args = [data] + reader_argss["data"]  # append args
            reactionInfo.setData(reader_names["data"], data_args.copy())
        if len(gens) > 0:
            gen = gens[i]
            gen_args = [gen] + reader_argss["genmc"]
            reactionInfo.setGenMC(reader_names["genmc"], gen_args.copy())
        if len(accs) > 0:
            acc = accs[i]
            acc_args = [acc] + reader_argss["accmc"]
            reactionInfo.setAccMC(reader_names["accmc"], acc_args.copy())
        if len(bkgnds) > 0:
            bkgnd = bkgnds[i]
            bkgnd_args = [bkgnd] + reader_argss["bkgnd"]
            reactionInfo.setBkgnd(reader_names["bkgnd"], bkgnd_args.copy())

        #############################################
        #### DEFINE COHERENT SUMS AND AMPLITUDES ####
        #############################################

        for quantum_number in quantum_numbers:
            ref, L, M = quantum_number[:3]
            
            J = quantum_number[3] if len(quantum_number) > 3 else None
            assume_zlm = J is None # else assume vec_ps_refl
            
            sumNames, numPairs = refl_sum_numpair_map[ref]
            for sumName, numPair in zip(sumNames, numPairs):
                
                print(f"sumName: {sumName}, numPair: {numPair}")
                
                # TODO: Actually another half of the terms disappear if polMag=0 so at some point we need to implement this
                if exclude_sums_zeroed_by_polmag_value and float(polMag) == 1:  # Terms with (1-polMag) prefactor disappear if polMag=1
                    if sumName in ["RealNeg", "ImagNeg"]:
                        continue

                cfgInfo.createCoherentSum(reactName, sumName)  # returns CoherentSumInfo*

                ampName = zlm_amp_name(ref, L, M) if assume_zlm else vps_amp_name(ref, J, M, L)
                ampInfo = cfgInfo.createAmplitude(reactName, sumName, ampName)  # AmplitudeInfo*

                num1, num2 = numPair.split()
                if assume_zlm:
                    angularFactor = [f"{amptools_zlm_ampName}", f"{L}", f"{M}", f"{num1}", f"{num2}", f"{polAngle}", f"{polMag}"]
                else:
                    # Vec_ps_refl 1 -1 0 -1  -1  LOOPPOLANG LOOPPOLVAL omega3pi
                    angularFactor = [f"{amptools_vps_ampName}", f"{J}", f"{M}", f"{L}", f"{num1}", f"{num2}", f"{polAngle}", f"{polMag}"]
                if append_to_decay:
                    angularFactor.extend(append_to_decay.split())
                ampInfo.addFactor(angularFactor)
                if add_amp_factor:
                    ampInfo.addFactor(add_amp_factor.split(" "))
                ampInfo.setScale(f"[{scaleName}]")

                if ampName not in constraintMap:
                    constraintMap[ampName] = [ampInfo]
                else:
                    constraintMap[ampName].append(ampInfo)

    #####################################
    ### RANDOMLY INITALIZE AMPLITUDES ###
    #####################################

    bAoV = False # boolean all one value
    if isinstance(initialization, complex):
        bAoV = True
    
    for amp, lines in constraintMap.items(): # see above for structure of constraintMap

        value = initialization if bAoV else random.uniform(0.0, 1.0) + 1j * random.uniform(0.0, 1.0)
        if amp in realAmps:
            lines[0].setReal(True)
            value = complex(np.abs(value), 0.0) # rotate to real
        if amp in fixedAmps:
            lines[0].setFixed(True)
        for line in lines[1:]:
            lines[0].addConstraint(line)
            
        # overwrite value if key exists in initialization dict
        #   this still comes after the above since we use it setReal/setFixed/addConstraint
        if isinstance(initialization, dict):
            if amp in initialization:
                value = complex(initialization[amp])

        lines[0].setValue(value)

    #########################
    ### WRITE CONFIG FILE ###
    #########################

    cfgInfo.display()
    cfgInfo.write(cfgFileOutputName)

    with open(cfgFileOutputName, "r") as original:
        data = original.read()

    # prepend help_header to top of the file
    data = header + data
    with open(cfgFileOutputName, "w") as modified:
        modified.write(data)
    if append_to_cfg:
        data = data + append_to_cfg
        with open(cfgFileOutputName, "w") as modified:
            modified.write(data)

    return data
