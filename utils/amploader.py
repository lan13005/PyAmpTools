import os

############################################################
# This file holds scripts to load in amptools information
############################################################

def load_amplitude_info(seedfile):
    ''' 
    Read in an amptools seed file

    Args:
        seedfile (str): path to seed file
    
    Returns:
        par_names: list[str]: list of amplitude names
        par_values: list[complex]: list of complex amplitude values
    '''
    par_names = []
    par_values = []
    with open(seedfile) as fit:
        for line in fit:
            line = line.split(" ")
            amp, real, imag = line[1], float(line[3].strip()), float(line[4].strip())
            complex_val = complex(real, imag)
            if complex_val not in par_values: # Ignore duplicates, these are likely "constrained"
                par_names.append(amp)
                par_values.append(complex_val)
    return par_names, par_values

def flatten_amplitude_parts(par_names, par_values):
    '''
    Flatten the complex amplitude values into a single array of real and imaginary parts.
    Ignores any amplitude values that are zero (i.e. if amplitude is real)

    Args:
        par_names: list[str]: list of amplitude names
        par_values: list[complex]: list of complex amplitude values

    Returns:
        par_array_parts: list[float]: list of real and imaginary parts of the complex amplitude values
        indicies: list[int]: list of indicies of the real and imaginary parts of the complex amplitude values
    '''
    par_array_parts = []
    par_name_parts = [] 
    indicies = []
    j=0
    for i in range(len(par_names)):
        real, imag =  par_values[i].real, par_values[i].imag
        if real !=0: par_array_parts.append(real); indicies.append(j); par_name_parts.append(par_names[i]+"_re")
        j+=1
        if imag !=0: par_array_parts.append(imag); indicies.append(j); par_name_parts.append(par_names[i]+"_im")
        j+=1
    return par_array_parts, par_name_parts, indicies

def collect_amplitude_parts(par_array_parts, par_names, indicies):
    '''
    Reconstruct the complex amplitude values from the flattened array of real and imaginary parts

    Args:
        par_array_parts: list[float]: list of real and imaginary parts of the complex amplitude values
        par_names: list[str]: list of amplitude names
        indicies: list[int]: list of indicies of the real and imaginary parts of the complex amplitude values

    Returns:
        par_values: list[complex]: list of complex amplitude values
    '''
    par_values = []
    for i in range(len(par_names)):
        real = par_array_parts[indicies.index(2*i)] if 2*i in indicies else 0
        imag = par_array_parts[indicies.index(2*i+1)] if 2*i+1 in indicies else 0
        par_values.append(complex(real, imag))
    return par_values