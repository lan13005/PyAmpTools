from utils import remove_all_whitespace

def define_if_not_observed(df, function, OBSERVED_FORMULAE, name, columns):
    '''
    If requested function has not observed, define it and return new name
    If requested function has been observed, return the name associated with the observed function

    Args:
        df: RDataFrame
        function: function string
        OBSERVED_FORMULAE: dict of observed {function: variable_name} pairs
        name: new variable name to define if not observed
        columns: list of column names in RDataFrame to check if we need to Redefine

    Returns:
        df: RDataFrame
        name: string
    '''
    if function in OBSERVED_FORMULAE: # check if function observed
        old_name = OBSERVED_FORMULAE[function]
        return df, old_name
    else:
        OBSERVED_FORMULAE[function] = name
        df = df.Redefine(name, function) if name in columns else df.Define(name, function) # check if name already exists
        return df, name

def book_histogram(df, HISTS_TO_BOOK, columns):
    '''
    Book histograms for a given type (data, bkgnd, accMC, genMC)

    Args:
        HISTS_TO_BOOK: dict of histograms to book, with associated function call, binning scheme, draw options
        columns: list of column names in RDataFrame to check if we need to Redefine

    Returns:
        BOOKED_HISTOGRAMS: list of booked Histos
        DRAW_OPTIONS: list of draw options
    '''
    ## ALL HISTOGRAMS FOR A SOURCE FILES
    BOOKED_HISTOGRAMS = []

    OBSERVED_FORMULAE = {} # Track observed functions matching to defined variable names: {function: variable_name}
    DRAW_OPTIONS      = []

    # Booking hists are lazy! Book all of them first, then run filling/Drawing
    for hname, booked_hist in HISTS_TO_BOOK.items():
        histIs1D = len(booked_hist) == 7
        if histIs1D:
            xname, xfunction, title, nx_bins, x_min, x_max, drawOptions = booked_hist
            xfunction = remove_all_whitespace(xfunction)
            df, xname = define_if_not_observed(df, xfunction, OBSERVED_FORMULAE, xname, columns)
            BOOKED_HISTOGRAMS.append( df.Histo1D((hname, title, nx_bins, x_min, x_max), xname, "weight") )
            DRAW_OPTIONS.append(drawOptions)
        else: # assume 2D
            xname, xfunction, title, nx_bins, x_min, x_max, yname, yfunction, ny_bins, y_min, y_max, drawOptions = booked_hist
            xfunction, yfunction = remove_all_whitespace(xfunction), remove_all_whitespace(yfunction)
            df, xname = define_if_not_observed(df, xfunction, OBSERVED_FORMULAE, xname, columns)
            df, yname = define_if_not_observed(df, yfunction, OBSERVED_FORMULAE, yname, columns)
            BOOKED_HISTOGRAMS.append( df.Histo2D((hname, title, nx_bins, x_min, x_max, ny_bins, y_min, y_max), xname, yname, "weight" ) )
            DRAW_OPTIONS.append(drawOptions)

    return BOOKED_HISTOGRAMS, DRAW_OPTIONS


def turn_on_specifc_waveset(plotGen, results, waveset, verbose=True):
    '''
    Turn on a specific waveset which is an underscore separated list of amplitude names.
       If waveset = all, then turn on all amplitudes

    Example:
        # Will turn on only resAmp1 and resAmp2
        wavesets = 'resAmp1_resAmp2'

    Args:
        plotGen: PlotGenerator
        results: FitResults
        waveset: string
        verbose: bool
    '''
    keepAllAmps = waveset == 'all'

    reactionList = results.reactionList() # vector<string>
    sums = plotGen.uniqueSums() # vector<string>
    amps = plotGen.uniqueAmplitudes() # vector<string>

    amp_map = {}
    if verbose: print(f' >> Plotting waveset: {waveset}')
    waves = waveset.split('_')
    for wave in waves:
        amp_map[wave] = -1 # Placeholder

    # Re-enable all amplitudes in all reactions
    for reaction in reactionList:
        plotGen.enableReaction(reaction)
    # Re-enable all sums
    for i in range(len(sums)):
        plotGen.enableSum(i)

    # Map index of your requested amplitudes
    for i, amp in enumerate(amps):
        if amp in amp_map or keepAllAmps:
            amp_map[amp] = i
    for k, v in amp_map.items():
        if v == -1: print(f' >> WARNING: Amplitude {k} not found in fit results. Exiting...'); exit(1)

    # Turn on only the requested amplitudes
    if not keepAllAmps:
        for i in range(len(amps)):
            plotGen.disableAmp(i)
        for i in amp_map.values:
            plotGen.enableAmp(i)
