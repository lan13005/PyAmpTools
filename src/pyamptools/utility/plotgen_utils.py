from pyamptools.utility.general import remove_all_whitespace
import ROOT
import numpy as np


def define_if_not_observed(df, function, OBSERVED_FORMULAE, name, columns):
    """
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
    """
    if function in OBSERVED_FORMULAE:  # check if function observed
        old_name = OBSERVED_FORMULAE[function]
        return df, old_name
    else:
        OBSERVED_FORMULAE[function] = name
        df = df.Redefine(name, function) if name in columns else df.Define(name, function)  # check if name already exists
        return df, name


def book_histogram(df, HISTS_TO_BOOK, columns):
    """
    Book histograms for a given type (data, bkgnd, accMC, genMC)

    Args:
        HISTS_TO_BOOK: dict of histograms to book, with associated function call, binning scheme, draw options
        columns: list of column names in RDataFrame to check if we need to Redefine

    Returns:
        BOOKED_HISTOGRAMS: list of booked Histos
        DRAW_OPTIONS: list of draw options
    """
    ## ALL HISTOGRAMS FOR A SOURCE FILES
    BOOKED_HISTOGRAMS = []

    OBSERVED_FORMULAE = {}  # Track observed functions matching to defined variable names: {function: variable_name}
    DRAW_OPTIONS = []

    # Booking hists are lazy! Book all of them first, then run filling/Drawing
    for hname, booked_hist in HISTS_TO_BOOK.items():
        histIs1D = len(booked_hist) == 7
        if histIs1D:
            xname, xfunction, title, nx_bins, x_min, x_max, drawOptions = booked_hist
            xfunction = remove_all_whitespace(xfunction)
            df, xname = define_if_not_observed(df, xfunction, OBSERVED_FORMULAE, xname, columns)
            BOOKED_HISTOGRAMS.append(df.Histo1D((hname, title, nx_bins, x_min, x_max), xname, "weight"))
            DRAW_OPTIONS.append(drawOptions)
        else:  # assume 2D
            xname, xfunction, title, nx_bins, x_min, x_max, yname, yfunction, ny_bins, y_min, y_max, drawOptions = booked_hist
            xfunction, yfunction = remove_all_whitespace(xfunction), remove_all_whitespace(yfunction)
            df, xname = define_if_not_observed(df, xfunction, OBSERVED_FORMULAE, xname, columns)
            df, yname = define_if_not_observed(df, yfunction, OBSERVED_FORMULAE, yname, columns)
            BOOKED_HISTOGRAMS.append(df.Histo2D((hname, title, nx_bins, x_min, x_max, ny_bins, y_min, y_max), xname, yname, "weight"))
            DRAW_OPTIONS.append(drawOptions)

    return BOOKED_HISTOGRAMS, DRAW_OPTIONS


def turn_on_specifc_waveset(plotGen, results, waveset, verbose=True):
    """
    Turn on a specific waveset which is an semicolon separated list of amplitude names.
       If waveset = all, then turn on all amplitudes

    Example:
        wavesets = 'resAmp1;resAmp2' # Will turn on only resAmp1 and resAmp2

    Args:
        plotGen: PlotGenerator
        results: FitResults
        waveset: string
        verbose: bool
    """
    keepAllAmps = waveset == "all"

    reactionList = results.reactionList()  # vector<string>
    sums = plotGen.uniqueSums()  # vector<string>
    amps = plotGen.uniqueAmplitudes()  # vector<string>

    amp_map = {}
    if verbose:
        print(f" >> Plotting waveset: {waveset}")
    waves = waveset.split(";")
    if keepAllAmps:
        print(f" >>   Keeping all amplitudes: {amps}")
    _waves = amps if keepAllAmps else waves
    for wave in _waves:
        amp_map[wave] = -1  # Placeholder

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
        if v == -1:
            print(f" >> WARNING: Amplitude {k} not found in fit results. Exiting...")
            exit()

    # Turn on only the requested amplitudes
    if keepAllAmps:
        for i in range(len(amps)):
            plotGen.enableAmp(i)
    else:
        for i in range(len(amps)):
            plotGen.disableAmp(i)
        for i in amp_map.values():
            plotGen.enableAmp(i)


def draw_histograms(
    results,
    hist_output_name,
    particles,
    HISTS_TO_BOOK,
    amplitudes="all",
):
    """
    Draw histograms from a FitResults object. Histograms are booked using the book_histogram() function that uses macros to compute
    kinematic quantities to plot. Booked histograms are lazily evaluated / filled with RDataFrame.

    Args:
        results (FitResults): FitResults object
        hist_output_name (str): Output file name, do not include file type
        particles (List[str]): List of particles in reaction
        HISTS_TO_BOOK (Dict[str, List]): Dictionary of histograms to book. See book_histogram() for details
        amplitudes (str): Space separated list of wavesets to turn on. Wavesets are semi-colon ; separated list of amplitudes. "all" turns on all waves.

    Returns:
        None, dumps a pdf file based on hist_output_name
    """

    THStack = ROOT.THStack
    TCanvas = ROOT.TCanvas
    plotGen = ROOT.PlotGenerator(results)

    assert "." not in hist_output_name, "Do not include file type in the output name ( -o flag )"

    N_BOOKED_HISTS = len(HISTS_TO_BOOK)
    N_PARTICLES = len(particles)

    kData, kBkgnd, kGenMC, kAccMC = plotGen.kData, plotGen.kBkgnd, plotGen.kGenMC, plotGen.kAccMC
    # kNumTypes = plotGen.kNumTypes
    kColors = {kData: ROOT.kBlack, kBkgnd: ROOT.kRed - 9, kAccMC: ROOT.kGreen - 8, kGenMC: ROOT.kAzure - 4}  # for the 4 data sources

    reactionNames = list(results.reactionList())

    ### FOR EACH WAVESET, PLOT THE HISTOGRAMS ###
    amplitudes = amplitudes.split(" ")
    for amp in amplitudes:
        turn_on_specifc_waveset(plotGen, results, amp)

        HISTOGRAM_STORAGE = {}  # {type: [hist1, hist2, ...]}
        DRAW_OPT_STORAGE = {}
        for srctype in [kData, kBkgnd, kGenMC, kAccMC]:
            ########### LOAD THE DATA ###########
            # Reaction: { Variable: [Values] } }
            value_map = plotGen.projected_values(reactionNames, srctype, N_PARTICLES)
            value_map = value_map[srctype]
            value_map = {k: np.array(v) for k, v in value_map}

            df = ROOT.RDF.FromNumpy(value_map)
            columns = df.GetColumnNames()

            ######### RESTRUCTURE DATA FOR NICER CALCULATIONS #########
            df.Define("GLUEXTARGET", "std::vector<float> p{0.938272, 0.0, 0.0, 0.0}; return p;")
            for i, particle in enumerate(particles):
                cmd = f"std::vector<float> p{{ PxP{i}, PyP{i}, PzP{i}, EnP{i} }}; return p;"
                df = df.Define(f"{particle}", cmd)
            # print(df.Describe())

            ######### BOOK HISTOGRAMS #########
            BOOKED_HISTOGRAMS, DRAW_OPTIONS = book_histogram(df, HISTS_TO_BOOK, columns)
            HISTOGRAM_STORAGE[srctype] = BOOKED_HISTOGRAMS
            DRAW_OPT_STORAGE[srctype] = DRAW_OPTIONS

        ###########################################################
        ### NOW CONFIGURE HOW YOU WANT TO DRAW THE HISTOGRAMS ####
        ### HERE IS AN EXAMPLE... BUT IMPOSSIBLE TO MAKE GENERIC #

        nrows = int(np.floor(np.sqrt(len(HISTS_TO_BOOK))))
        ncols = int(np.ceil(len(HISTS_TO_BOOK) / nrows))

        canvas = TCanvas("canvas", "canvas", 1440, 1080)
        canvas.Clear()
        canvas.Divide(ncols, nrows)

        output_name = hist_output_name + f"_{amp}"
        # canvas.Print(f"{output_name}.pdf[")
        stacks = []
        for ihist in range(N_BOOKED_HISTS):
            canvas.cd(ihist + 1)
            data_hist = HISTOGRAM_STORAGE[kData][ihist]
            data_hist.SetMarkerStyle(ROOT.kFullCircle)
            data_hist.SetMinimum(0)
            data_hist.SetMarkerSize(1.0)
            data_hist.Draw("E")  # Draw first to set labels and y-limits
            stacks.append(THStack("stack", ""))
            for srctype in [kBkgnd, kAccMC]:
                booked_hist = HISTOGRAM_STORAGE[srctype][ihist]
                # drawOptions not used ATM but will probably need to be turned back on?
                # drawOptions = DRAW_OPT_STORAGE[srctype][ihist]
                booked_hist.SetFillColorAlpha(kColors[srctype], 1.0)
                booked_hist.SetLineColor(0)
                booked_hist
                hist_ptr = booked_hist.GetPtr()
                stacks[-1].Add(hist_ptr)
            stacks[-1].Draw("HIST SAME")
            data_hist.Draw("E SAME")  # Redraw data

        canvas.Print(f"{output_name}.png")
        # canvas.Print(f"{output_name}.pdf")
        # canvas.Print(f"{output_name}.pdf]")

        # THStack is drawn on TCanvas. Deleting TCanvas (which normally happens when it goes out of scope)
        #   before THStack will lead to improper deallocation. Also deleting elements of stacks in a for loop
        #   does not work. The entire object needs to be deleted
        del stacks
        del canvas
