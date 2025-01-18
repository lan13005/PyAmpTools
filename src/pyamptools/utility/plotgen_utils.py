import numpy as np
import ROOT
from pyamptools.utility.general import remove_all_whitespace


def calculate_chi_squared(data_hist, mc_hist):
    chi2 = 0
    ndf = 0
    for bin in range(1, data_hist.GetNbinsX() + 1):
        observed = data_hist.GetBinContent(bin)
        expected = mc_hist.GetBinContent(bin)
        error = data_hist.GetBinError(bin)
        
        if error > 0:
            chi2 += ((observed - expected) ** 2) / (error ** 2)
            ndf += 1
    
    # Reduce by the number of parameters (usually the number of bins)
    reduced_chi2 = chi2 / ndf if ndf > 0 else float('inf')
    return reduced_chi2

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
		wavesets = 'resAmp1;resAmp2' # Will turn on only resAmp1 and resAmp2.
		AmpTools uses Reaction::Sum::Amp naming structure for amplitudes.
		The example will turn on all amplitudes with 'resAmp1' or 'resAmp2' anywhere in their name

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

	# Initialize a map of matching amplitude names to their index
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
	wavesets="all",
	stack_background=False,
	plot_acc_corrected=False,
	output_format="pdf"
):
	"""
	Draw histograms from a FitResults object. Histograms are booked using the book_histogram() function that uses macros to compute
	kinematic quantities to plot. Booked histograms are lazily evaluated / filled with RDataFrame.

	Args:
		results (FitResults): FitResults object
		hist_output_name (str): Output file name, do not include file type
		particles (List[str]): List of particles in reaction
		HISTS_TO_BOOK (Dict[str, List]): Dictionary of histograms to book. See book_histogram() for details
		wavesets (str): Space separated list of wavesets to turn on. Wavesets are semi-colon ; separated list of amplitudes. "all" turns on all waves.
		stack_background (bool): Stack backgrounds instead of subtracting them
		plot_acc_corrected (bool): Plot acceptance corrected data, compare to weighted genmc
		output_format (str): Output file format, "pdf" or "png"

	Returns:
		None, dumps a pdf file based on hist_output_name
	"""
 
	output_format = output_format.lower()
	assert output_format in ["pdf", "png"], "Output format must be 'pdf' or 'png'"
 
	THStack = ROOT.THStack
	TCanvas = ROOT.TCanvas
	plotGen = ROOT.PlotGenerator(results)
	cfgInfo = plotGen.cfgInfo()

	print("Loaded data into PlotGenerator...")

	assert "." not in hist_output_name, "Do not include file type in the output name ( -o flag )"

	N_BOOKED_HISTS = len(HISTS_TO_BOOK)
	N_PARTICLES = len(particles)

	kData, kBkgnd, kGenMC, kAccMC = plotGen.kData, plotGen.kBkgnd, plotGen.kGenMC, plotGen.kAccMC
	# kNumTypes = plotGen.kNumTypes
	kColors = {kData: ROOT.kBlack, kBkgnd: ROOT.kRed - 9, kAccMC: ROOT.kGreen - 8, kGenMC: ROOT.kAzure - 4}  # for the 4 data sources

	reactionNames = list(results.reactionList())
	zlm_polAngles = [float(cfgInfo.amplitudeList(reactionName, "", "").at(0).factors().at(0).at(5)) for reactionName in reactionNames]
	
	### FOR EACH WAVESET, PLOT THE HISTOGRAMS ###
	wavesets = wavesets.split(" ")
	for waveset in wavesets:
		turn_on_specifc_waveset(plotGen, results, waveset)

		HISTOGRAM_STORAGE = {}  # {type: [hist1, hist2, ...]}
		DRAW_OPT_STORAGE = {}
		for srctype in [kData, kBkgnd, kGenMC, kAccMC]:

			########### LOAD THE DATA ###########
			# Reaction: { Variable: [Values] } }
			value_map = {}
			for reactionName, zlm_polAngle in zip(reactionNames, zlm_polAngles):
				_value_map = plotGen.projected_values([reactionName], srctype, N_PARTICLES)
				_value_map = dict(_value_map[srctype])
				n_entries = len(_value_map['EnP0'])
				_value_map["POLANGLE"] = np.full(n_entries, zlm_polAngle, dtype=np.float32)
				for k, v in _value_map.items():
					if k not in value_map:
						value_map[k] = []
					value_map[k].extend(v)
			value_map = {k: np.array(v, dtype=np.float32) for k, v in value_map.items()}
			df = ROOT.RDF.FromNumpy(value_map)
			columns = df.GetColumnNames()

			######### ADD NEW COLUMNS DATA FOR NICER CALCULATIONS #########
			df = df.Define("GLUEXTARGET", "std::vector<float> p{0.0, 0.0, 0.0, 0.938272}; return p;")
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
		# ncols = int(np.ceil(np.sqrt(len(HISTS_TO_BOOK))))
		# nrows = ncols

		canvas = TCanvas("canvas", "canvas", ncols*400, nrows*400)
		canvas.Clear()
		canvas.Divide(ncols, nrows)

		if plot_acc_corrected:
			canvas_gen = TCanvas("canvas_gen", "canvas_gen", ncols*400, nrows*400)
			canvas_gen.Clear()
			canvas_gen.Divide(ncols, nrows)
			corrected_data_hists = []
			efficiency_hists = []

		# Turn off some axes if unused
		# for ihist in range(N_BOOKED_HISTS, ncols*nrows):
		# 	canvas.cd(ihist + 1)
		# 	ROOT.gPad.SetFrameLineWidth(0)
		# 	ROOT.gPad.SetBorderSize(0)
		# 	ROOT.gPad.SetTickx(0)
		# 	ROOT.gPad.SetTicky(0)
		# 	ROOT.gPad.SetFillColor(0)


		latex = ROOT.TLatex()
		latex.SetNDC()
		latex.SetTextSize(0.07)
		latex.SetTextColor(ROOT.kRed)

		_waveset = "_".join(waveset.split(";"))
		output_name = hist_output_name + f"_{_waveset}"
		# canvas.Print(f"{output_name}.pdf[")
		stacks = []
		for ihist in range(N_BOOKED_HISTS):

			data_hist = HISTOGRAM_STORAGE[kData][ihist]
			data_hist.SetMarkerStyle(ROOT.kFullCircle)
			data_hist.SetMarkerSize(1.0)
			data_hist.GetXaxis().SetNdivisions(5,5,0)
			data_hist.GetYaxis().SetNdivisions(5,5,0)

			bkgnd_hist = HISTOGRAM_STORAGE[kBkgnd][ihist]
			accmc_hist = HISTOGRAM_STORAGE[kAccMC][ihist]
			accmc_hist.SetFillColorAlpha(kColors[kAccMC], 0.9)
			accmc_hist.SetLineWidth(0)
			accmc_hist.GetXaxis().SetNdivisions(5,5,0)
			accmc_hist.GetYaxis().SetNdivisions(5,5,0)

			data_hist.Sumw2()
			bkgnd_hist.Sumw2()
			accmc_hist.Sumw2()
			
			if plot_acc_corrected:
				genmc_hist = HISTOGRAM_STORAGE[kGenMC][ihist]
				genmc_hist.Sumw2()
				genmc_hist.SetFillColorAlpha(kColors[kGenMC], 0.9)
				genmc_hist.SetLineWidth(0)
				genmc_hist.GetXaxis().SetNdivisions(5,5,0)
				genmc_hist.GetYaxis().SetNdivisions(5,5,0)
				corrected_data_hists.append(data_hist.Clone())
				corrected_data_hists[-1].SetMarkerStyle(ROOT.kFullCircle)
				corrected_data_hists[-1].SetMarkerSize(1.0)
				corrected_data_hists[-1].GetXaxis().SetNdivisions(5,5,0)
				corrected_data_hists[-1].GetYaxis().SetNdivisions(5,5,0)
				corrected_data_hists[-1].Sumw2()

			canvas.cd(ihist + 1)

			if not stack_background: # SUBTRACT BACKGROUND

				data_hist.Add(bkgnd_hist.GetPtr(), -1)
				if plot_acc_corrected:
					corrected_data_hists[-1].Add(bkgnd_hist.GetPtr(), -1)
     
				# data_hist.Draw("E") # Draw first to set labels and y-limits
				# data_hist.GetYaxis().SetLabelOffset(0.01)
				# data_hist.GetYaxis().SetTitleOffset(1.70)
				# accmc_hist.Draw("HIST SAME")
        
				accmc_hist.Draw("HIST")
				data_hist.Draw("E SAME")
				accmc_hist.GetYaxis().SetLabelOffset(0.01)
				accmc_hist.GetYaxis().SetTitleOffset(1.70)
				accmc_hist.SetMinimum(0)
				max_y = 1.2 * max(data_hist.GetMaximum(), accmc_hist.GetMaximum())
				accmc_hist.SetMaximum(max_y)

				# chi2 = data_hist.Chi2Test(accmc_hist.GetPtr(), "CHI2/NDF")
				chi2 = calculate_chi_squared(data_hist, accmc_hist)
				latex.DrawLatex(0.25, 0.87, f"#chi^{{2}}/bin = {chi2:.1f}")

				if plot_acc_corrected: # Overlay acceptance corrected data only if bkgnd subtracted
					canvas_gen.cd(ihist + 1)
					efficiency_hists.append(accmc_hist.Clone())
					efficiency_hists[-1].Divide(genmc_hist.GetPtr())

					# set bins where there is very little accmc data as it will lead to large errs in efficiency
					for bin in range(1, efficiency_hists[-1].GetNbinsX() + 1):
						if accmc_hist.GetBinContent(bin) < 20:
							efficiency_hists[-1].SetBinContent(bin, 0)

					corrected_data_hists[-1].Divide(efficiency_hists[-1])
     
					genmc_hist.Draw("HIST")
					genmc_hist.GetYaxis().SetLabelOffset(0.01)
					genmc_hist.GetYaxis().SetTitleOffset(1.70)
					corrected_data_hists[-1].Draw("E SAME")
					genmc_hist.SetMinimum(0)
					max_y = 1.2 * max(corrected_data_hists[-1].GetMaximum(), genmc_hist.GetMaximum())
					genmc_hist.SetMaximum(max_y)

					chi2_gen = calculate_chi_squared(corrected_data_hists[-1], genmc_hist)
					latex.DrawLatex(0.25, 0.87, f"#chi^{{2}}/bin = {chi2_gen:.1f}")

			else: # STACK BACKGROUND
				stacks.append(THStack("stack", ""))
				for booked_hist, srctype in zip([bkgnd_hist, accmc_hist], [kBkgnd, kAccMC]):
					booked_hist.SetFillColorAlpha(kColors[srctype], 0.9)
					booked_hist.SetLineWidth(0)
					hist_ptr = booked_hist.GetPtr()
					stacks[-1].Add(hist_ptr)
				stacks[-1].Draw("HIST")
				data_hist.Draw("E SAME")
				stacks[-1].SetMinimum(0)
				stacks[-1].GetYaxis().SetLabelOffset(0.01)
				stacks[-1].GetYaxis().SetTitleOffset(1.70)
				max_y = 1.2 * max(data_hist.GetMaximum(), stacks[-1].GetStack().Last().GetMaximum())
				stacks[-1].SetMaximum(max_y)

				chi2_stack = calculate_chi_squared(data_hist, stacks[-1].GetStack().Last())
				latex.DrawLatex(0.25, 0.87, f"#chi^{{2}}/bin = {chi2_stack:.1f}")
  
		canvas.Print(f"{output_name}.{output_format}")
		if plot_acc_corrected:
			canvas_gen.Print(f"{output_name}_acc_corrected.{output_format}")
		# canvas.Print(f"{output_name}.pdf")
		# canvas.Print(f"{output_name}.pdf]")

		# THStack is drawn on TCanvas. Deleting TCanvas (which normally happens when it goes out of scope)
		#   before THStack will lead to improper deallocation. Also deleting elements of stacks in a for loop
		#   does not work. The entire object needs to be deleted
		del stacks
		del canvas
