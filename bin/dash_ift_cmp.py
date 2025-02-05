import argparse
import io
import os
import pickle as pkl
import time
from base64 import b64encode
from multiprocessing import Pool

import dash
import iftpwa1
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from iftpwa1.utilities.helpers import reload_fields_and_components
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from plotly.subplots import make_subplots

from pyamptools.utility.general import prettyLabels

# TODO:
# * t-dependent plot figures (currently only t=0)
# * option to draw N samples or N% of samples
# * option to draw only specific waves
# * cache figures based on a hash of their (t, waves, models) maybe samples also
#   similarly the DF should not recreate subsets. If the data exists then just load it and draw the figure


def append_info(resultData, model_info, df, avail_waves, avail_phase_pairs, acceptance_correct=True):
    """
    Read a iftpwa results pkl file and store intensites and phases in a dataframe
    """
    model_name = model_info['name']
    scale_factor = model_info.get('scale', 1.0)  # Default to 1.0 if not specified

    resonance_paramater_data_frame = {}
    for resonance_parameter in resultData["fit_parameters_dict"]:
        if "scale" not in resonance_parameter:
            resonance_paramater_data_frame[resonance_parameter] = np.array(resultData["fit_parameters_dict"][resonance_parameter])

    resonance_paramater_data_frame = pd.DataFrame(resonance_paramater_data_frame)

    ##################################################################################################
    # THIS SECTION CONTAINS BASICALLY ALL YOU NEED TO START CREATING CUSTOM PLOTS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Load fields and parametric components
    _result = reload_fields_and_components(resultData=resultData)

    # signal_field_sample_values, amp_field, _res_amps_waves_tprime, _bkg_amps_waves_tprime, kinematic_mutliplier = _result
    signal_field_sample_values, amp_field, res_amps_waves_tprime, bkg_amps_waves_tprime, kinematic_mutliplier, threshold_selector, modelName, wave_names, wave_parametrizations, paras, resonances, calc_intens = _result

    threshold_selector = kinematic_mutliplier > 0

    # Reload general information
    wave_names = resultData["pwa_manager_base_information"]["wave_names"]
    wave_names = np.array([wave.split("::")[-1] for wave in wave_names])
    nmb_waves = len(wave_names)

    mass_bins = resultData["pwa_manager_base_information"]["mass_bins"]
    masses = 0.5 * (mass_bins[1:] + mass_bins[:-1])
    mass_limits = (np.min(mass_bins), np.max(mass_bins))

    tprime_bins = resultData["pwa_manager_base_information"]["tprime_bins"]
    tprimes = 0.5 * (tprime_bins[1:] + tprime_bins[:-1])

    prior_params = [par for par in resultData["fit_parameters_dict"].keys() if "scale_phase" not in par]

    nmb_samples, dim_signal, nmb_masses, nmb_tprime = signal_field_sample_values.shape  # Dimensions
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ##################################################################################################

    _masses = masses.tolist()

    for it in range(nmb_tprime):
        for iw, wave in enumerate(wave_names):
            amp = [signal_field_sample_values[:, 2 * iw, :, it] + 1j * signal_field_sample_values[:, 2 * iw + 1, :, it]]
            intens = calc_intens([wave], it, amp)
            assert len(intens.shape) == 2  # (nmb_samples, nmb_masses)
            intes = intens.flatten() * scale_factor  # Apply scaling factor here
            df[wave] += intes.tolist()

        # Some models might have less waves than others
        missing_waves = set(avail_waves) - set(wave_names)
        for wave in missing_waves:
            df[wave] += np.zeros(nmb_samples * nmb_masses).tolist()

        # Append phase differences. Only upper triangle
        phase_pairs = []
        for _iw in range(len(wave_names)):
            for _jw in range(_iw + 1, len(wave_names)):
                iw, jw = _iw, _jw
                phase_pair = f"{wave_names[iw]}_{wave_names[jw]}"
                if wave_names[iw][-1] != wave_names[jw][-1]:  # incoherent sectors, fill them later
                    continue
                if phase_pair not in avail_phase_pairs:  # try swapping them
                    iw, jw = _jw, _iw
                    phase_pair = f"{wave_names[iw]}_{wave_names[jw]}"
                if phase_pair not in avail_phase_pairs:
                    raise ValueError(f"Phase pair {phase_pair} not found in available phase pairs")
                if phase_pair in phase_pairs:  # already added
                    continue
                phase_pairs.append(phase_pair)
                amp = signal_field_sample_values[:, iw * 2, :, 0] + 1j * signal_field_sample_values[:, iw * 2 + 1, :, 0]
                ref_amp = signal_field_sample_values[:, jw * 2, :, 0] + 1j * signal_field_sample_values[:, jw * 2 + 1, :, 0]
                if np.all(np.abs(ref_amp) < 1e-10):  # precision check
                    ref_phase = np.zeros_like(ref_amp)
                else:
                    ref_phase = np.divide(np.conj(ref_amp), np.abs(ref_amp), out=np.zeros_like(ref_amp), where=np.abs(ref_amp) != 0)  # avoid division by zero
                if np.all(np.abs(amp) < 1e-10):
                    rotated_amp = np.zeros_like(amp)
                else:
                    rotated_amp = amp * ref_phase

                angles = np.unwrap(np.angle(rotated_amp, deg=True), period=360)

                # import scipy.optimize as opt
                # def total_absolute_deviation(median_curve, data):
                #     median_curve = np.array(median_curve).reshape(-1)
                #     return np.sum(np.abs(data - median_curve))

                # Initial guess: Point-wise median as the starting point
                # initial_guess = np.unwrap(np.median(angles, axis=0), period=360)
                # result = opt.minimize(
                #     fun=total_absolute_deviation,
                #     x0=initial_guess,  # Initial guess
                #     args=(angles,),
                #     method='L-BFGS-B',  # Optimization method
                # )

                # baseline = result.x # get values
                # aligned_angles = np.zeros_like(angles)
                # for i, curve in enumerate(angles):
                #     diff = curve - baseline
                #     shift = np.round(diff / 360) * 360
                #     aligned_angles[i] = curve - shift
                # aligned_angles = aligned_angles.flatten()

                aligned_angles = angles.flatten()
                df[phase_pair] += aligned_angles.tolist()

        # Some models might have less phase pairs than others
        missing_phase_pairs = set(avail_phase_pairs) - set(phase_pairs)
        for phase_pair in missing_phase_pairs:
            df[phase_pair] += np.zeros(nmb_samples * nmb_masses).tolist()

        df["mass"] += _masses * nmb_samples  # list multiply repeats list nmb_samples times
        df["t"] += [tprimes[it]] * nmb_samples * nmb_masses
        df["sample"] += np.repeat(np.arange(nmb_samples), nmb_masses).tolist()
        df["model"] += [model_name] * nmb_samples * nmb_masses

    return df


def generate_trace(args):
    """Multiprocessing worker function to generate traces"""
    variable, sample, color_map, group, showlegend = args
    nmbSamples = len(sample["sample"].unique())
    traces = []
    # print(f"Generating traces for {variable} in {group} with {nmbSamples} samples")
    for i in range(nmbSamples):
        showlegend = bool(showlegend * (i == 0))
        traces.append((variable, go.Scattergl(x=sample.query(f"sample == {i}")["mass"], y=sample.query(f"sample == {i}")[variable], mode="lines", name=group, line=dict(width=1, color=color_map[group]), opacity=0.3, legendgroup=group, showlegend=showlegend)))
    return traces


def generate_figure_with_multiprocessing(models, t, acceptance_correct, cache_loc, ntasks):
    """
    Generate the plotly figure using multiprocessing to generate the plot traces
    """

    cache_exists = False
    fig = None

    ################################################################
    ############## CHECK IF PREVIOUS DATAFRAME EXISTS ##############
    ################################################################

    if os.path.exists(cache_loc):
        with open(cache_loc, "rb") as f:
            cache = pkl.load(f)
            try:
                if models == cache["models"]:
                    print(f"Cache available in {cache_loc}. Loading...")
                    df = cache["df"]
                    avail_waves = cache["avail_waves"]
                    avail_phase_pairs = cache["avail_phase_pairs"]
                    if "fig" in cache:
                        fig = cache["fig"]
                    cache_exists = True
            except Exception as e:
                print("Error loading cache. Generating new dataframe...")

    ################################################################
    ############# GENERATE DATAFRAME IF NOT CACHED #################
    ################################################################

    if not cache_exists:
        print("Creating initial dataframe...")
        resultDatas = {}

        avail_waves = set()
        for model_name, model_info in models.items():
            resultData = pkl.load(open(f"{model_info['path']}", "rb"))
            resultDatas[model_name] = resultData
            avail_waves.update(resultData["pwa_manager_base_information"]["wave_names"])

        # Sort first by sector (reflectivity) then L, then M
        spec_map = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}
        pm_map = {"p": -1, "m": 1}  # NOTE! we swap their values to sort in descending order
        sign_map = {"+": 0, "-": 1}  # want + refl first
        avail_waves = sorted(list(avail_waves), key=lambda x: (sign_map[x[-1]], spec_map[x[0]], pm_map[x[1]] * int(x[2])))

        avail_phase_pairs = []
        for i in range(len(avail_waves)):
            for j in range(i + 1, len(avail_waves)):
                if avail_waves[i][-1] != avail_waves[j][-1]:
                    continue  # incoherent sectors
                avail_phase_pairs.append(f"{avail_waves[i]}_{avail_waves[j]}")

        df = {k: [] for k in ["mass", "t", "sample", "model"] + list(avail_waves) + avail_phase_pairs}

        for model_name, resultData in resultDatas.items():
            df = append_info(resultData, models[model_name], df, avail_waves, avail_phase_pairs, acceptance_correct)

        try:
            df = pd.DataFrame(df)
        except Exception as e:
            print("Array Lengths\n-------------")
            for k, v in df.items():
                print(f"{k}: {len(v)}")
            raise e

        cache_dump = {"models": models, "df": df, "avail_waves": avail_waves, "avail_phase_pairs": avail_phase_pairs}

        with open(cache_loc, "wb") as f:
            print("saving to dataframe to cache...")
            pkl.dump(cache_dump, f)

    ################################################################
    ########## GENERATE PLOTY FIGURE USING MULTIPROCESSING #########
    ################################################################

    # fig = None # for testing
    fig = None
    if fig is not None:
        print("reloading figure from cache...")
    else:
        print(f"creating {len(avail_waves)}x{len(avail_waves)} figure...")

        fig = make_subplots(
            rows=len(avail_waves),
            cols=len(avail_waves),
            horizontal_spacing=0.2 / len(avail_waves),
            vertical_spacing=0.2 / len(avail_waves),
        )

        # Get some colors for each group
        colors = px.colors.qualitative.Plotly
        color_map = {k: colors[i] for i, k in enumerate(models.keys())}

        # Prepare arguments for multiprocessing
        tasks = []
        for group in models.keys():
            sample = df.query(f"t=={t} and model == '{group}'")
            for iw, wave in enumerate(avail_waves):
                showlegend = iw == 0  # Do this once per group!
                tasks.append((wave, sample, color_map, group, showlegend))
            for ip, phase_pair in enumerate(avail_phase_pairs):
                tasks.append((phase_pair, sample, color_map, group, False))

        # Use multiprocessing to generate traces
        with Pool(ntasks) as pool:
            trace_groups = pool.map(generate_trace, tasks)

        # Add traces to the figure
        for trace_group in trace_groups:
            for var_trace in trace_group:
                variable, trace = var_trace
                if "_" not in variable:
                    iw = avail_waves.index(variable)
                    fig.add_trace(trace, row=iw + 1, col=iw + 1)
                else:
                    iw, jw = avail_waves.index(variable.split("_")[0]), avail_waves.index(variable.split("_")[1])
                    fig.add_trace(trace, row=iw + 1, col=jw + 1)

        print("adding annotations...")

        # Display amplitude name on intensity plots (diagonal elements)
        for iw, wave in enumerate(avail_waves):
            fig.add_annotation(
                text=prettyLabels[wave],  # Proper MathJax syntax for LaTeX
                x=0.95,
                y=0.95,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                font=dict(size=20),
                row=iw + 1,
                col=iw + 1,
            )

        # Display the phase difference pair in the (off-diagonal) plots
        for i, phase_pair in enumerate(avail_phase_pairs):
            amp1, amp2 = phase_pair.split("_")
            amp1, amp2 = prettyLabels[amp1].replace("$", ""), prettyLabels[amp2].replace("$", "")
            fig.add_annotation(
                text=f"$\phi({amp1}, {amp2})$",  # Proper MathJax syntax for LaTeX
                x=0.1,
                y=0.95,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                font=dict(size=12),
                row=avail_waves.index(phase_pair.split("_")[0]) + 1,
                col=avail_waves.index(phase_pair.split("_")[1]) + 1,
            )

        fig.update_layout(
            title="",
            height=175 * len(avail_waves),
            width=175 * len(avail_waves),
            showlegend=True,
            # dynamically set legend y position based on number of rows where it will be placed under the top left subplot where only upper triangle is plotted
            legend=dict(
                x=0.0,
                y=0.985 - 1.0 / (len(avail_waves)+1),
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0)",  # Transparent background
                bordercolor="black",
                borderwidth=1,
                itemsizing="constant",
            ),
        )

        # Override legend markers to show full-opacity lines
        for trace in fig.data:
            if trace.showlegend:
                trace.update(opacity=None, legendgrouptitle=dict(font=dict(size=12)))  # Ensure full opacity for legend markers

        # write to cache (to save reloading times)
        with open(cache_loc, "rb") as f:
            cache_dump = pkl.load(f)
            cache_dump["fig"] = fig
        with open(cache_loc, "wb") as f:
            print("saving fig to cache...")
            pkl.dump(cache_dump, f)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a Dash app comparing IFT PWA results (intensity and phases)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("yaml_file", type=str, help=(
        "The YAML file should (some optional) define the following keys:\n\n"
        "    - models: (required, dict) A dictionary mapping model names to IFT PWA result folders.\n"
        "    - t: (required, float) compare a specific t-bin\n"
        "    - no_browser (optional, bool): Do not open the browser, just dumo the results.\n"
        "    - cache_loc (optional, str): Path to the cache file for storing intermediate results (default: '.dash_ift_cache.pkl').\n"
        "    - ntasks (optional, int): Number of tasks to use for drawing plotly traces(default: 4).\n"
        "    - acceptance_correct (optional, bool): Whether to apply acceptance correction (default: True).\n"
        "    - html_dump (optional, str): Name of the output HTML file (default: 'dash_iftpwa.html').\n"
        "    - html_dump_static (optional, str): Name of the output HTML file (default: 'dash_iftpwa.html').\n"
        "    - pdf_dump (optional, str): Name of the output PDF file (default: 'dash_iftpwa.pdf').\n"
        "    - static_html (optional, bool): save dumped html in static (non-interactive) form, can be faster\n"
        "Example YAML file:\n\n"
        "    dash:\n"
        "        models:\n"
        "            Model1:\n"
        "               path: '/path/to/ift_results_model1'\n"
        "               scale: 1.0\n"
        "            Model2:\n"
        "               path: '/path/to/ift_results_model2'\n"
        "               scale: 1.0\n"
        "        t: 1.0\n"
        "        cache_loc: /path/to/cache.pkl\n"
        "        no_browser: false\n"
        "        ntasks: 4\n"
        "        acceptance_correct: true\n"
        "        html_dump_static: /path/to/dash_iftpwa.html\n"
        "        html_dump: /path/to/dash_iftpwa.html\n"
        "        pdf_dump: /path/to/dash_iftpwa.pdf\n"
        "        static_html: false\n"
        )
    )

    args = parser.parse_args()
    yaml_file = args.yaml_file

    config = OmegaConf.load(yaml_file)
    dash_cfg = config["dash"]
    if "models" not in dash_cfg:
        raise ValueError("models must be specified in the YAML file")
    if "t" not in dash_cfg:
        raise ValueError("t must be specified in the YAML file")
    models = {}
    for model_name, model_config in dash_cfg["models"].items():
        if isinstance(model_config, str):
            # Support old format where model_config is just a path
            models[model_name] = {'name': model_name, 'path': model_config, 'scale': 1.0}
        elif isinstance(model_config, (dict, DictConfig)):
            # New format with path and scale
            models[model_name] = {
                'name': model_name,
                'path': model_config['path'],
                'scale': model_config.get('scale', 1.0)
            }
        else:
            raise ValueError(f"Invalid model configuration for {model_name}")

    t = dash_cfg["t"] if "t" in dash_cfg else None
    cache_loc = dash_cfg["cache_loc"] if "cache_loc" in dash_cfg else ".dash_ift_cache.pkl"
    no_browser = dash_cfg["no_browser"] if "no_browser" in dash_cfg else False
    ntasks = dash_cfg["ntasks"] if "ntasks" in dash_cfg else 4
    acceptance_correct = dash_cfg["acceptance_correct"] if "acceptance_correct" in dash_cfg else True
    html_dump_static = dash_cfg["html_dump_static"] if "html_dump_static" in dash_cfg else None
    html_dump = dash_cfg["html_dump"] if "html_dump" in dash_cfg else "dash_iftpwa.html"
    pdf_dump  = dash_cfg["pdf_dump"]  if "pdf_dump"  in dash_cfg else None
    static_html = dash_cfg["static_html"] if "static_html" in dash_cfg else None
    
    if models is None or t is None:
        raise ValueError("models and t must be specified in the YAML file")

    print(f"\n-> Generating figure for models at t={t}:")
    for model_name, model_info in models.items():
        print(f"    - {model_name}: {model_info['path']} (scale: {model_info['scale']})")
    print(f"-> Writing/reading Cache location: {cache_loc}")
    print(f"-> Number of tasks for drawing: {ntasks}\n")

    start_time = time.time()
    fig = generate_figure_with_multiprocessing(models, t, acceptance_correct, cache_loc, ntasks)

    buffer = io.StringIO()
    
    static_config = {}
    if html_dump_static:
        static_config = {  # This will not affect dash's interactivity. It will affect the static html dumped
            "staticPlot": True,  # Disable full interactivity
            "responsive": True,  # Ensure responsiveness
            "displayModeBar": False,  # Hide the mode bar for a cleaner look
        }
    fig.write_html(buffer, include_mathjax="cdn", config=static_config)
    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()
    if html_dump: # non-empty strings will eval to true also
        print(f"user requested static html dump. Writing to {html_dump}")
        with open(html_dump, "wb") as f:
            f.write(html_bytes)
        print(" -> html exported!")
    if pdf_dump: 
        print(f"user requested pdf dump. Writing to {pdf_dump}")
        fig.write_image(pdf_dump)
        print(" -> pdf exported!")

    gen_time = time.time() - start_time

    print("took {:.2f} seconds to generate figure".format(gen_time))
    print("\nfinal step! rendering dash app. Can take awhile...")

    # Create Dash app
    iftpwa_folder = "/".join(iftpwa1.__file__.split("/")[:-2]) + "/docs/images"
    
    if not no_browser:
        
        app = dash.Dash(__name__, assets_folder=iftpwa_folder)

        app.layout = html.Div(
            [
                html.Div(
                    [
                        html.Img(src=app.get_asset_url("logo.png"), style={"height": "50px", "float": "left", "margin-right": "10px"}),
                    ],
                    style={"display": "flex", "align-items": "center"},
                ),  # Flexbox for alignment
                html.A(html.Button("Download as HTML"), id="download", href="data:text/html;base64," + encoded, download=f"{html_dump_static}"),
                dcc.Graph(id="dash_iftpwa", figure=fig, mathjax=True),  # need mathjax here to render latex
            ]
        )

        app.run_server(debug=True)
