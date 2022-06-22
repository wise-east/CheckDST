import json

from zmq import ctx_opt_names
from loguru import logger
from collections import defaultdict
import numpy as np
import pandas as pd
import glob
import re
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as Path_mpl
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from pathlib import Path


TARGET_METRICS = [
    "NoHF Orig",
    "TP cJGA",
    # "NoHF Swap",
    "SD cJGA",
    "test_jga",
    "coref_jga",
    "NEI cJGA",
]


def get_test_results_in_train_stats(trainstats_fn):
    test_result = None
    is_faulty_json = False

    ### try loading the json file as is
    try:
        with open(trainstats_fn, "r") as f:
            trainstats = json.load(f)
            # trainstats =f.read()
        # print(trainstats.keys())
        # print(str(trainstats_fn))
        test_result = trainstats['final_test_report']
        return test_result, is_faulty_json
    except Exception as e:
        # print(e)
        print(e)
        is_faulty_json = True

    ### sometimes, the trainstats file is saved as problematic json files.
    ### try loading as txt file and force the loading for the final_test_report column only
    with open(trainstats_fn, "r") as f:
        trainstat_txt = f.read()
    # trainstat_txt = re.search('"final_test_report": .*', trainstat_txt)[0]
    idx_start = trainstat_txt.find('"final_test_report": ')
    trainstat_txt = trainstat_txt[
        idx_start : idx_start + trainstat_txt[idx_start:].find("},") + 1
    ]
    json_txt = r"{" + trainstat_txt + "}"
    try:
        trainstats = json.loads(json_txt)
        test_result = trainstats['final_test_report']

    except Exception as e:
        logger.info(e)
        logger.info(json_txt)

    return test_result, is_faulty_json


def print_jga_summary(model_dirs: str):
    model_dirs = list(Path("./").glob(model_dirs))
    faulty_json_ct = 0
    properly_loaded = 0
    total = 0
    grouped_results = defaultdict(list)

    few_shot_jgas = []
    full_shot_jgas = []
    for md in model_dirs:
        print(md)
        subdirs = md.glob('*')
        for sd in sorted(subdirs):
            # print(sd)
            total += 1
            trainstats_fn = sd / "model.trainstats"
            # print(trainstats_fn.is_file())
            test_result, faulty_json = get_test_results_in_train_stats(trainstats_fn)
            faulty_json_ct += faulty_json
            trainconfig = trainstats_fn.parts[-2][
                trainstats_fn.parts[-2]
                .find("fs") : trainstats_fn.parts[-2]
                .rfind("_sd")
            ]
            # print(trainconfig)
            # break
            # print(test_result)

            ### skip if either could not be loaded
            if not test_result:
                error = (
                    "failed to load JSON"
                    if test_result is None
                    else "empty test results"
                )
                print(f"{error}: {trainstats_fn}")
                continue
            else:
                # print(f"Properly loaded: {trainstats_fn}")
                properly_loaded += 1
                jga = test_result['joint goal acc']
                grouped_results[trainconfig].append(jga)

        print(f"{Path(md).parts[-1]},median jga,mean jga,count")
        for k, v in grouped_results.items():
            print(
                f"{k},{np.median(v):.4f},{np.mean(v):.4f},{len(v)}, [{[round(res, 4) for res in v]}]"
            )

        # print(f"Fewshot results: \n\tMean: {np.mean(few_shot_jgas):.4f}\n\tMedian: {np.median(few_shot_jgas):.4f}\n\tSTDEV: {np.std(few_shot_jgas):.4f}")

        # print(f"Fullshot results: \n\tMean: {np.mean(full_shot_jgas):.4f}\n\tMedian: {np.median(full_shot_jgas):.4f}\n\tSTDEV: {np.std(full_shot_jgas):.4f}")

        # pprint(stats.describe(few_shot_jgas))
        # pprint(stats.describe(full_shot_jgas))

        print(faulty_json_ct, properly_loaded, total)


def get_parlai_inv_results(main_dir, print_=False, epoch_precision=1):
    if main_dir[-1] == "/":
        main_dir = main_dir[:-1]
    subdirs = glob.glob(f"{main_dir}/*")
    # print(subdirs)
    # print(len(subdirs))
    fullshot_results = []
    fewshot_results = []

    for sd in subdirs:

        if print_:
            print(sd)

        sd_name = Path(sd).name
        if "_sd" not in sd_name:
            continue
        config = sd_name[: sd_name.index("_sd")]
        # print(sd_name)
        # print(config)

        report_files = glob.glob(f"{sd}/*report*.json")

        # for rf in report_files:
        # print(rf)
        for aug in ["SD", "TP", "NEI"]:
            inv_report_files = sorted(
                [rf for rf in report_files if aug in rf],
                key=lambda x: int(
                    re.search("step([0-9]*)", x)[1] if "step" in x else 0
                ),
            )
            if print_:
                print(f"{aug} inv report files: {len(inv_report_files)}")
            # break
            ct = 1
            for irf in inv_report_files:
                print(irf)

                seed = re.search("sd([0-9]*)", irf)[1]
                step = re.search("step([0-9]*)", irf)
                if step:
                    step = step[1]
                else:
                    step = 0

                # load invariance report file
                with open(irf, "r") as f:
                    inv_result = json.load(f)

                # step numbers are inconsistent. get epoch number from trainstats file
                # TODO what todo when not available?
                ts_fn = Path(irf).with_suffix("").with_suffix(".trainstats")
                # print(ts_fn)

                if ts_fn.is_file():
                    with ts_fn.open("r") as f:
                        trainstat = json.load(f)
                    epochs = (
                        round(trainstat["total_epochs"] / epoch_precision)
                        * epoch_precision
                    )
                else:
                    epochs = ct
                    ct += 1
                    print(epochs)

                # load test report & valid report results
                test_fn = Path(irf).with_suffix("").with_suffix(".test_report")
                valid_fn = Path(irf).with_suffix("").with_suffix(".valid_report")
                if test_fn.is_file():
                    with test_fn.open("r") as f:
                        test_results = json.load(f)
                        test_jga = test_results["report"]["joint goal acc"]
                        coref_jga = test_results["report"]["coref_jga"]
                else:
                    test_jga = None
                    coref_jga = None
                if valid_fn.is_file():
                    with valid_fn.open("r") as f:
                        valid_results = json.load(f)
                        valid_jga = valid_results["report"]["joint goal acc"]
                else:
                    valid_jga = None

                result = {
                    "config": config,
                    "seed": seed,
                    "step": int(step),
                    "epochs": epochs,
                    "inv": aug,
                    "valid_jga": valid_jga,
                    "test_jga": test_jga,
                    "coref_jga": coref_jga,
                    # f"{aug} cJGA": inv_result["report"]["jga_new_conditional"],
                    **inv_result["report"],
                }
                if "all_ne/hallucination_perturbed" in result:
                    result["NoHF Swap"] = 1 - result["all_ne/hallucination_perturbed"]
                    result["NoHF Orig"] = 1 - result["all_ne/hallucination_original"]
                if "fs_True" in sd or "fewshot_True" in sd:
                    fewshot_results.append(result)
                else:
                    fullshot_results.append(result)

        # break

    return pd.DataFrame(fullshot_results), pd.DataFrame(fewshot_results)


def melt_and_format_target_df(df, custom_target_metrics=[], get_sum=False):

    for idx, row in df.iterrows():
        for aug in ["NEI", "SD", "TP"]:
            cjga_val = row['jga_new_conditional'] if row['inv'] == aug else None
            jga_val = row['jga_perturbed'] if row['inv'] == aug else None
            df.at[idx, f'{aug} cJGA'] = cjga_val
            df.at[idx, f'{aug} JGA'] = jga_val

        hall_swap_val = row['NoHF Swap'] if row['inv'] == "NEI" else None
        hall_orig_val = row['NoHF Orig'] if row['inv'] == "NEI" else None
        df.at[idx, f'NoHF Swap'] = hall_swap_val
        df.at[idx, f'NoHF Orig'] = hall_orig_val
    # target = df[df["test_jga"]> 0.1]

    ### brittle code that only works with equal number of rows for all invariances
    # target = df[df["inv"]=="TP"][df["test_jga"]> 0.1]
    # target["TP cJGA"] = target["jga_new_conditional"]
    # target["SD cJGA"] = df[df["inv"]=="SD"][df["test_jga"]> 0.1]["jga_new_conditional"].tolist()
    # # target[target["inv"]!="NEI"]["all_ne/hallucination_perturbed"] = None
    # target["NEI cJGA"] = df[df["inv"]=="NEI"][df["test_jga"]> 0.1]["jga_new_conditional"].tolist()

    # target["all_ne/hallucination_perturbed"] = df[df["inv"]=="NEI"][df["test_jga"]> 0.1]["all_ne/hallucination_perturbed"].tolist()

    if not custom_target_metrics:
        # target_metrics = [
        #     "test_jga",
        #     # "valid_jga",
        #     "coref_jga",
        #     "NEI cJGA",
        #     "NEI JGA",
        #     "SD cJGA",
        #     "SD JGA",
        #     "TP cJGA",
        #     "TP JGA",
        #     "NoHF Orig",
        #     "NoHF Swap",
        # ]
        target_metrics = TARGET_METRICS
    else:
        target_metrics = custom_target_metrics

    sum_metrics = [
        # "test_jga",
        # "coref_jga",
        "NEI cJGA",
        # "NEI JGA",
        "SD cJGA",
        # "SD JGA",
        "TP cJGA",
        # "TP JGA",
        # "NoHF Orig",
        # "NoHF Swap",
    ]

    sum_metrics2 = [
        # "test_jga",
        # "coref_jga",
        "NEI JGA",
        "SD JGA",
        "TP JGA",
        # "NoHF Orig",
        # "NoHF Swap",
    ]

    target = df.melt(["epochs"], target_metrics).drop_duplicates()

    # import pdb; pdb.set_trace()
    # if not get_sum and target.iloc[0]['value'] < 1:
    #     for tm in target_metrics:
    #         df[tm] = df[tm].apply(lambda x:x*100)

    if get_sum:
        sum1 = (
            target[target['variable'].isin(sum_metrics)]
            .groupby("epochs")['value']
            .agg('sum')
        )
        sum2 = (
            target[target['variable'].isin(sum_metrics2)]
            .groupby("epochs")['value']
            .agg('sum')
        )
        custom_target_metrics = ['sum cJGA', 'sum JGA']
        temp = (
            pd.concat([sum1, sum2], axis=1)
            .set_axis(custom_target_metrics, axis=1)
            .reset_index()
        )

        ep_max_sum_cJGA = temp['sum cJGA'].idxmax()
        ep_max_sum_JGA = temp['sum JGA'].idxmax()

        # if ep_max_sum_cJGA != ep_max_sum_JGA:
        print(ep_max_sum_cJGA, ep_max_sum_JGA)

        return temp.melt(["epochs"], custom_target_metrics)

    else:
        return target


def plot_cjga_trends(df, no_band=True, title="", log_scale=False):

    sns.set_theme()
    ci = None if no_band else "sd"
    ci = None if no_band else 95

    name_mapping = {
        "test_jga": "JGA",
        "coref_jga": "Coref JGA",
        "NEI cJGA": "NED cJGA",
        "SD cJGA": "SDI cJGA",
        "TP cJGA": "PI cJGA",
    }
    rename_columns = {"variable": "CheckDST"}

    df['%'] = df['value'].apply(lambda x: x * 100 if x < 1 else x)

    for idx, row in df.iterrows():
        df.at[idx, 'variable'] = name_mapping.get(row['variable'], row['variable'])
    df.rename(rename_columns, axis=1, inplace=True)
    # sns.axes_style("white")
    # sns.set_style("white")
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    # sns.set_theme("white")

    rel = sns.relplot(
        data=df,
        kind="line",
        x="epochs",
        y="%",
        hue=rename_columns.get("variable", "variable"),
        # style=rename_columns.get("variable", "variable"),
        estimator=np.median,
        ci=ci,
        aspect=1.2,
        linewidth=7,
    )
    # plt.legend([],[], frameon=False)

    if log_scale:
        rel.ax.set(xscale="log")
        # f = lambda x:
        # rel.ax.set(xscale="function", functions=())
        # rel.ax.set_xlim(-1,10)
        # rel.ax.set_xticks([0.02*i for i in range(0,50)]+list(range(1,11)))

    rel._legend.remove()
    # rel.legend(fontsize=5)
    # leg = rel.ax.legend()
    leg = rel._legend
    for line in leg.get_lines():
        line.set_linewidth(8.0)

    rel.fig.suptitle(title)


def example_data():
    # The following data is from the Denver Aerosol Sources and Health study.
    # See doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO)
    #  3)Inclusion of gas-phase specie ozone (O3).
    #  4)Inclusion of both gas-phase species is present...
    data = [
        TARGET_METRICS,
        (
            'Basecase',
            [
                [0.88, 0.01, 0.03, 0.03, 0.00, 0.06],
                [0.01, 0.01, 0.02, 0.71, 0.74, 0.70],
            ],
        ),
    ]
    return data


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path_mpl(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type='circle',
                    path=Path_mpl.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


print("hi")

# def plot_checkdst_radarchart(df, no_band=True, title="", log_scale=False):
def plot_checkdst_radarchart():

    N = 6
    theta = radar_factory(N, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(
        figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar')
    )
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip([axs], data):
        print(case_data)
        ax.set_rgrids([0.25, 0.5, 0.75, 1])
        # ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
        #              horizontalalignment='center', verticalalignment='center'))

        for d, color in zip(case_data, colors):
            d = np.array(d)
            ax.plot(theta, d, color=color)
            ax.errorbar(theta, [0.05] * len(d), yerr="error", fmt='o')
            # ax.fill(theta,  d + 0.05, color=color, alpha=0.25)
            # ax.fill(theta,  d - 0.05, color="white", alpha=0.25)
            # ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    labels = ('TripPy', 'BART-DST')
    legend = axs.legend(labels, loc=(0.9, 0.95), labelspacing=0.1, fontsize='small')

    # fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')

    plt.show()
