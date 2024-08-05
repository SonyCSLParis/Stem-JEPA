"""Summarize results for a model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from evar.common import RESULT_DIR
import fire


def get_weight(weight_file):
    weight_file = Path(weight_file)
    weight = weight_file.parent.name + '/' + weight_file.stem
    return weight


def available_tasks(df):
    ALL_TASKS = ['esc50', 'us8k', 'spcv2', 'vc1', 'voxforge', 'cremad', 'gtzan', 'gtzan_stems', 'nsynth', 'surge']
    tasks = [t for t in ALL_TASKS if t in list(df.columns)]
    return tasks


def summarize(weight_file, tmpfile=None):
    df = pd.read_csv(RESULT_DIR / "scores.csv")
    print(df)
    df = df[df.report.str.contains(weight_file)]
    weight_file = get_weight(weight_file)
    df['weight'] = weight_file
    src_df = df.copy()

    # summarize
    df = pd.pivot_table(df, index=['weight', 'runtime_id'], columns=['task'], values=['score'], aggfunc=np.mean)
    df.columns = df.columns.get_level_values(1)

    # df = df[available_tasks(df)]
    if len(df) == 0:
        print(f'No data for {weight_file}.')
        return

    if len(df) >= 2:
        best_scores = df.max(axis=0).to_frame().T
        best_scores["weight"] = weight_file
        best_scores["runtime_id"] = "best"
        best_scores.set_index(["weight", "runtime_id"], inplace=True)

        df = pd.concat([df, best_scores])

    df['average'] = df.mean(1)

    # Group by weight, run_id, and task, then calculate the mean score
    # df_grouped = df.groupby(['weight', 'runtime_id', 'task']).score.mean().reset_index()
    # print(df_grouped)
    #
    # # Check for Empty DataFrame
    # if df_grouped.empty:
    #     print(f'No data for {weight_file}.')
    # else:
    #     # Optionally, calculate the average score across all tasks for each (weight, run_id)
    #     df_grouped['average'] = df_grouped.groupby(['weight', 'runtime_id']).score.transform('mean')
    #     print(df_grouped)

    # report
    report = df.map(lambda x: f'{x*100:.2f}%' if str(x).isnumeric else x).map(lambda x: '-' if x == "nan%" else x)
    print(report.to_markdown())

    # save source results to a csv.
    report_csv = RESULT_DIR / (str(df.index[0]).replace('/', '_') + '.csv')
    src_df.report = src_df.report.str.replace('\n', ' ')

    print("Saving results in", report_csv)
    src_df.to_csv(report_csv, index=None)

    # write downstream accuracies in temporary csv
    accuracies = src_df[["task", "score"]]
    print(accuracies)
    accuracies.to_csv(tmpfile, index=None)


if __name__ == '__main__':
    fire.Fire(summarize)
