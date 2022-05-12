import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import logging
import sys

from sklearn.metrics import mean_absolute_error
import seaborn

from .common import INSOLES_FILE_NAME, LABELS_FILE_NAME

seaborn.set()
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.rc('text', usetex=True)


TARGET_FRAME_RATE = 60

STEP_FORCE_THRESHOLD = 0.2
LIFT_FORCE_THRESHOLD = 0.17


def get_step_indices_old(max_force, step_threshold=0.25, change_delta=0.05, min_duration_in_seconds=0.25):
    step_indices = [-1]
    lift_indices = [-1]
    last_change = -sys.maxsize
    min_duration = min_duration_in_seconds * TARGET_FRAME_RATE
    for idx, force in enumerate(max_force):
        if idx == 0:
            if force >= step_threshold:
                step_indices.append(idx)
            else:
                lift_indices.append(idx)
        elif force >= step_threshold + change_delta and lift_indices[-1] > step_indices[
            -1] and idx - last_change >= min_duration:
            step_indices.append(idx)
            last_change = idx
        elif force < step_threshold - change_delta and lift_indices[-1] < step_indices[
            -1] and idx - last_change >= min_duration:
            lift_indices.append(idx)
            last_change = idx
    step_indices = step_indices[1:]
    lift_indices = lift_indices[1:]
    return step_indices, lift_indices


def get_step_indices(max_force, step_threshold, lift_threshold, min_duration_in_seconds=0.2):
    step_indices = [-1]
    lift_indices = [-1]
    last_change = -sys.maxsize
    min_duration = min_duration_in_seconds * TARGET_FRAME_RATE
    for idx, force in enumerate(max_force):
        if idx == 0:
            if force >= step_threshold:
                step_indices.append(idx)
            else:
                lift_indices.append(idx)
        elif force >= step_threshold and lift_indices[-1] > step_indices[
            -1] and idx - last_change >= min_duration:
            step_indices.append(idx)
            last_change = idx
        elif force <= lift_threshold and lift_indices[-1] < step_indices[
            -1] and idx - last_change >= min_duration:
            lift_indices.append(idx)
            last_change = idx
    step_indices = step_indices[1:]
    lift_indices = lift_indices[1:]
    return step_indices, lift_indices


def get_step_lift_durations(data_frame, step_indices, lift_indices):
    assert abs(len(step_indices) - len(lift_indices)) <= 1
    if step_indices[0] < lift_indices[0]:
        if len(lift_indices) == len(step_indices):
            step_duration = np.array(lift_indices) - np.array(step_indices)
            lift_duration = np.array(step_indices[1:] + [len(data_frame.index)]) - np.array(lift_indices)
        elif len(lift_indices) < len(step_indices):
            step_duration = np.array(lift_indices + [len(data_frame.index)]) - np.array(step_indices)
            lift_duration = np.array(step_indices[1:]) - np.array(lift_indices)
        else:
            assert False
    else:
        if len(lift_indices) == len(step_indices):
            step_duration = np.array(lift_indices[1:] + [len(data_frame.index)]) - np.array(step_indices)
            lift_duration = np.array(step_indices) - np.array(lift_indices)
        elif len(step_indices) < len(lift_indices):
            step_duration = np.array(lift_indices[1:]) - np.array(step_indices)
            lift_duration = np.array(step_indices + [len(data_frame.index)]) - np.array(lift_indices)
        else:
            assert False
    return step_duration, lift_duration


def get_step_on_ground_column(step_indices, lift_indices, step_durations, lift_durations):
    step_on_ground_left = []
    if step_indices[0] < lift_indices[0]:
        for idx in range(max(len(step_durations), len(lift_durations))):
            if idx < len(step_durations):
                step_on_ground_left += [True] * step_durations[idx]
            if idx < len(lift_durations):
                step_on_ground_left += [False] * lift_durations[idx]
    else:
        for idx in range(max(len(step_durations), len(lift_durations))):
            if idx < len(lift_durations):
                step_on_ground_left += [False] * lift_durations[idx]
            if idx < len(step_durations):
                step_on_ground_left += [True] * step_durations[idx]
    return step_on_ground_left


def get_time_until_event_column(df, event_column):
    result = []
    indices = df.index[df[event_column] == True]
    last_index = -1
    for index in indices:
        event_time = df.iloc[index]['time']
        candidate_times = df[(df.index <= index) & (df.index > last_index)]['time'].to_numpy()
        result += (event_time - candidate_times).tolist()
        last_index = index
    result += [-1] * (df.index[-1] - indices[-1])
    return result


def step_detection(data_frame, step_threshold, lift_threshold, old=False):
    if old:
        step_indices_left, lift_indices_left = get_step_indices_old(data_frame['Left_Max_Force'].to_numpy())
        step_indices_right, lift_indices_right = get_step_indices_old(data_frame['Right_Max_Force'].to_numpy())
    else:
        step_indices_left, lift_indices_left = get_step_indices(data_frame['Left_Max_Force'].to_numpy(), step_threshold, lift_threshold)
        step_indices_right, lift_indices_right = get_step_indices(data_frame['Right_Max_Force'].to_numpy(), step_threshold, lift_threshold)

    tmp = np.array([False] * len(data_frame.index), dtype=bool)
    tmp[step_indices_right] = True
    data_frame['insoles_RightFoot_is_step'] = tmp
    tmp = np.array([False] * len(data_frame.index), dtype=bool)
    tmp[step_indices_left] = True
    data_frame['insoles_LeftFoot_is_step'] = tmp

    tmp = np.array([False] * len(data_frame.index), dtype=bool)
    tmp[lift_indices_right] = True
    data_frame['insoles_RightFoot_is_lift'] = tmp
    tmp = np.array([False] * len(data_frame.index), dtype=bool)
    tmp[lift_indices_left] = True
    data_frame['insoles_LeftFoot_is_lifted'] = tmp

    step_duration_left, lift_duration_left = get_step_lift_durations(data_frame, step_indices_left,
                                                                                        lift_indices_left)
    step_duration_right, lift_duration_right = get_step_lift_durations(data_frame,
                                                                                          step_indices_right,
                                                                                          lift_indices_right)

    step_on_ground_left = []
    if step_indices_left[0] < lift_indices_left[0]:
        for idx in range(max(len(step_duration_left), len(lift_duration_left))):
            if idx < len(step_duration_left):
                step_on_ground_left += [True] * step_duration_left[idx]
            if idx < len(lift_duration_left):
                step_on_ground_left += [False] * lift_duration_left[idx]
    else:
        for idx in range(max(len(step_duration_left), len(lift_duration_left))):
            if idx < len(lift_duration_left):
                step_on_ground_left += [False] * lift_duration_left[idx]
            if idx < len(step_duration_left):
                step_on_ground_left += [True] * step_duration_left[idx]
    data_frame['insoles_LeftFoot_on_ground'] = get_step_on_ground_column(step_indices_left, lift_indices_left,
                                                                             step_duration_left, lift_duration_left)
    data_frame['insoles_RightFoot_on_ground'] = get_step_on_ground_column(step_indices_right, lift_indices_right,
                                                                             step_duration_right, lift_duration_right)

    data_frame['insoles_LeftFoot_time_to_step'] = np.array(
        get_time_until_event_column(data_frame, 'insoles_LeftFoot_is_step')).astype(np.float32)
    data_frame['insoles_LeftFoot_time_to_lift'] = np.array(
        get_time_until_event_column(data_frame, 'insoles_LeftFoot_is_lifted')).astype(np.float32)
    data_frame['insoles_RightFoot_time_to_step'] = np.array(
        get_time_until_event_column(data_frame, 'insoles_RightFoot_is_step')).astype(np.float32)
    data_frame['insoles_RightFoot_time_to_lift'] = np.array(
        get_time_until_event_column(data_frame, 'insoles_RightFoot_is_lifted')).astype(np.float32)
    logging.info("stream summary:\n%s",
                  get_stats_summary(data_frame, TARGET_FRAME_RATE, step_duration_left, lift_duration_left,
                                         step_duration_right, lift_duration_right))
    return data_frame


def get_stats_summary(data_frame, sample_rate, step_durations_left, lift_durations_left, step_durations_right, lift_durations_right):
    n_right_steps = len(data_frame[data_frame['insoles_RightFoot_is_step']].index)
    n_left_steps = len(data_frame[data_frame['insoles_LeftFoot_is_step']].index)
    step_differnce = abs(n_left_steps - n_right_steps)
    duration = (data_frame.iloc[-1]['time'] - data_frame.iloc[0]['time']) / 1000.
    result = "duration %.2f\n" % duration
    result += "%d right_steps %d left_steps %d total steps %.2f steps/min\n" % (n_right_steps, n_left_steps, n_right_steps + n_left_steps, (n_right_steps + n_left_steps)/(duration/60))
    if step_differnce > 1:
        result += "STEP DIFFERENCE %d !!!\n" % step_differnce
    step_durations_in_seconds_left = step_durations_left / sample_rate
    lift_durations_in_seconds_left = lift_durations_left / sample_rate
    step_durations_in_seconds_right = step_durations_right / sample_rate
    lift_durations_in_seconds_right = lift_durations_right / sample_rate

    result += "L step duration %.2fs+-%.2f med %.2fs lift duration %.2fs+-%.2f med %.2fs \n" % (np.mean(step_durations_in_seconds_left), np.std(step_durations_in_seconds_left), np.median(step_durations_in_seconds_left), np.mean(lift_durations_in_seconds_left), np.std(lift_durations_in_seconds_left), np.mean(lift_durations_in_seconds_left))
    result += "R step duration %.2fs+-%.2f med %.2fs lift duration %.2fs+-%.2f med %.2fs \n" % (np.mean(step_durations_in_seconds_right), np.std(step_durations_in_seconds_right), np.median(step_durations_in_seconds_right), np.mean(lift_durations_in_seconds_right), np.std(lift_durations_in_seconds_right), np.median(lift_durations_in_seconds_right))
    return result


def visualize_detection(data_frame):
    max_index = 750
    data_frame = data_frame[data_frame.index.isin(range(0, max_index, 1))]
    '''plt.figure()
    ax = data_frame['Right_Max_Force'].plot.hist(bins=30, alpha=0.5)
    ax.set_title("right")'''

    '''plt.figure()
    ax = data_frame['Left_Max_Force'].plot.hist(bins=30, alpha=0.5)
    ax.set_title("left")'''

    plt.plot(data_frame['time'], data_frame['Left_Max_Force'])

    #ax = data_frame.plot(x='time', y='Left_Max_Force')
    '''ax.scatter(data_frame[data_frame['insoles_LeftFoot_is_step_gt']].index, data_frame[data_frame['insoles_LeftFoot_is_step_gt']]['Left_Max_Force'],
               label="gt_step", marker='x', c='r')
    ax.scatter(data_frame[data_frame['insoles_LeftFoot_is_lifted_gt']].index, data_frame[data_frame['insoles_LeftFoot_is_lifted_gt']]['Left_Max_Force'],
               label="gt_lift", marker='x', c='g')'''
    plt.scatter(data_frame[data_frame['insoles_LeftFoot_is_step']].time, data_frame[data_frame['insoles_LeftFoot_is_step']]['Left_Max_Force'],
               label="step", c=seaborn.color_palette(n_colors=8)[1])
    plt.scatter(data_frame[data_frame['insoles_LeftFoot_is_lifted']].time, data_frame[data_frame['insoles_LeftFoot_is_lifted']]['Left_Max_Force'],
               label="lift", c=seaborn.color_palette(n_colors=8)[2])
    plt.hlines(STEP_FORCE_THRESHOLD, 0, int(max_index*16.66), label=r"$\alpha_{step}$", color=seaborn.color_palette(n_colors=8)[1], linestyle='--')
    plt.hlines(LIFT_FORCE_THRESHOLD, 0, int(max_index*16.66), label=r"$\alpha_{lift}$", color=seaborn.color_palette(n_colors=8)[2], linestyle='--')
    plt.title("Left Foot Step Detection")
    plt.ylabel("max sensor value")
    plt.xlabel("time (ms)")
    plt.legend(fancybox=True, loc='upper right')
    plt.savefig('/hri/storage/user/vlosing/step_detection.pdf', bbox_inches='tight')
    '''plt.figure()
    ax = data_frame['Right_Max_Force'].plot()
    ax.scatter(data_frame[data_frame['insoles_RightFoot_is_step']].index, data_frame[data_frame['insoles_RightFoot_is_step']]['Right_Max_Force'],
               label="step", c='r')
    ax.scatter(data_frame[data_frame['insoles_RightFoot_is_lifted']].index, data_frame[data_frame['insoles_RightFoot_is_lifted']]['Right_Max_Force'],
               label="lift", c='g')
    ax.set_title("right")
    plt.legend()'''

    '''plt.figure()
    ax = data_frame['Right_Max_Force'].plot(label='right')
    ax = data_frame['Left_Max_Force'].plot(label='left')
    ax.scatter(data_frame[data_frame['insoles_RightFoot_is_step']].index, data_frame[data_frame['insoles_RightFoot_is_step']]['Right_Max_Force'],
               label="right_step")
    ax.scatter(data_frame[data_frame['insoles_RightFoot_is_lifted']].index, data_frame[data_frame['insoles_RightFoot_is_lifted']]['Right_Max_Force'],
               label="right_lift")
    ax.scatter(data_frame[data_frame['insoles_LeftFoot_is_step']].index, data_frame[data_frame['insoles_LeftFoot_is_step']]['Left_Max_Force'],
               label="left_step")
    ax.scatter(data_frame[data_frame['insoles_LeftFoot_is_lifteded']].index, data_frame[data_frame['insoles_LeftFoot_is_lifteded']]['Left_Max_Force'],
               label="left_lift")
    ax.set_title("both")
    plt.legend()'''
    plt.show()


def mae(gt_times, est_times):
    matched_times = []
    matched_idx = 0
    for gt_time in gt_times:
        min_dif = float("inf")
        while matched_idx < len(est_times) and abs(gt_time - est_times[matched_idx]) < min_dif:
            min_dif = abs(gt_time - est_times[matched_idx])
            last_matched_idx = matched_idx
            matched_idx += 1
        matched_times.append(est_times[last_matched_idx])
    return mean_absolute_error(gt_times, matched_times)


def get_mae_step_lift_times(data_frames):
    mae_steps = []
    mae_lifts = []
    for data_frame in data_frames:
        gt_step_time = data_frame[data_frame['insoles_LeftFoot_is_step_gt']]["time"].tolist()
        pred_step_time = data_frame[data_frame['insoles_LeftFoot_is_step']]["time"].tolist()
        mae_steps.append(mae(gt_step_time, pred_step_time))

        pred_lift_time = data_frame[data_frame['insoles_LeftFoot_is_lifted']]["time"].tolist()
        gt_lift_time = data_frame[data_frame['insoles_LeftFoot_is_lifted_gt']]["time"].tolist()
        mae_lifts.append(mae(gt_lift_time, pred_lift_time))
    return mae_steps, mae_lifts


def prepare_data_frame(data_frame):
    gt_is_step = []
    last_item = None
    for item in data_frame['walk_mode']:
        if item != last_item and item == "DOWN":
            gt_is_step.append(True)
        else:
            gt_is_step.append(False)
        last_item = item

    gt_is_lift = []

    last_item = "None"
    for item in data_frame['walk_mode']:
        if item != last_item and item == "UP":
            gt_is_lift.append(True)
        else:
            gt_is_lift.append(False)
        last_item = item
    data_frame['insoles_LeftFoot_is_step_gt'] = gt_is_step
    data_frame['insoles_LeftFoot_is_lifted_gt'] = gt_is_lift
    return data_frame


def minimize_step_diff(data_frames):
    thresholds = np.linspace(0.05, 1, num=20)
    errors = []
    for i, s in enumerate(thresholds):
        for l in thresholds[:i]:
            step_diffs = []
            for data_frame in data_frames:
                data_frame = step_detection(data_frame, s, l)
                data_frame = prepare_data_frame(data_frame)
                n_right_steps = len(data_frame[data_frame['insoles_RightFoot_is_step']].index)
                n_left_steps = len(data_frame[data_frame['insoles_LeftFoot_is_step']].index)
                step_diffs.append(abs(n_left_steps - n_right_steps))
            errors.append([np.mean(step_diffs), s, l])
    errors.sort(key=lambda x: x[0])
    print(errors[:10])

def minimize_mae(data_frames):
    thresholds = np.linspace(0.05, 1, num=95)
    errors = []
    for i, s in enumerate(thresholds):
        for l in thresholds[:i]:
            frames = []
            for data_frame in data_frames:
                data_frame = step_detection(data_frame, s, l)
                data_frame = prepare_data_frame(data_frame)
                frames.append(data_frame)
            mae_steps, mae_lifts = get_mae_step_lift_times(frames)

            mae_step = np.mean(mae_steps)
            mae_lift = np.mean(mae_lifts)
            errors.append([mae_step + mae_lift, mae_step, mae_lift, s, l])
    errors.sort(key=lambda x: x[0])
    print(errors[:10])


def clip_pressure_columns(data_frame):
    columns = [
        'Left_Hallux', 'Left_Toes', 'Left_Met1', 'Left_Met3', 'Left_Met5', 'Left_Arch', 'Left_Heel_R', 'Left_Heel_L',
        'Right_Hallux', 'Right_Toes', 'Right_Met1', 'Right_Met3', 'Right_Met5', 'Right_Arch', 'Right_Heel_L', 'Right_Heel_R',
        'Left_Max_Force', 'Right_Max_Force']
    for col in columns:
        data_frame[col] = np.clip(data_frame[col], 0, 1)
    return data_frame


def reprocess_soles_data_for_recording(path):
    labels_path = os.path.join(path, LABELS_FILE_NAME)
    insoles_path = os.path.join(path, INSOLES_FILE_NAME)
    insoles_data_frame = pd.read_csv(insoles_path)
    insoles_data_frame = clip_pressure_columns(insoles_data_frame)
    labels_data_frame = pd.read_csv(labels_path)
    columns = [column for column in labels_data_frame.columns if "Unnamed" in column]
    labels_data_frame.drop(columns=columns, inplace=True)
    data_frame = pd.merge(labels_data_frame, insoles_data_frame, how='outer', on=["time", "participant_id", "task"], validate="one_to_one")

    print("old")
    step_detection(data_frame, STEP_FORCE_THRESHOLD, LIFT_FORCE_THRESHOLD, old=True)
    n_right_steps = len(data_frame[data_frame['insoles_RightFoot_is_step']].index)
    n_left_steps = len(data_frame[data_frame['insoles_LeftFoot_is_step']].index)
    step_differnce_old = abs(n_left_steps - n_right_steps)

    print("new")
    data_frame = step_detection(data_frame, STEP_FORCE_THRESHOLD, LIFT_FORCE_THRESHOLD, old=False)
    n_right_steps = len(data_frame[data_frame['insoles_RightFoot_is_step']].index)
    n_left_steps = len(data_frame[data_frame['insoles_LeftFoot_is_step']].index)
    step_differnce_new = abs(n_left_steps - n_right_steps)
    insoles_data_frame = data_frame[insoles_data_frame.columns]
    insoles_data_frame.to_csv(insoles_path, index=False)
    labels_data_frame = data_frame[labels_data_frame.columns]
    labels_data_frame.to_csv(labels_path, index=False)
    return step_differnce_old, step_differnce_new


def optimize_thresholds_for_paths(paths):
    frames = []
    for subject_path in paths:
        insoles_data_frame = pd.read_csv(os.path.join(subject_path, INSOLES_FILE_NAME))
        insoles_data_frame = clip_pressure_columns(insoles_data_frame)
        labels_data_frame = pd.read_csv(os.path.join(subject_path, "labels_steps.csv"))
        columns = [column for column in labels_data_frame.columns if "Unnamed" in column]
        labels_data_frame.drop(columns=columns, inplace=True)
        data_frame = pd.merge(labels_data_frame, insoles_data_frame, how='outer', on=["time"], validate="one_to_one")

        #indices = data_frame[data_frame["walk_mode"] == "UP"].index
        #data_frame.drop(data_frame[(data_frame.index < min(indices) - 10) | (data_frame.index > max(indices) + 10)].index, inplace=True)
        #data_frame.drop(data_frame[data_frame.index > 1995].index, inplace=True)

        data_frame = prepare_data_frame(data_frame)
        data_frame = step_detection(data_frame, STEP_FORCE_THRESHOLD, LIFT_FORCE_THRESHOLD, old=False)
        visualize_detection(data_frame)
        #data_frame = prepare_data_frame(data_frame)
        visualize_detection(data_frame)
        frames.append(data_frame)

    mae_steps, mae_lifts = get_mae_step_lift_times(frames)
    print("mae_step %.4f +- %.4f mae_lift %.4f +- %.4f" % (np.mean(mae_steps), np.std(mae_steps), np.mean(mae_lifts), np.std(mae_lifts)))
    #minimize_step_diff(frames)
    #minimize_mae(frames)


def reprocess_all_soles_data():
    prefix = "/hri/rawstreams/user/RT-PHUME/NEWBEE_dataset/data_set/"
    paths = [os.path.join(prefix, "courseA"), os.path.join(prefix, "courseB"), os.path.join(prefix, "courseC")]
    step_diffs_old = []
    step_diffs_new = []
    for path in paths:
        for root, dir_list, file_list in os.walk(path):
            for dir in dir_list:
                print(root, dir)
                step_diff_old, step_diff_new = reprocess_soles_data_for_recording(os.path.join(root, dir))
                step_diffs_old.append(step_diff_old)
                step_diffs_new.append(step_diff_new)
    print(np.mean(step_diffs_old), np.std(step_diffs_old), np.mean(step_diffs_new), np.std(step_diffs_new))


def optimize_thresholds():
    prefix = "/hri/rawstreams/user/RT-PHUME/NEWBEE_dataset_tmp/data_set/"
    labeled_paths = [
        os.path.join(prefix, "courseA/id01/"),
        os.path.join(prefix, "courseA/id05/"),
        os.path.join(prefix, "courseB/id05/"),
        os.path.join(prefix, "courseC/id13/"),
        os.path.join(prefix, "courseC/id25/"),
        ]
    optimize_thresholds_for_paths(labeled_paths)


logging.basicConfig(format='%(message)s', level=logging.DEBUG)
#reprocess_all_soles_data()
#optimize_thresholds()

