import scipy, glob, os
from scipy.stats import linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from scipy.io.matlab.mio5_params import mat_struct
import seaborn as sns
from sklearn.model_selection import train_test_split

# load a mat file and return the behavior dataframe
def load_behavior_data(file_path):
    """
    Load behavior data from a .mat file and return a pandas DataFrame
    with standardized reward levels and session numbers.

    Parameters:
    file_path (str): Path to the .mat file.

    Returns:
    pd.DataFrame: DataFrame containing the behavior data.
    """

    data = loadmat(file_path)
    beh = data['A'][0, 0]

    fieldnames = ['catch', 'reward', 'reward_delay', 'block', 'hits', 'vios', 'optout',
                  'wait_time', 'trial_num', 'ITI', 'BlockNum', 'BlockPosition']
    beh_df = pd.DataFrame({field: beh[field].flatten() for field in fieldnames})

    # map reward values to standard set
    original_rewards = np.sort(np.unique(beh_df['reward']))
    standard_rewards = [5, 10, 20, 40, 80][:len(original_rewards)]

    reward_mapping = {orig: std for orig, std in zip(original_rewards, standard_rewards)}
    beh_df['reward'] = beh_df['reward'].map(reward_mapping)

    # add session number based on trial_num resets
    restarts = np.where(beh_df['trial_num'] == 1)[0]
    num_sessions = len(restarts)
    beh_df['session_num'] = 0
    for i in range(num_sessions):
        start = restarts[i]
        end = restarts[i + 1] if (i < num_sessions - 1) else len(beh_df)
        beh_df.loc[start:end - 1, 'session_num'] = i + 1

    return beh_df

# plot wait times by block and reward using combined data across sessions
def plot_wait_times_aggregated(beh_df, block_labels={1: 'Mixed', 2: 'High', 3: 'Low'},
                               block_colors={1: 'black', 2: 'red', 3: 'blue'}):
    """
    Plot mean wait times by block and reward using per-rat aggregated data.

    Error bars:
        - Light shaded area = standard deviation across rats
        - Dark shaded area = standard error across rats = std / sqrt(n - 1)
    """
    unique_blocks = np.sort(beh_df['block'].unique())
    unique_rewards = np.sort(beh_df['reward'].unique())

    # initialize result arrays
    wait_times_mean = np.zeros((len(unique_blocks), len(unique_rewards)))
    wait_times_std = np.zeros((len(unique_blocks), len(unique_rewards)))
    wait_times_se = np.zeros((len(unique_blocks), len(unique_rewards)))

    for i, block in enumerate(unique_blocks):
        for j, reward in enumerate(unique_rewards):
            per_rat_means = (
                beh_df[(beh_df['optout'] == 1) &
                       (beh_df['block'] == block) &
                       (beh_df['reward'] == reward)]
                .groupby("rat")["wait_time"].mean()
            )

            n = len(per_rat_means)
            if n > 1:
                std = per_rat_means.std(ddof=1)
                se = std / np.sqrt(n - 1)
            else:
                std = np.nan
                se = np.nan

            wait_times_mean[i, j] = per_rat_means.mean()
            wait_times_std[i, j] = std
            wait_times_se[i, j] = se

    # plotting
    plt.figure(figsize=(4, 4))
    for i, block in enumerate(unique_blocks):
        color = block_colors.get(block, 'gray')
        label = block_labels.get(block, f'Block {block}')
        mean = wait_times_mean[i, :]
        std = wait_times_std[i, :]
        se = wait_times_se[i, :]

        # plot standard deviation (wider shaded area)
        plt.fill_between(unique_rewards, mean - std, mean + std, color=color, alpha=0.2)

        # plot standard error (narrow shaded area)
        plt.fill_between(unique_rewards, mean - se, mean + se, color=color, alpha=0.4)

        # plot mean
        plt.plot(unique_rewards, mean, 'o-', label=label, color=color)

    plt.xlabel('Reward Amount')
    plt.ylabel('Mean Wait Time (seconds)')
    plt.xscale('log')
    plt.xticks(unique_rewards, [f'{int(r)} uL' for r in unique_rewards])
    plt.legend()
    plt.grid(True)

    return wait_times_mean, wait_times_se

# plot wait times by block and reward using combined data across sessions and take a figure and axes as input
def plot_aggregated_wait_times_on_ax(beh_df, ax, block_labels={1: 'Mixed', 2: 'High', 3: 'Low'},
                              block_colors={1: 'black', 2: 'red', 3: 'blue'}):
    unique_blocks = np.sort(beh_df['block'].unique())
    unique_rewards = np.sort(beh_df['reward'].unique())

    # create a matrix to hold mean wait times for each block and reward combination
    wait_times = np.zeros((len(unique_blocks), len(unique_rewards)))
    wait_times_std = np.zeros((len(unique_blocks), len(unique_rewards)))

    for i, block in enumerate(unique_blocks):
        for j, reward in enumerate(unique_rewards):
            wait_times[i, j] = beh_df['wait_time'][
                (beh_df['optout'] == 1) & (beh_df['block'] == block) & (beh_df['reward'] == reward)].mean()
            wait_times_std[i, j] = beh_df['wait_time'][
                (beh_df['optout'] == 1) & (beh_df['block'] == block) & (beh_df['reward'] == reward)].sem()

    # plot on the provided axis
    for i, block in enumerate(unique_blocks):
        ax.errorbar(unique_rewards, wait_times[i, :], yerr=wait_times_std[i, :], fmt='o-', label=block_labels[block],
                    color=block_colors[block])

    ax.set_xlabel('Reward Amount')
    ax.set_ylabel('Mean Wait Time (s)')
    ax.set_xscale('log')
    ax.set_xticks(unique_rewards)
    ax.set_xticklabels([f'{int(r)} uL' for r in unique_rewards])
    ax.grid(True)

    return wait_times, wait_times_std

# calculate block sensitivity, as well as slope of wait times for mixed block with data combined across session
def calc_block_sensitivity_and_mixed_slope(beh_df):
    """
    Calculate block sensitivity and slope of wait times for mixed block with data combined across sessions.

    Parameters:
    beh_df (pd.DataFrame): DataFrame containing the behavior data.

    Returns:
    block_sensitivity (float): Difference in wait times between Low and High blocks at 20 µL.
    block_sensitivity_ratio (float): Ratio of wait times between Low and High blocks at 20 µL.
    slope (float): Slope of wait times for mixed block across reward levels.
    """

    unique_blocks = np.sort(beh_df['block'].unique())
    unique_rewards = np.sort(beh_df['reward'].unique())
    wait_times = np.full((len(unique_blocks), len(unique_rewards)), np.nan)

    for i, block in enumerate(unique_blocks):
        for j, reward in enumerate(unique_rewards):
            mask = (beh_df['optout'] == 1) & (beh_df['block'] == block) & (beh_df['reward'] == reward) & (
                        beh_df['vios'] == 0) & (beh_df['wait_time'] <= 30)
            vals = beh_df.loc[mask, 'wait_time']
            if len(vals):
                wait_times[i, j] = np.nanmean(vals)

    # block sensitivity at 20 µL: Low (block==3) minus High (block==2)
    idx20 = np.where(unique_rewards == 20)[0][0]
    low_i = np.where(unique_blocks == 3)[0][0]
    high_i = np.where(unique_blocks == 2)[0][0]
    block_sensitivity = wait_times[low_i, idx20] - wait_times[high_i, idx20]

    # ratio version of block sensitivity: Low divided by High
    block_sensitivity_ratio = wait_times[low_i, idx20] / wait_times[high_i, idx20]

    # slope for Mixed block (block==1)
    mix_i = np.where(unique_blocks == 1)[0][0]
    y = wait_times[mix_i]
    x = np.log(unique_rewards)
    mask = ~np.isnan(y)
    slope, _, _, _, _ = linregress(x[mask], y[mask])

    return block_sensitivity, block_sensitivity_ratio, slope

# plot wait times per session
def plot_wait_times_per_session(beh_df, block_labels={1: 'Mixed', 2: 'High', 3: 'Low'},
                                block_colors={1: 'black', 2: 'red', 3: 'blue'}):
    """
    Plot mean wait times by block and reward using data per sessions.

    Parameters:
    beh_df (pd.DataFrame): DataFrame containing the behavior data.
    block_labels (dict): Dictionary mapping block numbers to labels.
    block_colors (dict): Dictionary mapping block numbers to colors.

    Returns:
    wait_times_per_session (np.ndarray): A 3D array containing mean wait times for each session, block, and reward combination.
    """

    unique_blocks = np.sort(beh_df['block'].unique())
    unique_rewards = np.sort(beh_df['reward'].unique())
    num_sessions = beh_df['session_num'].nunique()
    print(f'Number of sessions: {num_sessions}')

    wait_times_per_session = np.zeros((num_sessions, len(unique_blocks), len(unique_rewards)))

    # loop through each session and calculate mean wait times
    for session in range(num_sessions):
        for i, block in enumerate(unique_blocks):
            for j, reward in enumerate(unique_rewards):
                mask = (
                        (beh_df['session_num'] == session) &
                        (beh_df['block'] == block) &
                        (beh_df['reward'] == reward) &
                        (beh_df['optout'] == 1)
                )
                vals = beh_df.loc[mask, 'wait_time']
                if len(vals) > 0:
                    wait_times_per_session[session, i, j] = vals.mean()

    return wait_times_per_session


# calculate slope of wait times for mixed block per session
def calc_mixed_slope_per_session(beh_df):
    """
    Calculate the slope of wait times for mixed block per session,
    excluding violations and wait times > 30 seconds.

    Returns:
    slopes (np.array): Array of slopes for each session.
    """

    unique_rewards = np.sort(beh_df['reward'].unique())
    num_sessions = beh_df['session_num'].nunique()
    slopes = np.full(num_sessions, np.nan)

    for session in range(num_sessions):
        wait_times_mixed = np.full(len(unique_rewards), np.nan)

        for j, reward in enumerate(unique_rewards):
            mask = (
                (beh_df['session_num'] == session + 1) &
                (beh_df['block'] == 1) &  # Mixed block
                (beh_df['reward'] == reward) &
                (beh_df['optout'] == 1) &
                (beh_df['vios'] == 0) &
                (beh_df['wait_time'] <= 30)
            )
            vals = beh_df.loc[mask, 'wait_time']
            if len(vals) > 0:
                wait_times_mixed[j] = vals.mean()

        valid_mask = ~np.isnan(wait_times_mixed)
        if np.sum(valid_mask) > 1:
            slope, _, _, _, _ = linregress(np.log(unique_rewards[valid_mask]), wait_times_mixed[valid_mask])
            slopes[session] = slope

    return slopes

# calculate block sensitivity (Low - High wait time at 20 uL) and ratio per session
def calc_block_sensitivity_per_session(beh_df):
    """
    Calculate block sensitivity per session (Low - High wait time at 20 uL),
    and the ratio (Low / High), excluding violations and wait times > 30 seconds.

    Returns:
    block_sens (np.ndarray): Difference in wait times per session.
    block_sens_ratio (np.ndarray): Ratio of wait times per session.
    """

    unique_rewards = np.sort(beh_df['reward'].unique())
    if 20 not in unique_rewards:
        n = beh_df['session_num'].nunique()
        return np.full(n, np.nan), np.full(n, np.nan)

    num_sessions = beh_df['session_num'].nunique()
    block_sens = np.full(num_sessions, np.nan)
    block_sens_ratio = np.full(num_sessions, np.nan)

    for session in range(num_sessions):
        session_data = beh_df[
            (beh_df['session_num'] == session + 1) &
            (beh_df['vios'] == 0) &
            (beh_df['wait_time'] <= 30)
        ]

        wait_low = session_data[
            (session_data['block'] == 3) &
            (session_data['reward'] == 20) &
            (session_data['optout'] == 1)
        ]['wait_time'].mean()

        wait_high = session_data[
            (session_data['block'] == 2) &
            (session_data['reward'] == 20) &
            (session_data['optout'] == 1)
        ]['wait_time'].mean()

        if pd.notna(wait_low) and pd.notna(wait_high) and wait_high != 0:
            block_sens[session] = wait_low - wait_high
            block_sens_ratio[session] = wait_low / wait_high

    return block_sens, block_sens_ratio


# get optout probability as a function of wait time
def get_optout_probability(beh_df, wait_time_bins=np.arange(0.5, 10.5, 0.5), ax=None):
    """
    Calculate the probability of going to the optout port and the probability of getting reward as a function of wait time.

    Parameters:
    beh_df (pd.DataFrame): DataFrame containing the behavior data.
    wait_time_bins (np.ndarray): Array of wait time bins.

    Returns:
    optout_prob (np.ndarray): Probability of going to the optout port for each wait time bin.
    reward_prob (np.ndarray): Probability of getting reward for each wait time bin.
    """

    optout_prob = np.zeros(len(wait_time_bins) - 1)  # initialize with nan

    for i in range(len(wait_time_bins) - 1):
        trials_in_bin = beh_df[(beh_df['wait_time'] >= wait_time_bins[i]) &
                               (beh_df['wait_time'] < wait_time_bins[i + 1]) &
                               (beh_df['vios'] == 0) &
                               (beh_df['catch'] == 0)]

        optout_prob[i] = trials_in_bin['optout'].mean() if len(trials_in_bin) > 10 else np.nan

    reward_prob = 1 - optout_prob

    # plotting probabilities
    if ax is None:
        plt.figure(figsize=(4, 3))
        ax = plt.gca()
    ax.plot(wait_time_bins[:-1], optout_prob, marker='o', linestyle='-', color='purple', label='Optout Probability')
    ax.plot(wait_time_bins[:-1], reward_prob, marker='o', linestyle='-', color='green', label='Reward Probability')
    ax.set_xlabel('Wait Time (seconds)')
    ax.set_ylabel('Probability')

    return optout_prob, reward_prob


def get_transition_dynamics(file, max_trials=15):
    beh_df = load_behavior_data(file)
    tname = ['M_to_H', 'M_to_L', 'H_to_M', 'L_to_M']

    # get block switches
    b0 = beh_df['block'].values[:-1]
    b1 = beh_df['block'].values[1:]
    switches = np.where(b0 != b1)[0] + 1
    bprev = beh_df['block'].values[switches - 1]
    bcurr = beh_df['block'].values[switches]
    prev_block = [1, 1, 2, 3]
    curr_block = [2, 3, 1, 1]

    # initialize columns
    for name in tname:
        beh_df[name] = np.nan

    for i, t0 in enumerate(switches):
        curr_session = beh_df['session_num'].values[t0]

        ts = t0
        while ts > 0 and beh_df['session_num'].values[ts - 1] == curr_session:
            ts -= 1
            if beh_df['block'].values[ts] != bprev[i]:
                ts += 1
                break

        te = t0
        while te < len(beh_df) and beh_df['session_num'].values[te] == curr_session:
            if beh_df['block'].values[te] != bcurr[i]:
                break
            te += 1

        for j, (b0, b1) in enumerate(zip(prev_block, curr_block)):
            if (bprev[i] == b0) and (bcurr[i] == b1):
                idx = np.arange(ts, te)
                beh_df.loc[idx, tname[j]] = idx - t0

    # z-score wait times by reward
    unique_rewards = beh_df['reward'].unique()
    beh_df['z_wait_time'] = np.nan

    for reward in unique_rewards:
        trial_indices = beh_df[
            (beh_df['reward'] == reward) &
            (beh_df['optout'] == 1) &
            (~beh_df['wait_time'].isna()) &
            (beh_df['vios'] == 0)
            ].index
        if len(trial_indices) == 0:
            continue
        wait_times = beh_df.loc[trial_indices, 'wait_time']
        m = wait_times.mean()
        s = wait_times.std()
        z_scores = (wait_times - m) / s if s != 0 else np.zeros(len(wait_times))
        beh_df.loc[trial_indices, 'z_wait_time'] = z_scores

    # compute z_wait_times matrix
    ntrials = 2 * max_trials + 1
    nttypes = len(tname)
    z_wait_times = np.empty((ntrials, nttypes))
    z_wait_times.fill(np.nan)

    for i, t in enumerate(tname):
        trials_to_use = beh_df[
            (beh_df['optout'] == 1) &
            (~beh_df['z_wait_time'].isna()) &
            (beh_df['vios'] == 0) &
            (~beh_df[t].isna()) &
            (beh_df[t] >= -max_trials) &
            (beh_df[t] <= max_trials)
            ].index

        distances = beh_df.loc[trials_to_use, t].values.astype(int)
        z_scores = beh_df.loc[trials_to_use, 'z_wait_time'].values

        for d in range(-max_trials, max_trials + 1):
            mask = distances == d
            if np.any(mask):
                z_wait_times[d + max_trials, i] = np.nanmean(z_scores[mask])

    return z_wait_times

def label_transitions(df):
    """
    Add a 'transition_type' column to indicate transitions between blocks.
    Example types: 'M_to_H', 'M_to_L', 'H_to_M', 'L_to_M'
    """
    df = df.copy()
    prev_block = df['block'].shift(1)
    curr_block = df['block']
    transitions = {
        (1, 2): 'M_to_H',
        (1, 3): 'M_to_L',
        (2, 1): 'H_to_M',
        (3, 1): 'L_to_M'
    }
    df['transition_type'] = [transitions.get((p, c), None) for p, c in zip(prev_block, curr_block)]
    df['is_transition'] = df['transition_type'].notna()
    return df

def zscore_wait_times(df):
    """
    Z-score the wait times per reward level (across the full dataset for one rat).
    Returns the original DataFrame with a new column 'z_wait_time'.
    """
    df = df.copy()
    df['z_wait_time'] = np.nan
    for reward in df['reward'].dropna().unique():
        mask = df['reward'] == reward
        vals = df.loc[mask, 'wait_time']
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std > 0:
            df.loc[mask, 'z_wait_time'] = (vals - mean) / std
    return df


def plot_transition_dynamics_per_rat(df, ax=None, max_trials=15):
    """
    Plots z-scored wait time curves aligned to transitions (-15 to +15) for each transition type.
    """
    df = label_transitions(df)
    df = zscore_wait_times(df)

    tnames = ['M_to_H', 'M_to_L', 'H_to_M', 'L_to_M']
    aligned = {t: [] for t in tnames}

    for idx, row in df[df['is_transition']].iterrows():
        ttype = row['transition_type']
        if ttype not in tnames:
            continue
        start = idx - max_trials
        end = idx + max_trials + 1
        if start < 0 or end > len(df):
            continue
        segment = df.iloc[start:end]
        aligned[ttype].append(segment['z_wait_time'].values)

    x_vals = np.arange(-max_trials, max_trials + 1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for t in tnames:
        mat = np.stack([a for a in aligned[t] if len(a) == 2*max_trials + 1], axis=0)
        avg = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        ax.plot(x_vals, avg, label=t)
        ax.fill_between(x_vals, avg - std, avg + std, alpha=0.2)

    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel('Trials from Transition')
    ax.set_ylabel('Z-scored Wait Time')
    ax.set_title('Transition-Aligned Wait Time Dynamics')
    ax.legend()
    return ax

def plot_split_transition_curves(z_matrix, title_prefix=""):
    x = np.arange(-15, 16)
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # Top: High/Low → Mixed
    axs[0].plot(x, z_matrix[:, 2], color='blue', label='H_to_M')  # H → M
    axs[0].plot(x, z_matrix[:, 3], color='violet', label='L_to_M')  # L → M
    axs[0].axvline(0, linestyle='--', color='gray')
    axs[0].set_title(f"{title_prefix}Z-Scored Wait Time: High/Low → Mixed")
    axs[0].set_ylabel("Mean Z-Scored Wait Time")
    axs[0].legend()
    axs[0].grid(True)

    # Bottom: Mixed → High/Low
    axs[1].plot(x, z_matrix[:, 0], color='red', label='M_to_H')  # M → H
    axs[1].plot(x, z_matrix[:, 1], color='green', label='M_to_L')  # M → L
    axs[1].axvline(0, linestyle='--', color='gray')
    axs[1].set_title(f"{title_prefix}Z-Scored Wait Time: Mixed → High/Low")
    axs[1].set_xlabel("Trial Distance from Transition")
    axs[1].set_ylabel("Mean Z-Scored Wait Time")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def zscore_wait_times_by_reward(df):
    """
    Z-score wait times per reward (across all sessions),
    using only opt-out, non-violation, non-NaN trials with wait_time <= 30.
    Adds a new column 'z_wait_time' to df.
    """
    df['z_wait_time'] = np.nan
    unique_rewards = df['reward'].dropna().unique()

    for reward in unique_rewards:
        trial_indices = df[
            (df['reward'] == reward) &
            (df['optout'] == 1) &
            (~df['wait_time'].isna()) &
            (df['wait_time'] <= 30) &
            (df['vios'] == 0)
        ].index

        if len(trial_indices) == 0:
            continue

        wait_times = df.loc[trial_indices, 'wait_time']
        m = wait_times.mean()
        s = wait_times.std()
        z_scores = (wait_times - m) / s if s != 0 else np.zeros(len(wait_times))

        df.loc[trial_indices, 'z_wait_time'] = z_scores

    return df

def get_transition_dynamics_from_df(beh_df):
    """
    Add block transition columns and z-scored wait times to the behavior dataframe.

    Parameters:
    beh_df (pd.DataFrame): DataFrame containing the behavior data.

    Returns:
    beh_df (pd.DataFrame): DataFrame with added transition columns (M_to_H, M_to_L, H_to_M, L_to_M)
                          and z_wait_time column.
    """
    tname = ['M_to_H', 'M_to_L', 'H_to_M', 'L_to_M']

    # get block switches
    b0 = beh_df['block'].values[:-1]
    b1 = beh_df['block'].values[1:]
    switches = np.where(b0 != b1)[0] + 1
    bprev = beh_df['block'].values[switches - 1]
    bcurr = beh_df['block'].values[switches]
    prev_block = [1, 1, 2, 3]
    curr_block = [2, 3, 1, 1]

    for name in tname:
        beh_df[name] = np.nan

    for i, t0 in enumerate(switches):
        curr_session = beh_df['session_num'].values[t0]

        ts = t0
        while ts > 0 and beh_df['session_num'].values[ts - 1] == curr_session:
            ts -= 1
            if beh_df['block'].values[ts] != bprev[i]:
                ts += 1
                break

        te = t0
        while te < len(beh_df) and beh_df['session_num'].values[te] == curr_session:
            if beh_df['block'].values[te] != bcurr[i]:
                break
            te += 1

        for j, (b0, b1) in enumerate(zip(prev_block, curr_block)):
            if (bprev[i] == b0) and (bcurr[i] == b1):
                idx = np.arange(ts, te)
                beh_df.loc[idx, tname[j]] = idx - t0

    # z-score wait times by reward
    unique_rewards = beh_df['reward'].unique()
    beh_df['z_wait_time'] = np.nan

    for reward in unique_rewards:
        trial_indices = beh_df[
            (beh_df['reward'] == reward) &
            (beh_df['optout'] == 1) &
            (~beh_df['wait_time'].isna()) &
            (beh_df['vios'] == 0)
        ].index
        if len(trial_indices) == 0:
            continue
        wait_times = beh_df.loc[trial_indices, 'wait_time']
        m = wait_times.mean()
        s = wait_times.std()
        if len(trial_indices) == 1:
            s = 1  # set std to 1 for single trial case
        z_scores = (wait_times - m) / s
        beh_df.loc[trial_indices, 'z_wait_time'] = z_scores

    return beh_df

def identify_transition_types(df):
    """
    Classify transition types based on block sequence.
    Returns a list the same length as df, with transition type labels at transition trials.
    """
    transition_type = [None] * len(df)
    block_series = df['block'].values
    reward_series = df['reward'].values

    for i in range(1, len(df)):
        if block_series[i] != block_series[i - 1]:
            prev_reward = reward_series[i - 1]
            curr_reward = reward_series[i]

            if prev_reward == 10 and curr_reward == 5:
                transition_type[i] = 'M_to_L'
            elif prev_reward == 10 and curr_reward == 20:
                transition_type[i] = 'M_to_H'
            elif prev_reward == 5 and curr_reward == 10:
                transition_type[i] = 'L_to_M'
            elif prev_reward == 20 and curr_reward == 10:
                transition_type[i] = 'H_to_M'

    return transition_type

def add_session_numbers(df, ntrials):
    """Add session numbers to dataframe based on ntrials array"""
    if len(ntrials) == 0:
        df['session'] = 1
        return df

    # convert ntrials to numpy array and flatten if needed
    ntrials = np.array(ntrials).flatten()

    # calculate session start indices using cumulative sum
    session_starts = np.concatenate([[0], np.cumsum(ntrials[:-1])])

    # initialize session column
    df['session'] = 0

    # assign session numbers
    for i, (start_idx, n_trials) in enumerate(zip(session_starts, ntrials)):
        start_idx = int(start_idx)
        n_trials = int(n_trials)
        end_idx = start_idx + n_trials
        # Make sure we don't exceed dataframe length
        end_idx = min(end_idx, len(df))
        df.iloc[start_idx:end_idx, df.columns.get_loc('session')] = i + 1

    return df

def get_optout_probability_no_plot(beh_df, bins=np.arange(0.5, 10.5, 0.5)):
    optout_prob = []
    for i in range(len(bins) - 1):
        trials = beh_df[(beh_df['wait_time'] >= bins[i]) &
                        (beh_df['wait_time'] < bins[i + 1]) &
                        (beh_df['vios'] == 0) &
                        (beh_df['catch'] == 0)]
        if len(trials) >= 10:
            optout_prob.append(trials['optout'].mean())
        else:
            optout_prob.append(np.nan)
    return np.array(optout_prob), bins[:-1]

def apply_valid_trial_filter(df):
    return df[(df['vios'] == 0) & (df['optout'] == 1) & (df['wait_time'] <= 30)].copy()

def remove_outliers_iqr(data):
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data >= lower) & (data <= upper)]

def remove_plot_outliers(series, method='iqr', max_val=1e3):
    x = series.dropna()
    if method == 'iqr':
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        x = x[(x >= lower) & (x <= upper)]
    x = x[np.abs(x) < max_val]
    return x


# function to perform random splits and analysis with stratification
def analyze_random_splits(data_df, k=500, rat_id=None, seed=42):
    """
    Perform k random 50/50 splits of the data with stratification by block
    Only uses optout trials without violations
    """
    # set random seed for reproducibility
    np.random.seed(seed)

    # filter for only optout trials without violations
    filtered_df = data_df[(data_df['optout'] == 1) & (data_df['vios'] == 0)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    # check if we have all blocks
    unique_blocks = filtered_df['block'].unique()
    if len(unique_blocks) < 3:
        print(f"Warning: Only found blocks {unique_blocks}. Need all 3 blocks for proper analysis.")

    n = len(filtered_df)
    split1_reward_sens = []
    split2_reward_sens = []
    split1_block_sens = []
    split2_block_sens = []

    # perform k random splits
    for i in range(k):
        try:
            # use stratified split to ensure each split has all blocks represented equally
            split1_idx, split2_idx = train_test_split(
                np.arange(n),
                test_size=0.5,
                stratify=filtered_df['block'],
                random_state=seed + i
            )

            # get data for each split
            split1_data = filtered_df.iloc[split1_idx]
            split2_data = filtered_df.iloc[split2_idx]

            # verify splits have all blocks
            if len(split1_data['block'].unique()) < 3 or len(split2_data['block'].unique()) < 3:
                continue

            # calculate reward sensitivity and block sensitivity for split 1
            _, block_sens1_ratio, slope1 = calc_block_sensitivity_and_mixed_slope(split1_data)

            # calculate reward sensitivity and block sensitivity for split 2
            _, block_sens2_ratio, slope2 = calc_block_sensitivity_and_mixed_slope(split2_data)

            # Check for NaN values
            if np.isnan(block_sens1_ratio) or np.isnan(slope1) or np.isnan(block_sens2_ratio) or np.isnan(slope2):
                continue

            # store results
            split1_reward_sens.append(slope1)
            split2_reward_sens.append(slope2)
            split1_block_sens.append(block_sens1_ratio)
            split2_block_sens.append(block_sens2_ratio)

        except Exception as e:
            continue

    # check if we got any valid results
    if len(split1_reward_sens) == 0:
        print(f"No valid splits found for rat {rat_id}")
        return None, None, None, None

    # convert to arrays
    split1_reward_sens = np.array(split1_reward_sens)
    split2_reward_sens = np.array(split2_reward_sens)
    split1_block_sens = np.array(split1_block_sens)
    split2_block_sens = np.array(split2_block_sens)

    # create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Reward sensitivity
    ax1.scatter(split1_reward_sens, split2_reward_sens, alpha=0, s=20, color='blue')

    # add x=y line
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),
        np.max([ax1.get_xlim(), ax1.get_ylim()])
    ]
    ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

    # add 2D density plot
    valid_mask = ~(np.isnan(split1_reward_sens) | np.isnan(split2_reward_sens))
    if np.sum(valid_mask) > 10:
        try:
            # check if there's enough variation in the data
            if (np.std(split1_reward_sens[valid_mask]) > 1e-6 and
                    np.std(split2_reward_sens[valid_mask]) > 1e-6):
                sns.kdeplot(x=split1_reward_sens[valid_mask], y=split2_reward_sens[valid_mask],
                            fill=True, cmap='Blues', alpha=0.5, bw_adjust=1, levels=10, ax=ax1)
        except Exception as e:
            print(f"Couldn't create KDE plot for reward sensitivity: {e}")

    ax1.set_xlabel('Split 1 Reward Sensitivity (Slope)')
    ax1.set_ylabel('Split 2 Reward Sensitivity (Slope)')
    ax1.set_title(f'Rat {rat_id}: Reward Sensitivity Comparison (k={k})')
    ax1.set_aspect('equal')

    # Plot 2: Block sensitivity
    ax2.scatter(split1_block_sens, split2_block_sens, alpha=0, s=20, color='green')

    # add x=y line
    lims = [
        np.min([ax2.get_xlim(), ax2.get_ylim()]),
        np.max([ax2.get_xlim(), ax2.get_ylim()])
    ]
    ax2.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

    # add 2D density plot
    valid_mask = ~(np.isnan(split1_block_sens) | np.isnan(split2_block_sens))
    if np.sum(valid_mask) > 10:
        try:
            # check if there's enough variation in the data
            if (np.std(split1_block_sens[valid_mask]) > 1e-6 and
                    np.std(split2_block_sens[valid_mask]) > 1e-6):
                sns.kdeplot(x=split1_block_sens[valid_mask], y=split2_block_sens[valid_mask],
                            fill=True, cmap='Blues', alpha=0.5, bw_adjust=1, levels=10, ax=ax2)
        except Exception as e:
            print(f"Couldn't create KDE plot for block sensitivity: {e}")

    ax2.set_xlabel('Split 1 Block Sensitivity (Ratio)')
    ax2.set_ylabel('Split 2 Block Sensitivity (Ratio)')
    ax2.set_title(f'Rat {rat_id}: Block Sensitivity Comparison (k={k})')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    # calculate variances for the differences
    reward_diff = split1_reward_sens - split2_reward_sens
    block_diff = split1_block_sens - split2_block_sens

    # print variance (removing NaNs)
    print(f"Rat {rat_id}:")
    print(f"  Std of Reward Sensitivity (Split1 - Split2) = {np.nanstd(reward_diff):.6f}")
    print(f"  Std of Block Sensitivity Ratio (Split1 - Split2) = {np.nanstd(block_diff):.6f}")

    return split1_reward_sens, split2_reward_sens, split1_block_sens, split2_block_sens

# helper function to unpack matlab structs into dictionaries
def unpack_mat_struct(mat_obj):
    if isinstance(mat_obj, mat_struct):
        return {field: unpack_mat_struct(getattr(mat_obj, field)) for field in mat_obj._fieldnames}
    elif isinstance(mat_obj, np.ndarray):
        if mat_obj.dtype == object:
            return [unpack_mat_struct(el) for el in mat_obj]
        else:
            return mat_obj
    else:
        return mat_obj

# load .mat file and return structured content
def load_mat_file(filepath):
    try:
        return loadmat(filepath, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"error loading mat file: {e}")
        return {}

# best-fit parameters from each fold
def extract_bestfit_params(BestFit_Folds):
    param_list = []
    fold_ids = []
    for i, fold in enumerate(BestFit_Folds):
        try:
            bestfit = fold.get('BestFit', None) if isinstance(fold, dict) else getattr(fold, 'BestFit', None)
            if bestfit is None:
                raise ValueError("Missing 'BestFit'")
            unpacked = unpack_mat_struct(bestfit)
            if isinstance(unpacked, dict) and 'params' in unpacked:
                params = unpacked['params']
                param_list.append(params if len(params) == 5 else [np.nan]*5)
            else:
                raise ValueError("Missing 'params'")
        except Exception as e:
            print(f"Error in fold {i}: {e}")
            param_list.append([np.nan]*5)
        fold_ids.append(i)
    df = pd.DataFrame(param_list, columns=['param1', 'param2', 'param3', 'param4', 'param5'])
    df['Fold'] = fold_ids
    return df

# negative log-likelihoods
def extract_nlls(BestFit_Folds):
    nll_train = []
    nll_test = []
    for fold in BestFit_Folds:
        try:
            nll_train.append(np.mean(fold.nLL_TrainFold))
            nll_test.append(np.mean(fold.nLL_TestFold))
        except:
            nll_train.append(np.nan)
            nll_test.append(np.nan)
    return pd.DataFrame({'nll_train': nll_train, 'nll_test': nll_test})

# paramFits matrix
def extract_param_fits(BestFit_Folds):
    all_fits = []
    fold_id = []
    for i, fold in enumerate(BestFit_Folds):
        try:
            fits = fold.paramFits
            all_fits.append(fits)
            fold_id.extend([i] * fits.shape[0])
        except:
            continue
    if all_fits:
        df = pd.DataFrame(np.vstack(all_fits), columns=['param1', 'param2', 'param3', 'param4', 'param5'])
        df['fold'] = fold_id
        return df
    else:
        return pd.DataFrame()

# seed values
def extract_seeds(BestFit_Folds):
    all_seeds = []
    fold_id = []
    for i, fold in enumerate(BestFit_Folds):
        try:
            seeds = fold.seeds
            all_seeds.append(seeds)
            fold_id.extend([i] * seeds.shape[0])
        except:
            continue
    if all_seeds:
        df = pd.DataFrame(np.vstack(all_seeds), columns=['seed1', 'seed2', 'seed3', 'seed4', 'seed5'])
        df['fold'] = fold_id
        return df
    else:
        return pd.DataFrame()

# train trials
def extract_train_trials(folds, rat=None):
    all_tables = []
    for i, fold in enumerate(folds):
        try:
            D = unpack_mat_struct(fold.ratTrial_TrainFold)
            if isinstance(D, dict):
                # determine per-trial length
                lengths = {k: len(v) for k, v in D.items() if isinstance(v, np.ndarray) and v.ndim == 1}
                if lengths:
                    n_trials = max(set(lengths.values()), key=list(lengths.values()).count)
                    per_trial_fields = {k: v for k, v in D.items() if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n_trials}
                    df = pd.DataFrame(per_trial_fields)
                    df['Fold'] = i
                    if rat:
                        df['rat'] = rat
                    all_tables.append(df)
        except Exception as e:
            print(f"error extracting train for {rat} fold {i}: {e}")
            continue
    return pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()

# test trials
def extract_test_trials(folds, rat=None):
    all_tables = []
    for i, fold in enumerate(folds):
        try:
            D = unpack_mat_struct(fold.ratTrial_TestFold)
            if isinstance(D, dict):
                # determine per-trial length
                lengths = {k: len(v) for k, v in D.items() if isinstance(v, np.ndarray) and v.ndim == 1}
                if lengths:
                    n_trials = max(set(lengths.values()), key=list(lengths.values()).count)
                    per_trial_fields = {k: v for k, v in D.items() if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n_trials}
                    df = pd.DataFrame(per_trial_fields)
                    df['Fold'] = i
                    if rat:
                        df['rat'] = rat
                    all_tables.append(df)
        except Exception as e:
            print(f"error extracting test for {rat} fold {i}: {e}")
            continue
    return pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()

# noise model
def extract_noise_models(BestFit_Folds):
    models = []
    for fold in BestFit_Folds:
        try:
            models.append(str(fold.noiseModel))
        except:
            models.append('')
    return pd.DataFrame({'noise_model': models})

# general trial extractor
def extract_trials_as_table(trial_struct):
    try:
        unpacked = unpack_mat_struct(trial_struct)
        return pd.DataFrame(unpacked)
    except:
        return pd.DataFrame()