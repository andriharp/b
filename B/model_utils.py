import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab.mio5_params import mat_struct

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
            params = unpack_mat_struct(fold.BestFit)['params']
            param_list.append(params if len(params) == 5 else [np.nan]*5)
        except:
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