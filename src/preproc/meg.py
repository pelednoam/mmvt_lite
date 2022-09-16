import os
import os.path as op
import time
import glob
import shutil
import numpy as np
from functools import partial
from collections import defaultdict, Iterable
from itertools import product
import traceback
try:
    import mne
except:
    print('No mne!')
try:
    import nibabel as nib
except:
    print('No nibabel!')
import types
import warnings
from collections import Counter, OrderedDict
import inspect
import copy

try:
    from tqdm import tqdm
except:
    print('No tqdm!')
try:
    from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                                  write_inverse_operator, read_inverse_operator)
    from mne.minimum_norm.inverse import _prepare_forward
    from mne.preprocessing import ICA
    from mne.preprocessing import create_ecg_epochs, create_eog_epochs
except:
    print('No mne!')
from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import labels_utils as lu
from src.utils import args_utils as au
from src.utils import freesurfer_utils as fu
from src.utils import stat_utils
from src.preproc import anatomy as anat
from src.preproc import connectivity as connectivity_preproc

SUBJECTS_MRI_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

LINKS_DIR = utils.get_links_dir()
# print('LINKS_DIR = {}'.format(LINKS_DIR))
if not op.isdir(LINKS_DIR):
    raise Exception('No links dir!')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
if not op.isdir(MEG_DIR):
    print('No MEG dir! ({})'.format(MEG_DIR))
LOOKUP_TABLE_SUBCORTICAL = op.join(MMVT_DIR, 'sub_cortical_codes.txt')

STAT_AVG, STAT_DIFF = range(2)
HEMIS = ['rh', 'lh']

SUBJECT, MRI_SUBJECT, SUBJECT_MEG_FOLDER, RAW, RAW_ICA, INFO, EVO, EVE, COV, EPO, EPO_NOISE, FWD_EEG, FWD_MEG, FWD_MEEG, FWD_SUB, FWD_X, \
FWD_SMOOTH, INV_EEG, INV_MEG, INV_MEEG, INV_SMOOTH, INV_EEG_SMOOTH, INV_SUB, INV_X, EMPTY_ROOM, MRI, SRC, SRC_SMOOTH, BEM, STC, \
STC_HEMI, STC_HEMI_SAVE, STC_HEMI_SMOOTH, STC_HEMI_SMOOTH_SAVE, STC_ST, \
COR, LBL, STC_MORPH, ACT, ASEG, DATA_COV, NOISE_COV_EEG, NOISE_COV_MEG, NOISE_COV_MEEG, DATA_CSD, NOISE_CSD, MEG_TO_HEAD_TRANS = [
                                                                                                                                     ''] * 47

locating_meg_file = lambda x, y: (x, True) if op.isfile(x) else ('', False)
locating_subject_file = lambda x, y: (x, True) if op.isfile(x) else ('', False)


def check_globals(subject=''):
    def real_tryit(func):
        def wrapper(*args, **kwargs):
            if INV_MEG == '':
                subject = kwargs.get('subject', args[0] if len(args) > 0 else '')
                if subject == '':
                    print('check_globals: No subject!')
                mri_subject = kwargs.get('mri_subject', subject)
                atlas = kwargs.get('atlas', '')
                remote_subject_dir = kwargs.get('remote_subject_dir', '')
                if not op.isdir(op.join(SUBJECTS_MRI_DIR, mri_subject)):
                    raise Exception('Can\'t find {}!'.format(op.join(SUBJECTS_MRI_DIR, mri_subject)))
                _args = read_cmd_args(dict(subject=subject, mri_subject=subject, atlas=atlas))
                fname_format, fname_format_cond, conditions = init_main(
                    subject, mri_subject, remote_subject_dir, _args)
                init_globals_args(subject, mri_subject, fname_format, fname_format_cond, args=_args)
            return func(*args, **kwargs)

        return wrapper

    return real_tryit


def init_globals_args(subject, mri_subject, fname_format, fname_format_cond, args):
    return init_globals(subject, mri_subject, fname_format, fname_format_cond, args.raw_fname_format,
                        args.fwd_fname_format, args.inv_fname_format, args.events_fname, args.files_includes_cond,
                        args.cleaning_method, args.contrast, args.task, args.meg_dir, args.mri_dir, args.mmvt_dir,
                        args.fwd_no_cond, args.inv_no_cond, args.data_per_task, args.sub_dirs_for_tasks)


def init_globals(subject, mri_subject='', fname_format='', fname_format_cond='', raw_fname_format='',
                 fwd_fname_format='', inv_fname_format='', events_fname='', files_includes_cond=False,
                 cleaning_method='', contrast='', task='', subjects_meg_dir='', subjects_mri_dir='', mmvt_dir='',
                 fwd_no_cond=False, inv_no_cond=False, data_per_task=False, sub_dirs_for_tasks=False, root_dir=''):
    global SUBJECT, MRI_SUBJECT, SUBJECT_MEG_FOLDER, RAW, RAW_ICA, INFO, EVO, EVE, COV, EPO, EPO_NOISE, FWD_EEG, FWD_MEG, FWD_MEEG, FWD_SUB, \
        FWD_X, FWD_SMOOTH, INV_EEG, INV_MEG, INV_MEEG, INV_SMOOTH, INV_EEG_SMOOTH, INV_SUB, INV_X, EMPTY_ROOM, MRI, SRC, SRC_SMOOTH, \
        BEM, STC, STC_HEMI, STC_HEMI_SAVE, STC_HEMI_SMOOTH, STC_HEMI_SMOOTH_SAVE, STC_ST, COR, AVE, LBL, STC_MORPH, \
        ACT, ASEG, MMVT_SUBJECT_FOLDER, DATA_COV, NOISE_COV_EEG, NOISE_COV_MEG, NOISE_COV_MEEG, DATA_CSD, NOISE_CSD, MEG_TO_HEAD_TRANS, \
        locating_meg_file, locating_subject_file
    if files_includes_cond:
        fname_format = fname_format_cond
    SUBJECT = subject
    MRI_SUBJECT = mri_subject if mri_subject != '' else subject
    if subjects_mri_dir == '':
        subjects_mri_dir = SUBJECTS_MRI_DIR
    os.environ['SUBJECT'] = SUBJECT
    if task != '':
        if data_per_task:
            SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, task, SUBJECT)
        elif sub_dirs_for_tasks:
            SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, SUBJECT, task)
        else:
            SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, SUBJECT)
    else:
        SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, SUBJECT)
    locating_meg_file = partial(utils.locating_file, parent_fols=[SUBJECT_MEG_FOLDER])
    if root_dir == '':
        root_dir = SUBJECT_MEG_FOLDER
    # if not op.isdir(SUBJECT_MEG_FOLDER):
    #     SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir)
    # if not op.isdir(SUBJECT_MEG_FOLDER):
    #     raise Exception("Can't find the subject's MEG folder! {}".format(SUBJECT_MEG_FOLDER))
    utils.make_dir(SUBJECT_MEG_FOLDER)
    print('Subject meg dir: {}'.format(SUBJECT_MEG_FOLDER))
    SUBJECT_MRI_FOLDER = op.join(subjects_mri_dir, MRI_SUBJECT)
    locating_subject_file = partial(utils.locating_file, parent_fols=[SUBJECT_MRI_FOLDER])
    MMVT_SUBJECT_FOLDER = op.join(mmvt_dir, MRI_SUBJECT)
    _get_fif_name_cond = partial(get_file_name, fname_format=fname_format, file_type='fif',
                                 cleaning_method=cleaning_method, contrast=contrast, raw_fname_format=raw_fname_format,
                                 fwd_fname_format=fwd_fname_format, inv_fname_format=inv_fname_format,
                                 root_dir=root_dir)
    _get_fif_name_no_cond = partial(_get_fif_name_cond, cond='')
    _get_fif_name = _get_fif_name_cond if files_includes_cond else _get_fif_name_no_cond
    _get_txt_name = partial(get_file_name, fname_format=fname_format, file_type='txt',
                            cleaning_method=cleaning_method)  # , contrast=contrast)
    _get_stc_name = partial(get_file_name, fname_format=fname_format_cond, file_type='stc',
                            cleaning_method=cleaning_method, contrast=contrast)
    _get_pkl_name = partial(get_file_name, fname_format=fname_format_cond, file_type='pkl',
                            cleaning_method=cleaning_method, contrast=contrast)
    _get_pkl_name_no_cond = partial(_get_pkl_name, cond='')

    RAW = _get_fif_name('raw', contrast='', cond='')
    RAW_ICA = _get_fif_name('ica-raw', contrast='', cond='')
    alt_raw_fname = '{}.fif'.format(RAW[:-len('-raw.fif')])
    if not op.isfile(RAW) and op.isfile(alt_raw_fname):
        RAW = alt_raw_fname
    INFO = _get_pkl_name_no_cond('raw-info')
    EVE = _get_txt_name('eve', cond='', contrast=contrast) if events_fname == '' else events_fname
    if not op.isfile(EVE):
        EVE = _get_txt_name('eve', cond='', contrast='')
    EVO = _get_fif_name('ave')
    COV = _get_fif_name('cov')
    DATA_COV = _get_fif_name('data-cov')
    NOISE_COV_EEG = _get_fif_name_no_cond('eeg-noise-cov')
    NOISE_COV_MEG = _get_fif_name_no_cond('meg-noise-cov')
    NOISE_COV_MEEG = _get_fif_name_no_cond('meeg-noise-cov')
    DATA_CSD = _get_pkl_name('data-csd')
    NOISE_CSD = _get_pkl_name('noise-csd')
    EPO = _get_fif_name('epo')
    EPO_NOISE = _get_fif_name('noise-epo')
    # FWD = _get_fif_name_no_cond('fwd') if fwd_no_cond else _get_fif_name_cond('fwd')
    FWD_EEG = _get_fif_name_no_cond('eeg-fwd') if fwd_no_cond else _get_fif_name_cond('eeg-fwd')
    FWD_MEG = _get_fif_name_no_cond('meg-fwd') if fwd_no_cond else _get_fif_name_cond('meg-fwd')
    FWD_MEEG = _get_fif_name_no_cond('meeg-fwd') if fwd_no_cond else _get_fif_name_cond('meeg-fwd')
    FWD_SUB = _get_fif_name_no_cond('sub-cortical-fwd') if fwd_no_cond else _get_fif_name_cond('sub-cortical-fwd')
    FWD_X = _get_fif_name_no_cond('{region}-fwd') if fwd_no_cond else _get_fif_name_cond('{region}-fwd')
    FWD_SMOOTH = _get_fif_name_no_cond('smooth-fwd') if inv_no_cond else _get_fif_name_cond('smooth-fwd')
    INV_EEG = _get_fif_name_no_cond('eeg-inv') if inv_no_cond else _get_fif_name_cond('eeg-inv')
    INV_MEG = _get_fif_name_no_cond('meg-inv') if inv_no_cond else _get_fif_name_cond('meg-inv')
    INV_MEEG = _get_fif_name_no_cond('meeg-inv') if inv_no_cond else _get_fif_name_cond('meeg-inv')
    INV_SUB = _get_fif_name_no_cond('sub-cortical-inv') if inv_no_cond else _get_fif_name_cond('sub-cortical-inv')
    INV_X = _get_fif_name_no_cond('{region}-inv') if inv_no_cond else _get_fif_name_cond('{region}-inv')
    INV_SMOOTH = _get_fif_name_no_cond('smooth-inv') if inv_no_cond else _get_fif_name_cond('smooth-inv')
    INV_EEG_SMOOTH = _get_fif_name_no_cond('eeg-smooth-inv') if inv_no_cond else _get_fif_name_cond('eeg-smooth-inv')
    EMPTY_ROOM = _get_fif_name_no_cond('empty-raw').replace('-{}-'.format(task), '-').replace(
        '_{}'.format(cleaning_method), '')
    STC = _get_stc_name('{method}-{modal}')
    STC_HEMI = _get_stc_name('{method}-{modal}-{hemi}')
    STC_HEMI_SAVE = op.splitext(STC_HEMI)[0].replace('-{hemi}', '')
    STC_HEMI_SMOOTH = _get_stc_name('{method}-smoothed-{hemi}')
    STC_HEMI_SMOOTH_SAVE = op.splitext(STC_HEMI_SMOOTH)[0].replace('-{hemi}', '')
    STC_MORPH = op.join(SUBJECT_MEG_FOLDER, task, '{}', '{}-{}-inv.stc')  # cond, method
    STC_ST = _get_pkl_name('{method}_st')
    # LBL = op.join(SUBJECT_MEG_FOLDER, 'labels_data_{}_{}_{}.npz') # atlas, extract_method, hemi
    LBL = op.join(MMVT_SUBJECT_FOLDER, 'meg', 'labels_data_{}_{}_{}_{}_{}.npz')  # task, atlas, extract_method, hemi
    ACT = op.join(MMVT_SUBJECT_FOLDER, 'activity_map_{}')  # hemi
    # MRI files
    MRI = op.join(SUBJECT_MRI_FOLDER, 'mri', 'transforms', '{}-trans.fif'.format(MRI_SUBJECT))
    SRC = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-oct-6p-src.fif'.format(MRI_SUBJECT))
    SRC_SMOOTH = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-all-src.fif'.format(MRI_SUBJECT))
    BEM = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-5120-5120-5120-bem-sol.fif'.format(MRI_SUBJECT))
    COR = op.join(SUBJECT_MRI_FOLDER, 'mri', 'T1-neuromag', 'sets', 'COR.fif')
    if not op.isfile(COR):
        COR = op.join(SUBJECT_MEG_FOLDER, '{}-cor-trans.fif'.format(SUBJECT))
    ASEG = op.join(SUBJECT_MRI_FOLDER, 'ascii')
    MEG_TO_HEAD_TRANS = op.join(SUBJECT_MEG_FOLDER, 'trans', 'meg_to_head_trans.npy')


def print_files_names():
    print_file(RAW, 'raw')
    print_file(EVE, 'events')
    print_file(FWD, 'forward')
    print_file(INV, 'inverse')
    print_file(EPO, 'epochs')
    print_file(EVO, 'evoked')
    # MRI files
    print_file(MRI, 'subject MRI transform')
    print_file(SRC, 'subject MRI source')
    print_file(BEM, 'subject MRI BEM model')
    print_file(COR, 'subject MRI co-registration')


def print_file(fname, file_desc):
    print('{}: {} {}'.format(file_desc, fname, " !!!! doesn't exist !!!!" if not op.isfile(fname) else ""))


def get_file_name(ana_type, subject='', file_type='fif', fname_format='', cond='{cond}', cleaning_method='',
                  contrast='', root_dir='', raw_fname_format='', fwd_fname_format='', inv_fname_format=''):
    if fname_format == '':
        fname_format = '{subject}-{ana_type}.{file_type}'
    if subject == '':
        subject = SUBJECT
    args = {'subject': subject, 'ana_type': ana_type, 'file_type': file_type,
            'cleaning_method': cleaning_method, 'contrast': contrast}
    if '{cond}' in fname_format:
        args['cond'] = cond
    if ana_type == 'raw' and raw_fname_format != '':
        fname_format = raw_fname_format
    elif ana_type == 'fwd' and fwd_fname_format != '':
        fname_format = fwd_fname_format
    elif ana_type == 'inv' and inv_fname_format != '':
        fname_format = inv_fname_format
    if '{ana_type}' not in fname_format and '{file_type}' not in fname_format:
        fname_format = '{}-{}.{}'.format(fname_format, '{ana_type}', '{file_type}')
    fname = fname_format.format(**args)
    while '__' in fname:
        fname = fname.replace('__', '_')
    if '_-' in fname:
        fname = fname.replace('_-', '-')
    if root_dir == '':
        root_dir = SUBJECT_MEG_FOLDER
    return op.join(root_dir, fname)


def load_raw(raw_fname='', bad_channels=[], l_freq=None, h_freq=None, raw_template='*raw.fif'):
    # read the data
    raw_fname, raw_exist = locating_meg_file(raw_fname, glob_pattern=raw_template)
    if not raw_exist:
        print('Can\'t find the raw data using the template {}, trying \'*raw*.fif\''.format(raw_template))
        raw_fname, raw_exist = locating_meg_file(raw_fname, glob_pattern='*raw*.fif')
        if not raw_exist:
            raise Exception("Coulnd't find the raw file! {}".format(raw_fname))
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    if not op.isfile(INFO):
        utils.save(raw.info, INFO)
    if len(bad_channels) > 0:
        raw.info['bads'] = bad_channels
    if l_freq or h_freq:
        raw = raw.filter(l_freq, h_freq)
    return raw


def calcNoiseCov(epoches):
    noiseCov = mne.compute_covariance(epoches, tmax=None)
    # regularize noise covariance
    # noiseCov = mne.cov.regularize(noiseCov, evoked.info,
    #     mag=0.05, proj=True) # grad=0.05, eeg=0.1
    noiseCov.save(COV)
    # allEpoches = findEpoches(raw, picks, events, dict(onset=20), tmin=0, tmax=3.5)
    # evoked = allEpoches['onset'].average()
    # evoked.save(EVO)


# def calc_demi_epoches(windows_length, windows_shift, windows_num, raw, tmin, baseline,
#                       pick_meg=True, pick_eeg=False, pick_eog=False, reject=True,
#                       reject_grad=4000e-13, reject_mag=4e-12, reject_eog=150e-6, task='', epoches_fname=EPO):
#     demi_events, demi_events_ids = create_demi_events(raw, windows_length, windows_shift)
#     demi_tmax = windows_length / 1000.0
#     calc_epoches(raw, demi_events_ids, tmin, demi_tmax, baseline, False, demi_events, None, pick_meg, pick_eeg,
#                  pick_eog, reject, reject_grad, reject_mag, reject_eog, False, epoches_fname, task,
#                  windows_length, windows_shift, windows_num)


def create_demi_events(raw, windows_length, windows_shift, epoches_nun=0):
    # todo: replace with mne.event.make_fixed_length_events
    import math
    T = raw.last_samp - raw.first_samp + 1
    if epoches_nun == 0:
        epoches_nun = math.floor((T - windows_length) / windows_shift + 1)
    demi_events = np.zeros((epoches_nun, 3), dtype=np.uint32)
    for win_ind in range(epoches_nun):
        demi_events[win_ind] = [win_ind * windows_shift, win_ind * windows_shift + windows_length, 0]
    demi_events[:, :2] += raw.first_samp
    demi_conditions = {'demi': 0}
    return demi_events, demi_conditions


@check_globals()
def calc_epoches(subject, raw, conditions, tmin, tmax, baseline=None, read_events_from_file=False, events=None,
                 stim_channels=None, pick_meg=True, pick_eeg=False, pick_eog=False, reject=True,
                 reject_grad=4000e-13, reject_mag=4e-12, reject_eog=150e-6, remove_power_line_noise=True,
                 power_line_freq=60, epo_fname='', task='', windows_length=1000, windows_shift=500,
                 windows_num=0, overwrite_epochs=False, eve_template='*eve.fif', raw_fname='',
                 using_auto_reject=True, ar_compute_thresholds_method='random_search', ar_consensus_percs=None,
                 ar_n_interpolates=None, bad_ar_threshold=0.5, use_demi_events=False, notch_widths=None,
                 read_from_raw=False, read_events_as_annotation=False, n_jobs=6):
    if not read_from_raw and epo_fname != '':
        epo_fname = get_epo_fname(epo_fname, overwrite=overwrite_epochs)
        if op.isfile(epo_fname) and not overwrite_epochs:
            epochs = mne.read_epochs(epo_fname)
            return epochs
    try:
        events, _ = read_events(
            task, events, raw, tmax, read_events_from_file, stim_channels, windows_length, windows_shift, windows_num,
            eve_template, use_demi_events, read_events_as_annotation)
    except:
        # ret = input('No events, should create one epoch from raw (y/n)? ')
        # if not au.is_true(ret):
        #     raise Exception('No events!')
        events = np.array([0, len(raw.times), 1]).reshape((1, 3))
        conditions = {'all': 1}
    if tmax - tmin <= 0:
        raise Exception('tmax-tmin must be greater than zero!')
    print('Events: {}'.format(Counter(events[:, 2])))
    unique_events = np.unique(events[:, 2])
    if len(unique_events) == 1:
        events_conditions = {k: unique_events[0] for k, v in conditions.items()}
    else:
        events_conditions = {k: v for k, v in conditions.items() if v in unique_events}

    # if using_auto_reject and is_autoreject_installed:
    #     clean_epochs = calc_epochs_using_auto_reject(
    #         raw, events, events_conditions, tmin, tmax, picks, baseline, ar_compute_thresholds_method,
    #         ar_consensus_percs, ar_n_interpolates, bad_ar_threshold, epo_fname, overwrite_epochs, n_jobs)
    #     return clean_epochs

    picks = mne.pick_types(raw.info, meg=pick_meg, eeg=pick_eeg, eog=pick_eog, exclude='bads')
    if reject:
        reject_dict = {}
        if pick_meg and reject_mag > 0:
            reject_dict['mag'] = reject_mag
        if pick_meg and reject_grad > 0:
            reject_dict['grad'] = reject_grad
        if pick_eog and reject_eog > 0:
            reject_dict['eog'] = reject_eog
    else:
        reject_dict = None
    if remove_power_line_noise:
        raw.notch_filter(np.arange(power_line_freq, power_line_freq * 4 + 1, power_line_freq),
                         notch_widths=notch_widths, picks=picks)

    epochs = mne.Epochs(raw, events, events_conditions, tmin, tmax, proj=True, picks=picks,
                        baseline=baseline, preload=True, reject=reject_dict)
    if reject:
        epochs, bad_channels = auto_remove_bad_channels(
            raw, epochs, baseline, events, events_conditions, tmin, tmax, raw_fname, pick_meg, pick_eeg, pick_eog,
            reject_dict)
    print('Bad channels: {}'.format(bad_channels))
    print('{} good epochs'.format(len(epochs)))
    # epochs.info['bads'] = raw.info['bads']
    if epo_fname != '':
        save_epochs(epochs, epo_fname)
    return epochs


def read_events(task='', events=None, raw=None, tmax=0, read_events_from_file=False, stim_channels=None,
                windows_length=1000, windows_shift=500, windows_num=0, eve_template='*eve.fif', use_demi_events=False,
                read_events_as_annotation=False):
    if not raw is None and read_events_as_annotation:
        events, events_dict = mne.events_from_annotations(raw)
        print(events_dict)
    elif read_events_from_file:
        events_fname, event_fname_exist = locating_meg_file(EVE, glob_pattern=eve_template)
        if events is None and event_fname_exist:
            # events_fname = events_fname if events_fname != '' else EVE
            print('read events from {}'.format(events_fname))
            events = mne.read_events(events_fname)
    elif use_demi_events or (events is not None and events.shape[0] == 0):
        if task != 'rest':
            ans = input('Are you sure you want to have only one epoch, containing all the data (y/n)? ')
            if ans != 'y':
                return None
        events, conditions = create_demi_events(raw, windows_length, windows_shift, windows_num)
        tmax = windows_length / 1000.0
    elif events is None:
        try:
            print('Finding events in {}'.format(stim_channels))
            events = mne.find_events(raw, stim_channel=stim_channels, shortest_event=1)
        except:
            utils.print_last_error_line()

    if events is None:
        print('Trying to find events file')
        events_fname, event_fname_exist = locating_meg_file(EVE, glob_pattern=eve_template)
        if events is None and event_fname_exist:
            print('read events from {}'.format(events_fname))
            events = mne.read_events(events_fname)

    if events is None:
        raise Exception('Can\'t find events!')

    return events, tmax


def auto_remove_bad_channels(raw, epochs, baseline, events, events_conditions, tmin, tmax, raw_fname='',
                             pick_meg=True, pick_eeg=False, pick_eog=False, reject_dict=None):
    min_bad_num = len(events) * 0.5
    bad_channels_max_num = 20
    min_good_epochs_num = min(50, sum([sum(events[:, 2] == k) for k in events_conditions.values()]) * 0.5)
    bad_channels = Counter(utils.flat_list_of_lists(epochs.drop_log))
    bad_channels = {ch_name: num for ch_name, num in bad_channels.items() if ch_name in raw.info['ch_names']}
    if len(bad_channels) == 0:
        return epochs, []
    if len(epochs) < min_bad_num or max(bad_channels.values()) > bad_channels_max_num:
        for bad_ch, cnt in bad_channels.items():
            if cnt > bad_channels_max_num:
                raw.info['bads'].append(bad_ch)
        # if raw_fname == '':
        #     raw_fname = get_raw_fname(raw_fname)
        # try:
        #     info_fname, info_exist = get_info_fname('')
        #     utils.save(raw.info, info_fname)
        #     raw.save(raw_fname, overwrite=True)
        # except:
        #     print('Error in saving the new raw!')
        picks = mne.pick_types(raw.info, meg=pick_meg, eeg=pick_eeg, eog=pick_eog, exclude='bads')
        epochs = mne.Epochs(raw, events, events_conditions, tmin, tmax, proj=True, picks=picks,
                            baseline=baseline, preload=True, reject=reject_dict)
    if len(epochs) < min_good_epochs_num:
        raise Exception('Not enough good epochs!')
    print('Bad channles: {}'.format(raw.info['bads']))
    return epochs, raw.info['bads']


def is_autoreject_installed():
    try:
        from autoreject import LocalAutoRejectCV, compute_thresholds
        return True
    except:
        print('You should install first autoreject (http://autoreject.github.io/)!')
        return False


def calc_epochs_using_auto_reject(raw_or_epochs, events, events_conditions, tmin, tmax, picks, baseline,
                                  compute_thresholds_method='random_search', consensus_percs=None, n_interpolates=None,
                                  bad_ar_threshold=0.5, epo_fname='', overwrite=False, n_jobs=6):
    try:
        from autoreject import LocalAutoRejectCV, compute_thresholds
    except:
        print('You should install first autoreject (http://autoreject.github.io/)!')
        return False

    epo_fname = get_epo_fname(epo_fname, load_autoreject_if_exist=False)
    ar_epo_fname = '{}ar-epo.fif'.format(epo_fname[:-len('epo.fif')])
    if op.isfile(ar_epo_fname) and not overwrite:
        print('Autoreject already calculated, use \'overwrite=True\' to recalculate.')
        epochs_ar = mne.read_epochs(ar_epo_fname)
        return epochs_ar

    if consensus_percs is None:
        consensus_percs = np.linspace(0, 1.0, 11)
    # if n_interpolates is None:
    #     n_interpolates = [1,2,3,5,7,10,20], #np.array([1, 4, 32])
    # The reject params will be set to None because we do not want epochs to be dropped when instantiating mne.Epochs.
    if isinstance(raw_or_epochs, mne.io.fiff.Raw):
        epochs = mne.Epochs(
            raw_or_epochs, events, events_conditions, tmin, tmax, proj=True, picks=picks, baseline=baseline,
            preload=True, reject=None, detrend=0)
    elif isinstance(raw_or_epochs, mne.Epochs):
        epochs = raw_or_epochs
    # compute_thresholds_method in ['bayesian_optimization' or 'random_search']
    thresh_func = partial(compute_thresholds, random_state=89, method=compute_thresholds_method, n_jobs=n_jobs)
    ar = LocalAutoRejectCV(thresh_func=thresh_func, consensus_percs=consensus_percs,
                           n_interpolates=n_interpolates, verbose='tqdm', picks=picks)
    ar.fit(epochs)
    rejected = float(len(ar.bad_epochs_idx)) / len(epochs)
    print('Autoreject rejected {} of epochs'.format(100 * rejected))
    epochs_ar = ar.transform(epochs)
    # if autoreject would throw out over half of epochs for a channel, mark as bad
    previous_bads = epochs_ar.info['bads']
    for i in range(ar.bad_segments.shape[1]):
        if sum(ar.bad_segments[:, i]) > len(ar.bad_segments[:, i]) * bad_ar_threshold:
            epochs_ar.info['bads'].append(epochs_ar.ch_names[i])
    if previous_bads != epochs_ar.info['bads']:
        difference = [ch for ch in previous_bads if ch not in epochs_ar.info['bads']]
        print('Greater than {} of epochs thrown out for: {}'.format(
            100 * bad_ar_threshold, ' '.join([str(ch) for ch in difference])))
        epochs_ar = calc_epochs_using_auto_reject(
            epochs_ar, events, events_conditions, tmin, tmax, picks, baseline, compute_thresholds_method,
            consensus_percs, n_interpolates, bad_ar_threshold, epo_fname, overwrite, n_jobs)
    else:
        save_epochs(epochs_ar, ar_epo_fname)
    return epochs_ar


def save_epochs(epochs, epo_fname=''):
    if epo_fname == '':
        epo_fname = EPO  # get_epo_fname(epo_fname)
    if '{cond}' in epo_fname:
        for event in epochs.event_id:  # events.keys():
            epochs[event].save(get_cond_fname(epo_fname, event))
    else:
        try:
            epochs.save(epo_fname, overwrite=True)
        except:
            print(traceback.format_exc())


# def createEventsFiles(behavior_file, pattern):
#     make_ecr_events(RAW, behavior_file, EVE, pattern)

def calc_epochs_necessary_files(args):
    necessary_files = []
    if args.calc_epochs_from_raw:
        necessary_files.append(RAW)
        if args.read_events_from_file:
            necessary_files.append(EVE)
    else:
        necessary_files.append(EPO)
    return necessary_files


def calc_epochs_wrapper_args(subject, conditions, args, raw=None):
    return calc_epochs_wrapper(
        subject, conditions, args.t_min, args.t_max, args.baseline, raw, args.read_events_from_file,
        None, args.stim_channels,
        args.pick_meg, args.pick_eeg, args.pick_eog, args.reject,
        args.reject_grad, args.reject_mag, args.reject_eog, args.remove_power_line_noise, args.power_line_freq,
        args.bad_channels, args.l_freq, args.h_freq, args.task, args.windows_length, args.windows_shift,
        args.windows_num, args.overwrite_epochs, args.epo_fname, args.raw_fname, args.eve_template,
        args.using_auto_reject, args.ar_compute_thresholds_method, args.ar_consensus_percs,
        args.ar_n_interpolates, args.bad_ar_threshold, args.use_demi_events, args.power_line_notch_widths,
        args.read_events_as_annotation, args.downsample_r, args.n_jobs)


def calc_epochs_wrapper(
        subject, conditions, tmin, tmax, baseline, raw=None, read_events_from_file=False, events_mat=None,
        stim_channels=None, pick_meg=True, pick_eeg=False, pick_eog=False,
        reject=True, reject_grad=4000e-13, reject_mag=4e-12, reject_eog=150e-6, remove_power_line_noise=True,
        power_line_freq=60, bad_channels=[], l_freq=None, h_freq=None, task='', windows_length=1000, windows_shift=500,
        windows_num=0, overwrite_epochs=False, epo_fname='', raw_fname='', eve_template='*eve.fif',
        using_auto_reject=True, ar_compute_thresholds_method='random_search', ar_consensus_percs=None,
        ar_n_interpolates=None, bad_ar_threshold=0.5, use_demi_events=False, notch_widths=None,
        read_events_as_annotation=False, downsample=1, n_jobs=6):
    # Calc evoked data for averaged data and for each condition
    try:
        if epo_fname == '':
            epo_fname = EPO
        if '{cond}' not in epo_fname:
            epo_exist = op.isfile(epo_fname)
        else:
            epo_exist = all([op.isfile(get_cond_fname(epo_fname, cond)) for cond in conditions.keys()])
        if epo_exist and not overwrite_epochs:
            if '{cond}' in epo_fname:
                epo_exist = True
                epochs = {}
                for cond in conditions.keys():
                    if op.isfile(get_cond_fname(epo_fname, cond)):
                        epochs[cond] = mne.read_epochs(get_cond_fname(epo_fname, cond))
                    else:
                        epo_exist = False
                        break
            else:
                if op.isfile(epo_fname):
                    epochs = mne.read_epochs(epo_fname)
        if not epo_exist or overwrite_epochs:
            read_from_raw = False
            if raw is None:
                if raw_fname == '':
                    raw_fname = get_raw_fname(raw_fname)
                raw = load_raw(raw_fname, bad_channels, l_freq, h_freq)
                read_from_raw = True
            epochs = calc_epoches(
                subject, raw, conditions, tmin, tmax, baseline, read_events_from_file, events_mat, stim_channels,
                pick_meg, pick_eeg, pick_eog, reject, reject_grad, reject_mag, reject_eog, remove_power_line_noise,
                power_line_freq, epo_fname, task, windows_length, windows_shift, windows_num, overwrite_epochs,
                eve_template, raw_fname, using_auto_reject, ar_compute_thresholds_method, ar_consensus_percs,
                ar_n_interpolates, bad_ar_threshold, use_demi_events, notch_widths, read_from_raw,
                read_events_as_annotation, n_jobs)
        # if task != 'rest':
        #     all_evoked = calc_evoked_from_epochs(epochs, conditions)
        # else:
        #     all_evoked = None
        flag = True
    except:
        print(traceback.format_exc())
        epochs = None
        flag = False

    return flag, epochs


def calc_epochs_psd(subject, events, mri_subject='', epo_fname='', apply_SSP_projection_vectors=True,
                    add_eeg_ref=True, epochs=None, fmin=0, fmax=200, bandwidth=2., tmin=None, tmax=None,
                    adaptive=False, modality='meg', max_epochs_num=0, raw_template='', precentiles=(1, 99),
                    overwrite=False, n_jobs=4):
    if mri_subject == '':
        mri_subject = subject
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) else ['all']
    epo_fname = get_epo_fname(epo_fname)
    sensors_picks, sensors_names = get_sensors_picks(subject, modality, raw_template=raw_template)
    fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, modality))
    ret = True
    for cond_ind, cond_name in enumerate(events_keys):
        output_fname = op.join(fol, '{}_sensors_psd.npz'.format(cond_name))
        if op.isfile(output_fname) and not overwrite:
            continue
        if epochs is None:
            epo_cond_fname = get_cond_fname(epo_fname, cond_name)
            if not op.isfile(epo_cond_fname):
                print('Epochs file was not found! ({})'.format(epo_cond_fname))
                return False
            epochs = mne.read_epochs(epo_cond_fname, apply_SSP_projection_vectors, add_eeg_ref)
        if max_epochs_num > 0:
            epochs = epochs[:max_epochs_num]
        try:
            mne.set_eeg_reference(epochs, ref_channels=None)
            epochs.apply_proj()
        except:
            print('annot create EEG average reference projector (no EEG data found)')
        picks = mne.pick_types(epochs.info, meg=True, exclude='bads')
        ch_names = [epochs.info['ch_names'][k].replace(' ', '') for k in picks]
        channels_sensors_dict = np.array([np.where(sensors_names == c)[0][0] for c in ch_names])
        psds, freqs = mne.time_frequency.psd_multitaper(
            epochs, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, bandwidth=bandwidth, adaptive=adaptive,
            low_bias=True, normalization='length', picks=picks, proj=False, n_jobs=n_jobs)
        all_psds = np.empty((len(epochs), len(sensors_picks), len(freqs)))
        all_psds.fill(np.nan)
        all_psds[:, channels_sensors_dict, :] = psds
        np.savez(output_fname, psds=all_psds, freqs=freqs)
        ret = ret and _calc_epochs_bands_psd(mri_subject, all_psds, freqs, None, cond_name, precentiles, overwrite)
    return ret


def get_sensors_picks(subject, modality, info_fname='', info=None, raw_template=''):
    if info is None:
        info_fname, info_exist = get_info_fname(subject, info_fname)
        if not info_exist:
            raw_fname = get_raw_fname(raw_template, include_empty=False)
            if not op.isfile(raw_fname):
                print('No raw or raw info file!')
                return None
            raw = mne.io.read_raw_fif(raw_fname)
            info = raw.info
            utils.save(info, info_fname)
        else:
            info = utils.load(info_fname)
    sensors_picks = mne.io.pick.pick_types(info, meg=modality == 'meg', eeg=modality == 'eeg', exclude=[])
    sensors_names = np.array([info['ch_names'][k].replace(' ', '') for k in sensors_picks])
    return sensors_picks, sensors_names


def calc_epochs_bands_psd(mri_subject, events, precentiles=(1, 99), bands=None, overwrite=False):
    meg_fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg'))
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) else ['all']
    if bands is None:
        bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    ret = True
    for cond_ind, cond_name in enumerate(events_keys):
        input_fname = op.join(meg_fol, '{}_sensors_psd.npz'.format(cond_name))
        if not op.isfile(input_fname):
            print('No power specturm for {}!'.format(cond_name))
            continue
        d = utils.Bag(np.load(input_fname))
        psds = d.psds  # (epochs_num, len(sensors), len(freqs), len(events))
        freqs = d.freqs
        _calc_epochs_bands_psd(mri_subject, psds, freqs, bands, cond_name, precentiles, overwrite)
    return ret


def _calc_epochs_bands_psd(mri_subject, psds, freqs, bands=None, cond_name='all', precentiles=(1, 99), overwrite=False):
    meg_fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg'))
    if bands is None:
        bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    ret = True
    for band, (lf, hf) in bands.items():
        output_fname = op.join(meg_fol, '{}_sensors_{}_psd.npz'.format(cond_name, band))
        if op.isfile(output_fname) and not overwrite:
            continue
        band_mask = np.where((freqs >= lf) & (freqs <= hf))[0]
        if psds.ndim == 3:
            band_psd = psds[:, :, band_mask].mean(axis=2).squeeze().T  # sensors x epochs
        elif psds.ndim == 2:
            band_psd = psds[:, band_mask].mean(axis=1)  # sensors
        data_max = utils.calc_max(band_psd, norm_percs=precentiles)
        print('calc_labels_power_bands: Saving results in {}'.format(output_fname))
        np.savez(output_fname, data=band_psd, title='sensors {} power ({})'.format(band, cond_name),
                 data_min=0, data_max=data_max)
        ret = ret and op.isfile(output_fname)
    return ret


def calc_baseline_sensors_bands_psd(mri_subject, epo_fname='', raw_template='', modality='meg', bad_channels=[],
                                    baseline_len=10000, l_freq=1, h_freq=120, bandwidth=2., cond_name='all',
                                    precentiles=(1, 99),
                                    overwrite=False, adaptive=False, bands=None, n_jobs=4):
    fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, modality))
    output_fname = op.join(fol, 'raw_sensors_psd.npz')
    if op.isfile(output_fname) and not overwrite:
        print('{} already exist'.format(output_fname))
        return True
    raw_fname = get_raw_fname(raw_template, include_empty=False)
    if not op.isfile(raw_fname):
        print('calc_raw_bands_psd: No raw file!')
        return None
    raw = load_raw(raw_fname, bad_channels, l_freq, h_freq)
    try:
        raw.set_eeg_reference('average', projection=True)  # set EEG average reference
    except:
        print('calc_raw_bands_psd: Can\'t set_eeg_reference')
    cond_name = 'baseline_{}'.format(cond_name)
    sensors_picks, sensors_names = get_sensors_picks(mri_subject, modality, raw_template=raw_template)
    epo_fname = get_epo_fname(epo_fname)
    epochs = mne.read_epochs(epo_fname)
    t_end = epochs.events[0, 0]
    t_start = 0 if t_end <= baseline_len else t_end - baseline_len
    raw.crop(t_start / raw.info['sfreq'], t_end / raw.info['sfreq'])
    picks = mne.pick_types(raw.info, meg=modality == 'meg', eeg=modality == 'eeg', exclude='bads')
    psds, freqs = mne.time_frequency.psd_multitaper(
        raw, fmin=l_freq, fmax=h_freq, bandwidth=bandwidth, adaptive=adaptive,
        low_bias=True, normalization='length', picks=picks, proj=False, n_jobs=n_jobs)
    print(psds.shape)
    ch_names = [raw.info['ch_names'][k].replace(' ', '') for k in picks]
    channels_sensors_dict = np.array([np.where(sensors_names == c)[0][0] for c in ch_names])
    all_psds = np.empty((len(sensors_picks), len(freqs)))
    all_psds.fill(np.nan)
    all_psds[channels_sensors_dict, :] = psds
    np.savez(output_fname, psds=all_psds, freqs=freqs)
    ret = _calc_epochs_bands_psd(mri_subject, all_psds, freqs, bands, cond_name, precentiles, overwrite)
    return ret


def calc_source_baseline_psd(subject, task, mri_subject='', raw_fname='', epo_fname='', inv_fname='', method='dSPM',
                             snr=3.0,
                             baseline_len=10000, l_freq=1, h_freq=120, n_fft=2048, overlap=0.5,
                             pick_ori='normal', bad_channels=[], nave=1, pca=True, bandwidth='hann', adaptive=False,
                             fwd_usingMEG=True, fwd_usingEEG=True, output_as_db=False, bands=None, overwrite=False,
                             n_jobs=4):
    if mri_subject == '':
        mri_subject = subject
    if bands is None:
        bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    if inv_fname == '':
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    if not op.isfile(inv_fname):
        print('calc_source_baseline_psd: No inv! {}'.format(inv_fname))
        return False
    epo_fname = get_epo_fname(epo_fname)
    if not op.isfile(epo_fname):
        print('calc_source_baseline_psd: Can\'t find the epochs file! {}'.format(epo_fname))
        return False
    raw_fname = get_raw_fname(raw_fname, include_empty=False)
    if not op.isfile(raw_fname):
        print('calc_raw_bands_psd: No raw file!')
        return None
    lambda2 = 1.0 / snr ** 2
    fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg'))
    baseline_psd_output_fname = op.join(fol, '{}_{}_baseline_psd'.format(task, method))
    sensors_output_fname = op.join(fol, '{}_sensors_baseline_psd-eve.fif'.format(task))
    if utils.both_hemi_files_exist('{}-{}.stc'.format(baseline_psd_output_fname, '{hemi}')) and not overwrite:
        print('source psd already exist'.format(baseline_psd_output_fname))
        return True

    raw = load_raw(raw_fname, bad_channels, l_freq, h_freq)
    try:
        raw.set_eeg_reference('average', projection=True)  # set EEG average reference
    except:
        print('calc_raw_bands_psd: Can\'t set_eeg_reference')

    epochs = mne.read_epochs(epo_fname)
    t_end = epochs.events[0, 0]
    t_start = 0 if t_end <= baseline_len else t_end - baseline_len
    raw.crop(t_start / raw.info['sfreq'], t_end / raw.info['sfreq'])

    inverse_operator = read_inverse_operator(inv_fname)
    stc_psd, sensors_psd = mne.minimum_norm.compute_source_psd(
        raw, inverse_operator, lambda2, method, 0, None, l_freq, h_freq, n_fft, overlap, pick_ori, None, nave, pca,
        bandwidth=bandwidth, adaptive=adaptive, return_sensor=True, dB=output_as_db, n_jobs=n_jobs)
    mne.write_evokeds(sensors_output_fname, sensors_psd)
    stc_psd.save(baseline_psd_output_fname)
    freqs = stc_psd.times
    for band, (lf, hf) in bands.items():
        stc_band_fname = op.join(fol, '{}_{}_baseline_{}'.format(task, method, band))
        if utils.both_hemi_files_exist('{}-{}.stc'.format(stc_band_fname, '{hemi}')) and not overwrite:
            continue
        band_mask = np.where((freqs >= lf) & (freqs <= hf))[0]
        data = np.concatenate([stc_psd.lh_data[:, band_mask].mean(axis=1), stc_psd.rh_data[:, band_mask].mean(axis=1)])
        data = np.reshape(data, (len(data), 1))
        stc_band_power = mne.SourceEstimate(data, stc_psd.vertices, 0, 0, subject=subject)
        print('Saving power stc to: {}'.format(stc_band_fname))
        stc_band_power.save(stc_band_fname)
    return all([utils.both_hemi_files_exist(op.join(fol, '{}_{}_baseline_{}-{}.stc'.format(
        task, method, band, '{hemi}'))) for band in bands.keys()]) and \
           utils.both_hemi_files_exist('{}-{}.stc'.format(baseline_psd_output_fname, '{hemi}')) and \
           op.isfile(sensors_output_fname)


def normalize_stc(subject, stc, baseline):
    norm_data = {}
    for hemi_ind, hemi in enumerate(['lh', 'rh']):
        psd_data = stc.lh_data if hemi == 'lh' else stc.rh_data
        baseline_data = baseline.lh_data if hemi == 'lh' else baseline.rh_data
        norm_data[hemi] = np.zeros((len(psd_data)))
        baseline_ind = 0
        for vert_ind, vert in enumerate(stc.vertices[hemi_ind]):
            if vert != baseline.vertices[hemi_ind][baseline_ind]:
                baseline_inds = np.where(baseline.vertices[hemi_ind] == vert)[0]
                if len(baseline_inds) > 0:
                    baseline_ind = baseline_inds[0]
                else:
                    print('Can\'t find baseline for {} {} {}'.format(subject, hemi, vert))
            norm_data[hemi][vert_ind] = psd_data[vert_ind] / baseline_data[baseline_ind]
            baseline_ind += 1
    data = np.concatenate([norm_data['lh'], norm_data['rh']])
    data = np.reshape(data, (len(data), 1))
    stc_norm = mne.SourceEstimate(data, stc.vertices, 0, 0, subject=subject)
    return stc_norm


@utils.tryit(print_only_last_error_line=False)
def calc_source_power_spectrum(
        subject, events, atlas='aparc.DKTatlas40', inverse_method='dSPM', extract_modes=['mean_flip'],
        fmin=1, fmax=120, bandwidth=2., bands=None, max_epochs_num=0,
        mri_subject='', epo_fname='', inv_fname='', snr=3.0, pick_ori=None, apply_SSP_projection_vectors=True,
        add_eeg_ref=True, fwd_usingMEG=True, fwd_usingEEG=True, surf_name='pial', precentiles=(1, 99),
        baseline_times=(None, None), epochs=None, src=None, overwrite=False, do_plot=False, save_tmp_files=False,
        save_vertices_data=False, label_stat='mean', n_jobs=6):
    """

    :param subject: The test subject
    :param events: An array with event codes ?
    :param atlas: The map of brain regions
    :param inverse_method: The source localization method
    :param extract_modes: The method for extracting the time course
    :param fmin: The minimum frequency in Hz
    :param fmax: The maximum frequency in Hz
    :param bandwidth: The bandwidth of the multi taper windowing function in Hz
    :param bands:
    :param max_epochs_num: The maximum number of epochs to use; 0 if no maximum
    :param mri_subject: If different than "subject"
    :param epo_fname: Path to the epochs file
    :param inv_fname: Path to the inverse operator file
    :param snr: The signal to noise ratio
    :param pick_ori: If “normal”, rather than pooling the orientations by taking the norm, only the radial component is
                     kept. This is only implemented when working with loose orientations. If “vector”, no pooling of the
                     orientations is done and the vector result will be returned in the form of a
                     mne.VectorSourceEstimate object. This does not work when using an inverse operator with fixed
                     orientations.
    :param apply_SSP_projection_vectors: Apply the signal space projection (SSP) operators to the data
    :param add_eeg_ref: Re-reference the EEG signal
    :param fwd_usingMEG:
    :param fwd_usingEEG:
    :param surf_name: Surface used to obtain vertex locations, e.g., ‘white’, ‘pial’
    :param precentiles:
    :param baseline_times: Times used for baseline correction
    :param epochs: The epochs object
    :param src: The source spaces
    :param overwrite: Overwrite existing power spectrum
    :param do_plot: Plot the power spectrum
    :param save_tmp_files:
    :param save_vertices_data: Save the indices of the dipoles in the source space
    :param label_stat:
    :param n_jobs: The number of jobs to run in parallel
    :return: True
    """
    if isinstance(events, str):
        events = {events: 1}
    if mri_subject == '':
        mri_subject = subject
    if inv_fname == '':
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    if epochs is None:
        epo_fname = get_epo_fname(epo_fname)
    if isinstance(extract_modes, str):
        extract_modes = [extract_modes]
    label_stat = getattr(np, label_stat)
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) else ['all']
    lambda2 = 1.0 / snr ** 2
    fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg'))
    plots_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'meg', 'plots'))
    tmp_fol = utils.make_dir(op.join(fol, 'labels_power'))
    power_spectrum, power_spectrum_baseline = None, None
    first_time = True
    labels = None
    freqs = None
    freqs_bins = np.arange(fmin, fmax)
    for (cond_ind, cond_name), em in product(enumerate(events_keys), extract_modes):
        vertices_data = {hemi: dict() for hemi in utils.HEMIS}
        vertices_baseline_data = {hemi: dict() for hemi in utils.HEMIS}
        vertices_data_fname = op.join(fol, '{}_{}_{}_vertices_power_spectrum.pkl'.format(cond_name, inverse_method, em))
        output_fname = op.join(fol, '{}_{}_{}_power_spectrum.npz'.format(cond_name, inverse_method, em))
        if op.isfile(output_fname) and not overwrite:
            print('{} already exist'.format(output_fname))
            d = np.load(output_fname)
            if do_plot:
                labels = lu.read_labels(mri_subject, SUBJECTS_MRI_DIR, atlas, surf_name=surf_name, n_jobs=n_jobs)
                plot_psds(subject, d['power_spectrum'], d['frequencies'], labels, cond_ind, cond_name, plots_fol)
            if not (baseline_times[0] is None and baseline_times[1] is None) and \
                    'power_spectrum_baseline' not in d or d['power_spectrum_baseline'] is None:
                print('Baseline needed to be computed')
            else:
                continue

        if epochs is None:
            epo_cond_fname = get_cond_fname(epo_fname, cond_name)
            print('Reading epochs from {}'.format(epo_cond_fname))
            if not op.isfile(epo_cond_fname):
                print('single_trial_stc and not epochs file was found! ({})'.format(epo_cond_fname))
                return False
            epochs = mne.read_epochs(epo_cond_fname, apply_SSP_projection_vectors)  # , preload=False) # add_eeg_ref
            epochs_times = (None, 1)  # todo: should be None, None!!!
            epochs.crop(epochs_times[0], epochs_times[1])
            if not (baseline_times[0] is None and baseline_times[1] is None):
                baseline = epochs.copy().crop(baseline_times[0], baseline_times[1])
                epochs = epochs.crop(baseline_times[1], None)
            else:
                baseline = None

        if first_time:
            first_time = False
            labels = lu.read_labels(mri_subject, SUBJECTS_MRI_DIR, atlas, surf_name=surf_name, n_jobs=n_jobs)
            if len(labels) == 0:
                print('Can\'t find {} labels!'.format(atlas))
                return False
            inverse_operator, src = get_inv_src(inv_fname, src)

        try:
            mne.set_eeg_reference(epochs, ref_channels=None)
            epochs.apply_proj()
        except:
            print('annot create EEG average reference projector (no EEG data found)')
        if inverse_operator is None:
            inverse_operator, src = get_inv_src(inv_fname, src, cond_name)

        now = time.time()
        epochs_num = min(max_epochs_num, len(epochs)) if max_epochs_num != 0 else len(epochs)

        for label_ind, label in enumerate(labels):
            utils.time_to_go(now, label_ind, len(labels), 1)
            _, src_sel = mne.minimum_norm.inverse.label_src_vertno_sel(label, inverse_operator['src'])
            if len(src_sel) == 0:
                continue
            stcs = mne.minimum_norm.compute_source_psd_epochs(
                epochs, inverse_operator, lambda2=lambda2, method=inverse_method, fmin=fmin, fmax=fmax,
                bandwidth=bandwidth, label=label, return_generator=True, n_jobs=n_jobs)
            if baseline is not None:
                baseline_stcs = mne.minimum_norm.compute_source_psd_epochs(
                    baseline, inverse_operator, lambda2=lambda2, method=inverse_method, fmin=fmin, fmax=fmax,
                    bandwidth=bandwidth, label=label, return_generator=True, n_jobs=n_jobs)
            else:
                baseline_stcs = [None] * epochs_num
            for epoch_ind, (stc, baseline_stc) in enumerate(zip(stcs, baseline_stcs)):
                if epoch_ind >= epochs_num:
                    break
                freqs = stc.times
                baseline_freqs = baseline_stc.times if baseline is not None else None
                label_vertices = stc.lh_vertno if label.hemi == 'lh' else stc.rh_vertno
                if power_spectrum is None:
                    power_spectrum = np.zeros((epochs_num, len(labels), len(freqs_bins), len(events)))
                    if baseline is not None:
                        power_spectrum_baseline = np.zeros((epochs_num, len(labels), len(freqs_bins), len(events)))
                label_power_spectrum = bin_power_spectrum(label_stat(stc.data, axis=0), freqs, freqs_bins)
                label_power_spectrum = 10 * np.log10(label_power_spectrum)  # dB/Hz
                power_spectrum[epoch_ind, label_ind, :, cond_ind] = label_power_spectrum
                if baseline is not None:
                    label_baseline_power_spectrum = bin_power_spectrum(
                        label_stat(baseline_stc.data, axis=0), baseline_freqs, freqs_bins)
                    label_baseline_power_spectrum = 10 * np.log10(label_baseline_power_spectrum)  # dB/Hz
                    power_spectrum_baseline[epoch_ind, label_ind, :, cond_ind] = label_baseline_power_spectrum
                if save_vertices_data:
                    if epoch_ind == 0:
                        for vert_ind, vert_no in enumerate(label_vertices):
                            vertices_data[label.hemi][vert_no] = np.zeros((epochs_num, len(freqs)))
                            if baseline is not None:
                                vertices_baseline_data[label.hemi][vert_no] = np.zeros(
                                    (epochs_num, len(baseline_freqs)))
                    for vert_ind, vert_no in enumerate(label_vertices):
                        vertices_data[label.hemi][vert_no][epoch_ind] = stc.data[vert_ind]
                        if baseline is not None:
                            vertices_baseline_data[label.hemi][vert_no][epoch_ind] = baseline_stc.data[vert_ind]
            # if save_tmp_files:
            #     bsp = power_spectrum_baseline[:, label_ind, :, cond_ind] if baseline is not None else None
            #     np.savez(output_fname, power_spectrum=power_spectrum[:, label_ind, :, cond_ind], frequencies=freqs,
            #              label=label.name, cond=cond_name, power_spectrum_basline=bsp, baseline_freqs=baseline_freqs)
            if do_plot:
                plot_label_psd(power_spectrum[:, label_ind, :, cond_ind], freqs, label, cond_name, plots_fol)
        # for hemi in utils.HEMIS:
        #     for vert_no in vertices_data[hemi].keys():
        #         vertices_data[hemi][vert_no]
        if save_vertices_data:
            utils.save((vertices_data, vertices_baseline_data, freqs, baseline_freqs), vertices_data_fname)
        bsp = power_spectrum_baseline if baseline is not None else None
        np.savez(output_fname, power_spectrum=power_spectrum, frequencies=freqs, power_spectrum_baseline=bsp,
                 baseline_frequencies=baseline_freqs)

    if save_vertices_data:
        calc_vertices_data_power_bands(subject, events, mri_subject, inverse_method, extract_modes, vertices_data,
                                       freqs)
    # calc_labels_power_bands(
    #     mri_subject, atlas, events, inverse_method, extract_modes, precentiles, bands, labels, overwrite, n_jobs=n_jobs)
    return True


def bin_power_spectrum(power_spectrum, frequencies, freqs_bins):
    round_freqs = np.round(frequencies)
    bin_powers = np.array([power_spectrum[np.where(round_freqs == f)[0]].mean(0) for f in freqs_bins])
    # bin_powers[np.where(np.isnan(bin_powers))] = np.nanmin(bin_powers)
    return bin_powers


def calc_vertices_data_power_bands(
        subject, events, mri_subject='', inverse_method='dSPM', extract_modes=['mean_flip'],
        vertices_data=None, freqs=None, bands=None, overwrite=False):
    if mri_subject == '':
        mri_subject = subject
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) else ['all']
    fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg'))
    if bands is None:
        bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])

    results_num = 0
    for (cond_ind, cond_name), em in product(enumerate(events_keys), extract_modes):
        if vertices_data is None or freqs is None:
            vertices_data_fname = op.join(fol,
                                          '{}_{}_{}_vertices_power_spectrum.pkl'.format(cond_name, inverse_method, em))
            if not op.isfile(vertices_data_fname):
                print('Can\'t find {}!'.format(vertices_data_fname))
                continue
            vertices_data, freqs = utils.load(vertices_data_fname)
        for band, (lf, hf) in bands.items():
            stc_fname = op.join(fol, '{}_{}_{}_{}_power'.format(cond_name, inverse_method, em, band))
            if utils.both_hemi_files_exist('{}-{}.stc'.format(stc_fname, '{hemi}')) and not overwrite:
                continue
            band_mask = np.where((freqs >= lf) & (freqs <= hf))[0]
            data = {
                hemi: [vertices_data[hemi][vert_ind][:, band_mask].mean() for vert_ind in vertices_data[hemi].keys()]
                for hemi in utils.HEMIS}
            stc_power = creating_stc_obj(data, vertices_data, subject)
            print('Saving power stc to: {}'.format(stc_fname))
            stc_power.save(stc_fname)
            results_num += 1 if utils.both_hemi_files_exist('{}-{}.stc'.format(stc_fname, '{hemi}')) else 0
    return results_num == len(events_keys) * len(bands)


def creating_stc_obj(data_dict, vertno_dict, subject, tmin=0, tstep=0):
    vertno, data = {}, {}
    for hemi in utils.HEMIS:
        if isinstance(vertno_dict, list) and isinstance(vertno_dict[0], np.ndarray):
            verts_indices = vertno_dict[0] if hemi == 'lh' else vertno_dict[1]
        elif isinstance(vertno_dict, dict) and 'lh' in vertno_dict and 'rh' in vertno_dict:
            if isinstance(vertno_dict[hemi], np.ndarray):
                verts_indices = vertno_dict[hemi]
            elif isinstance(vertno_dict[hemi], list):
                verts_indices = np.array(vertno_dict[hemi])
            elif isinstance(vertno_dict[hemi], dict):
                verts_indices = np.array(list(vertno_dict[hemi].keys()))
            else:
                raise Exception('Wrong type of vertno!s')
        else:
            raise Exception('Wrong type of vertno!s')
        indices_ord = np.argsort(verts_indices)
        vertno[hemi] = verts_indices[indices_ord]
        data[hemi] = np.array(data_dict[hemi])[indices_ord]
    data = np.concatenate([data['lh'], data['rh']])
    if data.ndim == 1:
        data = np.reshape(data, (len(data), 1))
    vertices = [vertno['lh'], vertno['rh']]
    stc = mne.SourceEstimate(data, vertices, tmin, tstep, subject=subject)
    return stc


def plot_psds(subject, power_spectrum, freqs, labels, cond_ind, cond_name, plots_fol):
    print('Saving plots in {}'.format(plots_fol))
    for label_ind, label in enumerate(labels):
        psd = power_spectrum[:, label_ind, :, cond_ind]
        plot_label_psd(psd, freqs, label, cond_name, plots_fol)


def plot_label_psd(psd, freqs, label, cond_name, plots_fol):
    import matplotlib.pyplot as plt
    psd_mean = psd.mean(0)
    psd_std = psd.std(0)
    plt.plot(freqs, psd_mean, color='k')
    plt.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std, color='k', alpha=.5)
    plt.title('{} {} Multitaper PSD'.format(label.name, cond_name))
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density (dB)')
    plt.savefig(op.join(plots_fol, 'psd_{}_{}.jpg'.format(label.name, cond_name)))
    plt.close()


def calc_labels_induced_power(subject, atlas, events, inverse_method='dSPM', extract_modes=['mean_flip'],
                              bands=None, max_epochs_num=0, average_over_label_indices=True, n_cycles=7.0,
                              mri_subject='', epo_fname='',
                              inv_fname='', snr=3.0, pick_ori='normal', apply_SSP_projection_vectors=True,
                              add_eeg_ref=True,
                              fwd_usingMEG=True, fwd_usingEEG=True, epochs=None, overwrite=False, n_jobs=6):
    if mri_subject == '':
        mri_subject = subject
    if inv_fname == '':
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    if not op.isfile(inv_fname):
        raise Exception('Can\'t find the inverse file! {}'.format(inv_fname))
    inverse_operator, _ = get_inv_src(inv_fname)
    epo_fname = get_epo_fname(epo_fname)
    if isinstance(extract_modes, str):
        extract_modes = [extract_modes]
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) and len(events) > 0 \
        else ['all']
    lambda2 = 1.0 / snr ** 2
    if bands is None:
        bands = dict(theta=np.arange(4, 8, 1), alpha=np.arange(8, 15, 1),  # delta=np.arange(1, 4, 1),
                     beta=np.arange(15, 30, 2), gamma=np.arange(30, 55, 3), high_gamma=np.arange(65, 120, 5))
    labels = lu.read_labels(mri_subject, SUBJECTS_MRI_DIR, atlas, n_jobs=n_jobs)
    if len(labels) == 0:
        return False

    ws = None
    ret = True
    # fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg', 'labels'))
    fol = utils.make_dir(op.join(MEG_DIR, subject, 'labels_induced_power'))
    for (cond_ind, cond_name), em in product(enumerate(events_keys), extract_modes):
        output_template = op.join(fol, '{}_{}_{}_{}_{}_{}induced_power.npz'.format(
            cond_name, '{label}', atlas, inverse_method, em, '' if average_over_label_indices else 'vertices_'))
        if not overwrite:
            all_done = True
            for label_ind, label in enumerate(labels):
                output_fname = output_template.format(label=label)
                if not op.isfile(output_fname) or overwrite:
                    all_done = False
                    break
            if all_done:
                continue

        epo_cond_fname = get_cond_fname(epo_fname, cond_name)
        if not op.isfile(epo_cond_fname):
            print('single_trial_stc and not epochs file was found! ({})'.format(epo_cond_fname))
            return False
        if epochs is None:
            epochs = mne.read_epochs(epo_cond_fname, apply_SSP_projection_vectors, add_eeg_ref)
        epochs_num = min(max_epochs_num, len(epochs)) if max_epochs_num != 0 else len(epochs)
        if ws is None:
            ws = [(mne.time_frequency.morlet(
                epochs.info['sfreq'], freqs, n_cycles=n_cycles, zero_mean=False)) for freqs in bands.values()]
        label_now = time.time()
        for label_ind, label in enumerate(labels):
            output_fname = output_template.format(label=label.name)
            if op.isfile(output_fname) and not overwrite:
                print('calc_labels_induced_power: {} is already calculated'.format(label.name))
                continue
            utils.time_to_go(label_now, label_ind, len(labels), runs_num_to_print=1)
            powers, times = None, None
            stcs = mne.minimum_norm.apply_inverse_epochs(
                epochs, inverse_operator, lambda2, inverse_method, label, pick_ori=pick_ori, return_generator=True)
            stc_now = time.time()
            for stc_ind, stc in enumerate(stcs):
                utils.time_to_go(stc_now, stc_ind, len(epochs), runs_num_to_print=1)
                if stc_ind >= epochs_num:
                    break
                if powers is None:
                    if average_over_label_indices:
                        powers = np.empty((len(bands), epochs_num, stc.shape[1]))
                    else:
                        powers = np.empty((len(bands), epochs_num, stc.shape[0], stc.shape[1]))
                    times = stc.times

                params = [(stc.data, ws[band_ind], band_ind, average_over_label_indices)
                          for band_ind in range(len(bands.keys()))]
                powers_bands = utils.run_parallel(_calc_tfr_cwt_parallel, params, len(bands.keys()))
                for power_band, band_ind in powers_bands:
                    powers[band_ind, stc_ind] = power_band
            print('calc_labels_induced_power: Saving results in {}'.format(output_fname))
            # powers = 10 * np.log10(powers)
            np.savez(output_fname, label_name=label.name, atlas=atlas, data=power, times=times)
            ret = ret and op.isfile(output_fname)

    return ret


def _calc_tfr_cwt_parallel(p):
    stc_data, ws_band, band_ind, average_over_label_indices = p
    tfr = mne.time_frequency.tfr.cwt(stc_data, ws_band, use_fft=False)
    power = (tfr * tfr.conj()).real
    if average_over_label_indices:
        power = power.mean((0, 1))  # avg over label vertices and band's freqs
    else:
        power = power.mean(1)
    # power = power.mean(0)
    return power, band_ind


def calc_labels_power_bands(mri_subject, atlas, events, inverse_method='dSPM', extract_modes=['mean_flip'],
                            precentiles=(1, 99), bands=None, labels=None, overwrite=False, n_jobs=6):
    meg_fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg'))
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) else ['all']
    if bands is None:
        bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    if labels is None:
        labels = lu.read_labels(mri_subject, SUBJECTS_MRI_DIR, atlas, only_names=True, n_jobs=n_jobs)
    ret = True
    fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, 'labels', 'labels_data'))
    for (cond_ind, cond_name), em in product(enumerate(events_keys), extract_modes):
        input_fname = op.join(meg_fol, '{}_{}_{}_power_spectrum.npz'.format(cond_name, inverse_method, em))
        if not op.isfile(input_fname):
            print('No power specturm for {}!'.format(cond_name))
            continue
        d = utils.Bag(np.load(input_fname))
        psd = d.power_spectrum  # (epochs_num, len(labels), len(freqs), len(events))
        freqs = d.frequencies
        for band, (lf, hf) in bands.items():
            output_fname = op.join(fol, '{}_labels_{}_{}_{}_power.npz'.format(cond_name, inverse_method, em, band))
            if op.isfile(output_fname) and not overwrite:
                return True
            band_mask = np.where((freqs >= lf) & (freqs <= hf))
            band_power = np.empty((len(labels), psd.shape[0]))
            for label_ind, label_name in enumerate(labels):
                band_power[label_ind] = psd[:, label_ind, band_mask, cond_ind].mean(axis=2).squeeze()
            data_max = utils.calc_max(band_power, norm_percs=precentiles)
            print('calc_labels_power_bands: Saving results in {}'.format(output_fname))
            np.savez(output_fname, names=np.array(labels), atlas=atlas, data=band_power,
                     title='labels {} power ({})'.format(band, cond_name), data_min=0, data_max=data_max, cmap='RdOrYl')
            ret = ret and op.isfile(output_fname)

    return ret


def get_modality(fwd_usingMEG, fwd_usingEEG):
    if fwd_usingMEG and fwd_usingEEG:
        modality = 'meeg'
    elif fwd_usingMEG and not fwd_usingEEG:
        modality = 'meg'
    elif not fwd_usingMEG and fwd_usingEEG:
        modality = 'eeg'
    else:
        raise Exception('fwd_usingMEG and fwd_usingEEG are False!')
    return modality


@utils.tryit(throw_exception=True)
@check_globals()
def calc_labels_connectivity(
        subject, atlas, events, mri_subject='', subjects_dir='', mmvt_dir='', inverse_method='dSPM',
        epo_fname='', inv_fname='', raw_fname='', snr=3.0, pick_ori=None, apply_SSP_projection_vectors=True,
        add_eeg_ref=True, fwd_usingMEG=True, fwd_usingEEG=True, extract_modes=['mean_flip'], surf_name='pial',
        con_method='coh', con_mode='cwt_morlet', cwt_n_cycles=7, max_epochs_num=0, min_order=1, max_order=100,
        estimate_order=False, windows_length=0, windows_shift=0, calc_only_granger_causality_likelihood=False,
        overwrite_connectivity=False, raw=None, epochs=None, src=None, inverse_operator=None, bands=None, labels=None,
        cwt_frequencies=None, con_indentifer='', symetric_con=None, downsample=1, crops_times=None, output_fname='',
        n_jobs=6):
    modality = get_modality(fwd_usingMEG, fwd_usingEEG)
    if mri_subject == '':
        mri_subject = subject
    if subjects_dir == '':
        subjects_dir = SUBJECTS_MRI_DIR
    if mmvt_dir == '':
        mmvt_dir = MMVT_DIR
    if inv_fname == '':
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    if epochs is None:
        epo_fname = get_epo_fname(epo_fname)
    if isinstance(extract_modes, str):
        extract_modes = [extract_modes]
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) and len(events) > 0 \
        else ['all']
    if symetric_con is None:
        symetric_con = con_method not in ['gc']
    lambda2 = 1.0 / snr ** 2
    if bands is None or bands == '':
        bands = utils.calc_bands(1, 120)
        # bands = [[1, 4], [4, 8], [8, 15], [15, 30], [30, 55], [65, 120]]
    if cwt_frequencies is None or cwt_frequencies == '':
        cwt_frequencies = np.arange(4, 120, 2)
    ret = True
    first_time = True

    for cond_name, em in product(events_keys, extract_modes):
        connectivity_template = connectivity_preproc.get_output_fname(
            subject, con_method, modality, em, '{}_{}_{}'.format('{band_name}', cond_name, con_indentifer))
        files_exist = all([op.isfile(connectivity_template.format(band_name=band)) for band in bands.keys()])
        if files_exist and not overwrite_connectivity:
            print('Connectivity files already exist ({})'.format(connectivity_template))
            continue

        if first_time:
            first_time = False
            if labels is None:
                labels = lu.read_labels(mri_subject, subjects_dir, atlas, surf_name=surf_name, n_jobs=n_jobs)
                if len(labels) == 0:
                    print('No labels!')
                    return False
            labels_names = [l.name for l in labels]
            if inverse_operator is None or src is None:
                inverse_operator, src = get_inv_src(inv_fname, src)
            if inverse_operator is None or src is None:
                print('Can\'t find the inverse_operator!')

        if epochs is None:
            epo_cond_fname = get_cond_fname(epo_fname, cond_name)
            if not op.isfile(epo_cond_fname):
                print('single_trial_stc and not epochs file was found! ({})'.format(epo_cond_fname))
                return False
            epochs = mne.read_epochs(epo_cond_fname, apply_SSP_projection_vectors, add_eeg_ref)
        if type(epochs) is mne.Evoked or type(epochs) is mne.EvokedArray:
            epochs = evoked_to_epochs(epochs)
        if crops_times is not None:
            epochs = epochs.crop(crops_times[0], crops_times[1])
        sfreq = epochs.info['sfreq']
        try:
            mne.set_eeg_reference(epochs, ref_channels=None)
            epochs.apply_proj()
        except:
            print('annot create EEG average reference projector (no EEG data found)')
        if inverse_operator is None:
            inverse_operator, src = get_inv_src(inv_fname, src, cond_name)
        if max_epochs_num > 0:
            epochs = epochs[:max_epochs_num]
        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori, return_generator=False)
        con_indentifer = '' if con_indentifer == '' else '_{}'.format(con_indentifer)
        for con_data, band_name in calc_stcs_spectral_connectivity(
                stcs, labels, src, em, bands, con_method, con_mode, sfreq, cwt_frequencies, cwt_n_cycles,
                connectivity_template, min_order, max_order, estimate_order, downsample, windows_length, windows_shift,
                calc_only_granger_causality_likelihood, overwrite_connectivity, n_jobs):
            if output_fname == '':
                output_fname = connectivity_template.format(band_name=band_name)
            connectivity_preproc.save_connectivity(
                subject, con_data, atlas, con_method, connectivity_preproc.ROIS_TYPE, labels_names, [cond_name],
                output_fname, norm_by_percentile=True, norm_percs=[1, 99], symetric_colors=True, labels=labels,
                symetric_con=symetric_con)
            del con_data
            ret = ret and op.isfile(output_fname)
    return ret


def evoked_to_epochs(evoked):
    if type(evoked) is mne.EvokedArray:
        evoked = evoked[0]
    C, T = evoked.data.shape
    return mne.EpochsArray(
        evoked.data.reshape((1, C, T)), evoked.info, np.array([[0, 0, 1]]), evoked.times[0], 1)[0]


def save_connectivity(subject, atlas, events, modality='meg', extract_modes=['mean_flip'],
                      con_method='coh', con_indentifer='', bands=None, labels=None, symetric_con=None,
                      norm_percs=[1, 99],
                      reduce_to_3d=False, overwrite=False, n_jobs=4):
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) and len(events) > 0 \
        else ['all']
    if bands is None:
        bands = utils.calc_bands(1, 120, include_all_freqs=True)
    if labels is None:
        labels = lu.read_labels(subject, SUBJECTS_MRI_DIR, atlas, n_jobs=n_jobs)
        if len(labels) == 0:
            print('No labels!')
            return False
    labels_names = [l.name for l in labels]
    if symetric_con is None:
        symetric_con = con_method not in ['gc']
    con_indentifer = '' if con_indentifer == '' else '_{}'.format(con_indentifer)
    for cond_name, em, band_name in product(events_keys, extract_modes, bands.keys()):
        output_fname = connectivity_preproc.get_output_fname(
            subject, con_method, modality, em, '{}_{}{}'.format(band_name, cond_name, con_indentifer))
        if op.isfile(output_fname) and not overwrite:
            continue
        tmp_con_input_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_{}_{}_{}.npy'.format(
            con_method, band_name, cond_name, con_indentifer))
        if not op.isfile(tmp_con_input_fname):
            print('No tmp connectivity file for {} {} {} {}!'.format(con_method, band_name, cond_name, con_indentifer))
            continue
        con_data = np.load(tmp_con_input_fname)
        connectivity_preproc.save_connectivity(
            subject, con_data, atlas, con_method, connectivity_preproc.ROIS_TYPE, labels_names, [cond_name],
            output_fname, norm_by_percentile=True, norm_percs=norm_percs, symetric_colors=True, labels=labels,
            symetric_con=symetric_con, reduce_to_3d=reduce_to_3d)


def calc_stcs_spectral_connectivity(
        stcs, labels, src, em, bands, con_method, con_mode, sfreq, cwt_frequencies,
        cwt_n_cycles, connectivity_template, min_order=1, max_order=100, estimate_order=False, downsample=1,
        windows_length=0, windows_shift=0, calc_only_granger_causality_likelihood=False, overwrite=False, n_jobs=1):
    label_ts = mne.extract_label_time_course(stcs, labels, src, mode=em, allow_empty=True, return_generator=False)
    # We don't care here about the units, so make label_ts numbers with one digit before the decimal point
    label_ts = [x * 10 ** (-np.rint(np.log10(np.max(x))).astype(float)) for x in label_ts]
    if downsample > 1:
        label_ts = [utils.downsample_2d(ts, downsample) for ts in label_ts]
        sfreq /= downsample
    if con_method == 'gc':
        bands['all'] = [None, None]
    for band_ind, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        output_fname = connectivity_template.format(band_name=band_name)
        if op.isfile(output_fname) and not overwrite:
            print('{} already exist'.format(output_fname))
            continue
        if con_method == 'gc':  # granger-causality
            con = granger_causality(
                label_ts, sfreq, max_order, min_order, estimate_order, fmin, fmax, windows_length,
                windows_shift, calc_only_granger_causality_likelihood, n_jobs > 1)
        else:
            con, _, _, _, _ = spectral_connectivity(
                label_ts, con_method, con_mode, sfreq, fmin, fmax, faverage=True, mt_adaptive=True,
                cwt_frequencies=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=n_jobs)
            con = con.squeeze()
        yield con, band_name


def calc_labels_connectivity_from_stc(subject, atlas, events, stc_name, meg_file_with_info, mri_subject='',
                                      subjects_dir='', mmvt_dir='', inv_fname='', fwd_usingMEG=True, fwd_usingEEG=True,
                                      extract_modes=['mean_flip'],
                                      surf_name='pial', con_method='coh', con_mode='cwt_morlet', cwt_n_cycles=7,
                                      overwrite_connectivity=False,
                                      src=None, bands=None, cwt_frequencies=None, windows_length=0.2, windows_shift=0.1,
                                      connectivity_modality='meg', do_plot=False, n_jobs=6):
    if mri_subject == '':
        mri_subject = subject
    if subjects_dir == '':
        subjects_dir = SUBJECTS_MRI_DIR
    if mmvt_dir == '':
        mmvt_dir = MMVT_DIR
    if inv_fname == '':
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    first_time = True
    if cwt_frequencies is None or cwt_frequencies == '':
        cwt_frequencies = np.arange(4, 120, 2)
    if bands is None or bands == '':
        bands = OrderedDict(
            delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    info = load_file_for_info(meg_file_with_info)
    stc_fol = op.join(mmvt_dir, mri_subject, 'meg')
    stc_fname = op.join(stc_fol, '{}-rh.stc'.format(stc_name))
    if not op.isfile(stc_fname):
        print('{} can\'t be found in {}!'.format(stc_fol))
        return False
    stc = mne.read_source_estimate(stc_fname)
    fol = utils.make_dir(op.join(mmvt_dir, mri_subject, 'connectivity'))
    ret = True
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) and len(events) > 0 \
        else ['all']
    for cond_name, em in product(events_keys, extract_modes):
        output_fname = op.join(fol, '{}_{}_{}_{}.npz'.format(cond_name, em, con_method, con_mode))
        if op.isfile(output_fname) and not overwrite_connectivity:
            print('{} already exist'.format(output_fname))
            continue
        if first_time:
            first_time = False
            labels = lu.read_labels(mri_subject, subjects_dir, atlas, surf_name=surf_name, n_jobs=n_jobs)
            _, src = get_inv_src(inv_fname, src)
            if src is None:
                print('Source is None!')
                return False
        windows = calc_windows(stc, windows_length, windows_shift)
        first_w = True
        all_con = None
        now = time.time()
        for w_ind, w in enumerate(windows):
            utils.time_to_go(now, w_ind, len(windows), 1)
            stc_w = stc.copy().crop(w[0], w[1])
            print('Calc spectral_connectivity for {}'.format(stc_w))
            con, freqs, times, n_epochs, n_tapers = calc_stcs_spectral_connectivity(
                [stc_w], labels, src, em, bands, con_method, con_mode, info['sfreq'], cwt_frequencies, cwt_n_cycles,
                n_jobs)
            if do_plot:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(np.mean(con, axis=(0, 1)).T)
                plt.savefig(op.join(fol, 'figures', 'win{}'.format(w_ind)))
            # Average over time
            con = np.mean(con, axis=3)
            for band_ind, band_name in enumerate(bands.keys()):
                print('band:{} min: {} max: {}, mean: {}'.format(
                    band_name, np.nanmin(con[:, :, band_ind]), np.nanmax(con[:, :, band_ind]),
                    np.nanmean(con[:, :, band_ind])))
            if first_w:
                all_con = np.zeros((con.shape[0], con.shape[1], con.shape[2], len(windows)))
                first_w = False
            all_con[:, :, :, w_ind] = con
        if all_con is not None:
            np.savez(output_fname, con=all_con, freqs=freqs, times=times, n_epochs=n_epochs, n_tapers=n_tapers,
                     names=[l.name for l in labels], windows=windows)
            ret = ret and op.isfile(output_fname)
        else:
            ret = False

    first = True
    for band_ind, band_name in enumerate(bands.keys()):
        mmvt_connectivity_output_fname = connectivity_preproc.get_output_fname(
            con_method, em, band_name)
        if op.isfile(mmvt_connectivity_output_fname):
            continue
        if first:
            d = utils.Bag(np.load(output_fname))
            first = False
        connectivity_preproc.save_connectivity(
            subject, d.con[:, :, band_ind, :], atlas, con_method, connectivity_preproc.ROIS_TYPE, d.names, events_keys,
            mmvt_connectivity_output_fname, norm_by_percentile=True, norm_percs=[1, 99],
            symetric_colors=True)
        ret = ret and op.isfile(mmvt_connectivity_output_fname)

    ret = op.isfile(output_fname)
    return ret


def calc_windows(stc, windows_length=0.1, windows_shift=0.05):
    import math
    T = stc.times[-1] - stc.times[0]
    if windows_length == 0:
        windows_length = T
        windows_num = 1
    else:
        windows_num = math.floor((T - windows_length) / windows_shift + 1)
    windows = np.zeros((windows_num, 2))
    for win_ind in range(windows_num):
        windows[win_ind] = [win_ind * windows_shift, win_ind * windows_shift + windows_length]
    return windows + stc.tmin


def load_file_for_info(meg_file_with_info):
    meg_fname = op.join(SUBJECT_MEG_FOLDER, '{}.fif'.format(meg_file_with_info))
    try:
        raw = mne.io.read_raw_fif(meg_fname)
        return raw.info
    except:
        pass
    try:
        evokes = mne.read_evokeds(meg_fname)
        return evokes[0].info
    except:
        pass
    try:
        epochs = mne.read_epochs(meg_fname)
        return epochs.info
    except:
        pass
    raise Exception('Can\'t find {}!'.format(meg_fname))


def spectral_connectivity(label_ts, con_method, con_mode, sfreq, fmin, fmax, faverage=True, mt_adaptive=True,
                          cwt_frequencies=None, cwt_n_cycles=7, n_jobs=1):
    try:
        if con_mode == 'cwt_morlet':
            con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                label_ts, method=con_method, mode=con_mode, sfreq=sfreq, fmin=fmin,
                fmax=fmax, faverage=faverage, mt_adaptive=mt_adaptive,
                cwt_freqs=cwt_frequencies, cwt_n_cycles=cwt_n_cycles, n_jobs=n_jobs)
        elif con_mode == 'multitaper':
            con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                label_ts, method=con_method, mode=con_mode, sfreq=sfreq, fmin=fmin,
                fmax=fmax, faverage=faverage, mt_adaptive=True, n_jobs=n_jobs)
        else:
            con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                label_ts, method=con_method, mode=con_mode, sfreq=sfreq, fmin=fmin,
                fmax=fmax, faverage=True, n_jobs=n_jobs)
        return con, freqs, times, n_epochs, n_tapers
    except:
        print(traceback.format_exc())
        return None, None, None, 0, 0


def granger_causality(
        epochs_ts, sfreq, max_order, min_order=1, estimate_order=False, fmin=None, fmax=None,
        windows_length=0, windows_shift=0, calc_only_likelihood=False, parallel=True,
        check_const_channels=True, max_windows_num=0, take_only_significant_values=True):
    # todo: check other methods:
    # https://towardsdatascience.com/inferring-causality-in-time-series-data-b8b75fe52c46
    # https://github.com/akelleh/causality
    # https://medium.com/@akelleh/causal-inference-with-pandas-dataframes-fc3e64fce5d

    # We don't like to many zeros after the decimal point
    epochs_ts = [x * 10 ** (-np.rint(np.log10(np.max(x))).astype(float)) for x in epochs_ts]
    ijs = []
    N = epochs_ts[0].shape[0]
    # ijs are the indices of the lower-triangle (np.tril_indices) without indices of identical channels
    for i in range(N):
        for j in range(i, N):
            if not np.all(epochs_ts[0][i] == epochs_ts[0][j]):
                ijs.append((i, j))

    params = [(epoch_ts, sfreq, min_order, max_order, estimate_order, fmin, fmax, ijs, windows_length, windows_shift,
               calc_only_likelihood, check_const_channels, max_windows_num, take_only_significant_values)
              for epoch_ts in epochs_ts]
    results = utils.run_parallel(_granger_causality_parallel, params, len(epochs_ts) if parallel else 1)
    res = np.array(results).mean(0)
    return res


def _granger_causality_parallel(p):
    (epoch_ts, sfreq, min_order, max_order, estimate_order, fmin, fmax, ijs, windows_length, windows_shift,
     calc_only_likelihood, check_const_channels, max_windows_num, take_only_significant_values) = p

    C, T = epoch_ts.shape # Channels x Time
    if check_const_channels:
        # Remove constant channels from ijs
        const_c = np.where(np.sum(np.diff(epoch_ts, axis=1), axis=1) == 0)[0]
        if len(const_c) > 0:
            ijs = [ij for ij in ijs if len(set(ij) & set(const_c)) == 0]
    # Initialize the variables
    windows = connectivity_preproc.calc_windows(T, windows_length, windows_shift)
    if max_windows_num > 0 and len(windows) > max_windows_num:
        windows = windows[:max_windows_num]
    W = len(windows)
    ord_values = [None] if estimate_order else range(min_order, max_order + 1)
    gc_values = np.zeros((C, C, W, len(ord_values) if not estimate_order else 1))
    likelihoods = np.empty((C, C, W, max_order if not estimate_order else 1))
    likelihoods.fill(np.inf)
    now1 = time.time()
    max_likelihood_values = np.zeros((C, C, W, 1))

    # Go over all the windows
    for w_ind, (t_from, t_to) in enumerate(windows):
        if len(windows) > 10:
            utils.time_to_go(now1, w_ind, len(windows), 1)
        # Calculate likelihoods
        likelihoods[:, :, w_ind, :] = calc_granger_causality_likelihood(
            epoch_ts, ijs, t_from, t_to, ord_values, max_order)
        # If we only want to calculate the likelihood, not the granger causality
        if calc_only_likelihood:
            # Continue to the next window without calculating the Granger causality values
            max_likelihood_values[:, :, w_ind, :] = calc_max_likelihoods(likelihoods, w_ind)
            continue

        # Calculating the actual Granger Causality for all lags (ord_values)
        now2 = time.time()
        for order in ord_values:
            ord_ind = 0 if order is None else order - 1
            if not estimate_order:
                utils.time_to_go(now2, order, max_order, 1)
                print('Calc granger causality for order {}, {}-{} Hz, {} windows'.format(order, fmin, fmax, len(windows)))
            epoch_gc = calc_epoch_granger_causality(epoch_ts, ijs, t_from, t_to, order, max_order, sfreq, fmin, fmax)
            gc_values[:, :, w_ind, ord_ind] = epoch_gc
    if calc_only_likelihood:
        return max_likelihood_values
    else:
        max_likelihood_gc_values = calc_gc_max_likelihood(likelihoods, gc_values, ijs, take_only_significant_values)
        return max_likelihood_gc_values


def calc_epoch_granger_causality(epoch_ts, ijs, t_from, t_to, order, max_order, sfreq, fmin=None, fmax=None):
    import nitime.timeseries as ts
    import nitime.analysis as nta
    # Create the time window (t_from->t_to)
    time_series = ts.TimeSeries(epoch_ts[:, t_from: t_to], sampling_interval=1 / sfreq)
    # Calculate the Granger Causality for all ijs for a specific lag (order)
    # If we want to estimate the order, order is None, and it's being estimated from z to max_order. `
    # Otherwise, max_order is not being used
    G = nta.GrangerAnalyzer(time_series, order=order, ij=ijs, max_order=max_order)
    if fmin is None and fmax is None:
        freq_idx_G = np.arange(len(G.frequencies))
    elif fmin is None and fmax is not None:
        freq_idx_G = np.where((G.frequencies < fmax))[0]
    elif fmin is not None and fmax is None:
        freq_idx_G = np.where((G.frequencies > fmin))[0]
    else:
        freq_idx_G = np.where((G.frequencies > fmin) * (G.frequencies < fmax))[0]

    g1 = np.mean(G.causality_xy[:, :, freq_idx_G], -1)
    g2 = np.mean(G.causality_yx[:, :, freq_idx_G], -1)
    g1[np.where(np.isnan(g1))] = 0
    g2[np.where(np.isnan(g2))] = 0
    res_all_zeros = False
    # [10, 1] is how 10 influences 1. It's in g1 (lower tri).
    # [1, 10] is how 1 influences 10. It's the [10, 1] value in g2, and [1, 10] in g2.T
    # Therefore, the full influence matrix is g1 + g2.T
    # Remember: The upper tri of g1 and g2 are all zeros
    epoch_gc = g1 + g2.T
    return epoch_gc


def calc_granger_causality_likelihood(epoch_ts, ijs, t_from, t_to, ord_values, max_order):
    from statsmodels.tsa.stattools import grangercausalitytests
    C = epoch_ts.shape[0]  # Channels x Time
    likelihoods = np.empty((C, C, max_order))
    likelihoods.fill(np.inf)
    with warnings.catch_warnings():  # Not really recommended...
        warnings.simplefilter("ignore")
        # ijs is only the lower tri indices, so we need to also use the transformed version
        for switch_i_j in [False, True]:
            if switch_i_j:
                ijs = [(j, i) for (i, j) in ijs]
            for i, j in tqdm(ijs):
                x = np.vstack((epoch_ts[i, t_from: t_to], epoch_ts[j, t_from: t_to])).T
                try:
                    gctest = grangercausalitytests(x, maxlag=max_order, verbose=False)
                except:
                    print('Error while calculting grangercausalitytests for {}-{}'.format(i, j))
                    print(traceback.format_exc())
                    continue
                # Take the likelohood for all the possible orders:
                # lrtest[1] is the stats.chi2.sf(lr, lag), where lr is the likelihood ratio test pvalue,
                # and stats.chi2.sf is the Survival function (1-cdf) of the chi2, the chi-squared continuous
                # random variable.
                for ord_ind, ord_val in enumerate(ord_values):
                    likelihoods[i, j, ord_ind] = gctest[ord_val][0]['lrtest'][1]
    return likelihoods


def calc_gc_max_likelihood(likelihoods, gc_values, ijs, take_only_significant_values):
    C, W = likelihoods.shape[0], likelihoods.shape[2]
    argmin_likehood = np.argmin(likelihoods, axis=3)
    max_likelihood_gc_values = np.zeros((C, C, W, 1))
    for switch_i_j in [False, True]:
        if switch_i_j:
            ijs = [(j, i) for (i, j) in ijs]
        for i, j in ijs:
            for w in range(W):
                if take_only_significant_values and likelihoods[i, j, w, argmin_likehood[i, j, w]] < 0.05 or \
                        not take_only_significant_values:
                    max_likelihood_gc_values[i, j, w, 0] = gc_values[i, j, w, argmin_likehood[i, j, w]]
    return max_likelihood_gc_values


def calc_max_likelihoods(likelihoods, w_ind):
    C = likelihoods.shape[0]
    max_likelihood_res = np.zeros((C, C))
    # Find the lags that gives the min likelihood
    argmin_likehood = np.argmin(likelihoods, axis=3)
    for switch_i_j in [False, True]:
        if switch_i_j:
            ijs = [(j, i) for (i, j) in ijs]
        for i, j in ijs:
            min_pval = likelihoods[i, j, w_ind, argmin_likehood[i, j, w_ind]]
            # Take only the significant values (likelihood < 0.05)
            if min_pval < 0.05:
                # We don't like very small values, and want that bigger is better, so we transform
                # the values from likelihood to surprise by using -log
                max_likelihood_res[i, j] = -np.log10(min_pval)
    return max_likelihood_res


def granger_causality_likelihood(
        data, max_order, windows_length=0, windows_shift=0, check_const_channels=True, max_windows_num=0, ijs=None,
        n_jobs=4):

    if ijs is None:
        ijs = []
        N = data.shape[0]
        for i in range(N):
            for j in range(N):
                if i != j and not np.all(data[i] == data[j]):
                    ijs.append((i, j))
        if check_const_channels:
            const_c = np.where(np.sum(np.diff(data, axis=1), axis=1) == 0)[0]
            if len(const_c) > 0:
                ijs = [ij for ij in ijs if len(set(ij) & set(const_c)) == 0]
    C, T = data.shape # Channels x Time
    windows = connectivity_preproc.calc_windows(T, windows_length, windows_shift)
    if 0 < max_windows_num < len(windows):
        windows = windows[:max_windows_num]
    W = len(windows)
    params = [(data[:, t_from: t_to], w, ijs, max_order)
              for w, (t_from, t_to) in enumerate(windows)]
    max_likelihood_res = np.zeros((C, C, W))
    results = utils.run_parallel(_granger_causality_likelihood_parallel, params, min(len(windows), n_jobs))
    for win_res, w in results:
        max_likelihood_res[:, :, w] = win_res
    return max_likelihood_res


def _granger_causality_likelihood_parallel(p):
    data_window, w, ijs, max_order = p
    C = data_window.shape[0]
    max_likelihood_res = np.zeros((C, C))
    N = len(ijs)

    with warnings.catch_warnings():  # Not really recommended...
        warnings.simplefilter("ignore")
        now = time.time()
        for ind, (i, j) in enumerate(ijs):
            utils.time_to_go(now, ind, N, runs_num_to_print=10, to_hours=True)
            x = np.vstack((data_window[i], data_window[j])).T
            p_vals = stat_utils.calc_granger_causality_likelihood_ratio_p(x, max_order)
            min_pval = np.min(p_vals)
            if min_pval < 0.05:
                max_likelihood_res[i, j] = -np.log10(min_pval)

    return max_likelihood_res, w





# def calc_granger_ij(time_series, order):
#     from nitime import index_utils as iu
#     from nitime import utils as niutils
#     import nitime.algorithms as alg
#     N = time_series.shape[0]
#     x, y = np.meshgrid(np.arange(N), np.arange(N))
#     ij = list(zip(x[iu.tril_indices_from(x, -1)],
#                   y[iu.tril_indices_from(y, -1)]))
#     good_ijs = []
#     for i, j in ij:
#         x1, x2 = time_series[i], time_series[j]
#         lag = order + 1
#         try:
#             Rxx = niutils.autocov_vector(np.vstack([x1, x2]), nlags=lag)
#             alg.lwr_recursion(np.array(Rxx).transpose(2, 0, 1))
#             good_ijs.append((i, j))
#         except:
#             print('error with {} {}'.format(i, j))
#     return good_ijs


def get_inv_src(inv_fname, src=None, cond_name=''):
    if '{cond}' not in inv_fname:
        if not op.isfile(inv_fname):
            # print('No inverse operator found!')
            return None, None
        if src is None:
            inverse_operator = read_inverse_operator(inv_fname)
            src = inverse_operator['src']
    else:
        if not op.isfile(inv_fname.format(cond=cond_name)):
            # print('No inverse operator found!')
            return None, None
        inverse_operator = read_inverse_operator(inv_fname.format(cond=cond_name))
        src = inverse_operator['src']
    return inverse_operator, src


def calc_evokes(epochs, events, mri_subject, normalize_data=False, epo_fname='', evoked_fname='',
                norm_by_percentile=False, norm_percs=None, modality='meg', calc_max_min_diff=True,
                calc_evoked_for_all_epoches=False, overwrite_evoked=False, task='', set_eeg_reference=True,
                average_per_event=True, bad_channels=[]):
    try:
        epo_fname = get_epo_fname(epo_fname)
        if evoked_fname == '':
            evoked_fname = EVO
        # evoked_fname = get_evo_fname(evoked_fname)
        fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, modality_fol(modality)))
        task_str = '{}_'.format(task) if task != '' else ''
        events_keys = list(events.keys())
        # todo: check for all sensors types
        mmvt_files = [op.join(fol, '{}_{}sensors_evoked_data.npy'.format(modality, task_str)),
                      op.join(fol, '{}_{}sensors_evoked_data_meta.npz'.format(modality, task_str)),
                      op.join(fol, '{}_{}sensors_evoked_minmax.npy'.format(modality, task_str))]
        if op.isfile(evoked_fname) and not overwrite_evoked:
            evoked = mne.read_evokeds(evoked_fname)
            if not all([op.isfile(f) for f in mmvt_files]):
                save_evokes_to_mmvt(evoked, events_keys, mri_subject, None, normalize_data, norm_by_percentile,
                                    norm_percs, modality, calc_max_min_diff, task, bad_channels)
            return True, evoked
        if epochs is None:
            epochs = mne.read_epochs(epo_fname)
        if average_per_event and not (len(events_keys) == 1 and events_keys[0] == 'rest'):
            if any([event not in epochs.event_id for event in events_keys]):
                print('Not all the events can be found in the epochs! (events = {})'.format(events_keys))
                events_keys = list(set(epochs.event_id.keys()) & set(events.keys()))
                if len(events_keys) == 0:
                    events_keys = list(set(epochs.event_id.keys()))
                # return False, None
            evokes = [epochs[event].average() for event in events_keys]  # if event in list(epochs.event_id.keys())]
        else:
            evokes = [epochs.average()]
            keys = list(events.keys())
            key = utils.select_one_file(keys)
            evokes[0].comment = key
        if set_eeg_reference:
            try:
                for evoked in evokes:
                    mne.set_eeg_reference(evoked, ref_channels=None)
                    evoked.apply_proj()
            except:
                print("_calc_inverse_operator: Can't add eeg_reference to the evoked")

        if calc_evoked_for_all_epoches:
            epochs_all = mne.concatenate_epochs([epochs[event] for event in events_keys])
            evokes_all = epochs_all.average()
            evokes_all_fname = op.join(utils.get_parent_fol(evoked_fname), '{}-all-eve.fif'.format(mri_subject))
            mne.write_evokeds(evokes_all_fname, evokes_all)
            # save_evokes_to_mmvt(epochs_all, [1], mri_subject, None, normalize_data, norm_by_percentile,
            #                     norm_percs, modality, calc_max_min_diff, task, bad_channels)
        else:
            evokes_all = None
        save_evokes_to_mmvt(evokes, events_keys, mri_subject, evokes_all, normalize_data, norm_by_percentile,
                            norm_percs, modality, calc_max_min_diff, task, bad_channels)
        if '{cond}' in evoked_fname:
            # evokes = {event: epochs[event].average() for event in events_keys}
            for event, evoked in zip(events_keys, evokes):
                mne.write_evokeds(get_cond_fname(evoked_fname, event), evoked)
        else:
            # evokes = [epochs[event].average() for event in events_keys]
            for ev in evokes:
                ev.info['bads'] = bad_channels
            mne.write_evokeds(evoked_fname, evokes)
    except:
        print(traceback.format_exc())
        return False, None
    else:
        if '{cond}' in evoked_fname:
            flag = all([op.isfile(get_cond_fname(evoked_fname, event)) for event in evokes.keys()])
        else:
            flag = op.isfile(evoked_fname)
    return flag, evokes


def save_evokes_to_mmvt(evokes, events_keys, mri_subject, evokes_all=None, normalize_data=False,
                        norm_by_percentile=False, norm_percs=None, modality='meg', calc_max_min_diff=True, task='',
                        bad_channels=[]):
    fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, modality))
    first_evokes = evokes if isinstance(evokes, mne.evoked.EvokedArray) else evokes[0]
    info = first_evokes.info
    for c in bad_channels:
        if c not in info['bads'] and c in info['ch_names']:
            info['bads'].append(c)
    picks, sensors_picks, ch_names, channels_sensors_dict = get_sensros_info(
        mri_subject, modality, info, first_evokes.ch_names)

    task_str = '{}_'.format(task) if task != '' else 'all_'
    # sensors_meta = utils.Bag(np.load(op.join(fol, '{}_sensors_positions.npz'.format(modality))))
    dt = np.diff(first_evokes.times[:2])[0]
    if isinstance(evokes, mne.evoked.EvokedArray):
        data = evokes.data[picks]
    else:
        data_shape = len(events_keys) + (0 if evokes_all is None else 1)
        data = np.zeros((len(picks), first_evokes.data.shape[1], data_shape))
        for event_ind, event in enumerate(events_keys):
            data[:, :, event_ind] = evokes[event_ind].data[picks]
        if evokes_all is not None:
            data[:, :, data_shape - 1] = evokes_all.data[picks]
    if normalize_data:
        data = utils.normalize_data(data, norm_by_percentile, norm_percs)
    else:
        factor = 6 if modality == 'eeg' else 12  # micro V for EEG, fT (Magnetometers) and fT/cm (Gradiometers) for MEG
        data *= np.power(10, factor)
    data = data.squeeze()
    if calc_max_min_diff and len(events_keys) == 2:
        data_diff = np.diff(data) if len(events_keys) > 1 else np.squeeze(data)
        data_max, data_min = utils.get_data_max_min(data_diff, norm_by_percentile, norm_percs)
    else:
        data_max, data_min = utils.get_data_max_min(data, norm_by_percentile, norm_percs)
    max_abs = utils.get_max_abs(data_max, data_min)
    if evokes_all is not None:
        events_keys.append('all')
    np.save(op.join(fol, '{}_{}sensors_evoked_data.npy'.format(modality, task_str)), data)
    np.savez(op.join(fol, '{}_{}sensors_evoked_data_meta.npz'.format(modality, task_str)),
             names=np.array(ch_names), conditions=events_keys, dt=dt, picks=sensors_picks,
             channels_sensors_dict=channels_sensors_dict)
    np.save(op.join(fol, '{}_{}sensors_evoked_minmax.npy'.format(modality, task_str)),
            [-max_abs, max_abs])


def get_sensros_info(subject, modality, info, all_ch_names):
    sensors_ch_names = []
    if modality in ['meg', 'meeg']:
        sensors_picks, channels_sensors_dict = {}, {}
        picks = mne.pick_types(info, meg=True, eeg=False, exclude='bads')
        for sensor_type in ['mag', 'planar1', 'planar2']:
            # Load sensors info
            input_fname = op.join(op.join(
                MMVT_DIR, subject, modality_fol(modality), 'meg_{}_sensors_positions.npz'.format(sensor_type)))
            if not op.isfile(input_fname):
                print('Can\'t find the sensors info! Please run the create_helmet_mesh function first')
                return False
            sensors_info = np.load(input_fname)
            sensors_names = sensors_info['names']
            sensors_picks[sensor_type] = mne.pick_types(info, meg=sensor_type, exclude='bads')
            ch_names = [all_ch_names[k].replace(' ', '') for k in sensors_picks[sensor_type]]
            sensors_ch_names.extend(ch_names)
            channels_sensors_dict[sensor_type] = np.array([ch_names.index(s) for s in sensors_names if s in ch_names])
    elif modality == 'eeg':
        # Load sensors info
        input_fname = op.join(op.join(MMVT_DIR, subject, modality_fol(modality), 'eeg_sensors_positions.npz'))
        if not op.isfile(input_fname):
            print('Can\'t find the sensors info! Please run the create_helmet_mesh function first')
            return False
        sensors_info = np.load(input_fname)
        sensors_names = sensors_info['names']
        picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
        ch_names = [all_ch_names[k].replace(' ', '') for k in picks]
        sensors_ch_names.extend(ch_names)
        channels_sensors_dict = np.array([ch_names.index(s) for s in sensors_names if s in ch_names])
        sensors_picks = {
            sensor_type: mne.pick_types(info, meg=False, eeg=True, exclude='bads')
            for sensor_type in ['eeg']}
    else:
        raise Exception('The modality {} is not supported! (only eeg/meg)'.format(modality))

    return picks, sensors_picks, sensors_ch_names, channels_sensors_dict


def equalize_epoch_counts(events, method='mintime'):
    if '{cond}' not in EPO:
        epochs = mne.read_epochs(EPO)
    else:
        epochs = []
        for cond_name in events.keys():
            epochs_cond = mne.read_epochs(EPO.format(cond=cond_name))
            epochs.append(epochs_cond)
    mne.epochs.equalize_epoch_counts(epochs, method='mintime')
    if '{cond}' not in EPO == 0:
        epochs.save(EPO)
    else:
        for cond_name, epochs in zip(events.keys(), epochs):
            epochs.save(EPO.format(cond=cond_name))


def find_epoches(raw, picks, events, event_id, tmin, tmax, baseline=(None, 0)):
    # remove events that are not in the events table
    event_id = dict([(k, ev) for (k, ev) in event_id.iteritems() if ev in np.unique(events[:, 2])])
    return mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, proj=True,
                      picks=picks, baseline=baseline, preload=True, reject=None)  # reject can be dict(mag=4e-12)


def check_src_ply_vertices_num(src):
    # check the vertices num with the ply files
    ply_vertices_num = utils.get_ply_vertices_num(op.join(MMVT_DIR, MRI_SUBJECT, 'surf', '{}.pial.ply'))
    if ply_vertices_num is not None:
        print(ply_vertices_num)
        src_vertices_num = [src_h['np'] for src_h in src]
        print(src_vertices_num)
        if not src_vertices_num[0] in ply_vertices_num.values() or \
                not src_vertices_num[1] in ply_vertices_num.values():
            raise Exception("src and ply files doesn't have the same vertices number! {}".format(SRC))
    else:
        print('No ply files to check the src!')


# def make_smoothed_forward_solution(events, n_jobs=4, usingEEG=True, usingMEG=True):
#     src = create_smooth_src(MRI_SUBJECT)
#     if '{cond}' not in EPO:
#         fwd = _make_forward_solution(src, RAW, EPO, COR, usingMEG, usingEEG, n_jobs=n_jobs)
#         mne.write_forward_solution(FWD_SMOOTH, fwd, overwrite=True)
#     else:
#         for cond in events.keys():
#             fwd = _make_forward_solution(src, RAW, get_cond_fname(EPO, cond), COR, usingMEG, usingEEG, n_jobs)
#             mne.write_forward_solution(get_cond_fname(FWD_SMOOTH, cond), fwd, overwrite=True)
#     return fwd


def create_smooth_src(subject, surface='pial', overwrite=False, fname=SRC_SMOOTH):
    src = mne.setup_source_space(subject, surface=surface, overwrite=overwrite, spacing='all', fname=fname)
    return src


def check_src(mri_subject, recreate_the_source_space=False, recreate_src_spacing='oct6', recreate_src_surface='white',
              remote_subject_dir='', n_jobs=2):
    # https://martinos.org/mne/stable/manual/cookbook.html#anatomical-information
    # src_fname, src_exist = locating_subject_file(SRC, '*-src.fif')
    src_fname = op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem',
                        '{}-{}-{}-src.fif'.format(mri_subject, recreate_src_spacing[:-1], recreate_src_spacing[-1]))
    if not op.isfile(src_fname):
        src_fname = op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem', '{}-{}-{}-src.fif'.format(
            mri_subject, recreate_src_spacing[:3], recreate_src_spacing[3:]))
    if not op.isfile(src_fname):
        src_files = glob.glob(op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem', '*-src.fif'))
        if len(src_files) > 0:
            src_fname = utils.select_one_file(src_files, '', 'source files')
        else:
            src_files = glob.glob(op.join(remote_subject_dir, 'bem', '*-src.fif'))
            if len(src_files) > 0:
                src_fname = utils.select_one_file(src_files, '', 'source files')
    if op.isfile(src_fname) and utils.get_parent_fol(src_fname, 3) != SUBJECTS_MRI_DIR:
        fol = utils.make_dir(op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem'))
        utils.copy_file(src_fname, op.join(fol, utils.namebase_with_ext(src_fname)))
    if not recreate_the_source_space and op.isfile(src_fname):
        src = mne.read_source_spaces(src_fname)
    else:
        if not recreate_the_source_space:
            ans = input("Can't find the source file ({}), recreate it (y/n)? (spacing={}, surface={}) ".format(
                src_fname, recreate_src_spacing, recreate_src_surface))
        if recreate_the_source_space or ans == 'y':
            # oct_name, oct_num = recreate_src_spacing[:3], recreate_src_spacing[-1]
            # prepare_subject_folder(
            #     mri_subject, args.remote_subject_dir, op.join(SUBJECTS_MRI_DIR, mri_subject),
            #     {'bem': '{}-{}-{}-src.fif'.format(mri_subject, oct_name, oct_num)}, args)
            # https://martinos.org/mne/dev/manual/cookbook.html#source-localization
            utils.make_dir(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'bem'))
            try:
                src = mne.setup_source_space(MRI_SUBJECT, spacing=recreate_src_spacing, surface=recreate_src_surface,
                                             overwrite=True, subjects_dir=SUBJECTS_MRI_DIR, n_jobs=n_jobs)
            except:
                # No overwrite keyword (not sure why... versions?)
                src = mne.setup_source_space(MRI_SUBJECT, spacing=recreate_src_spacing, surface=recreate_src_surface,
                                             subjects_dir=SUBJECTS_MRI_DIR, n_jobs=n_jobs)
            if src is not None:
                mne.write_source_spaces(src_fname, src)
        else:
            print(traceback.format_exc())
            raise Exception("Can't calculate the fwd solution without the source")
    return src


def check_bem(mri_subject, remote_subject_dir, subjects_dir='', bem_fname='', recreate_bem_solution=False,
              bem_ico=4, look_for_bem=True, args={}):
    if not op.isdir(subjects_dir):
        subjects_dir = SUBJECTS_MRI_DIR
    if not op.isfile(bem_fname):
        bem_fname = op.join(subjects_dir, mri_subject, 'bem', '{}-5120-5120-5120-bem-sol.fif'.format(MRI_SUBJECT))
    bem_sol = None
    if len(args) == 0:
        args = utils.Bag(
            sftp=False, sftp_username='', sftp_domain='', sftp_password='',
            overwrite_fs_files=False, print_traceback=False, sftp_port=22)
    if look_for_bem:
        bem_fname, bem_exist = locating_subject_file(bem_fname, '*-bem-sol.*')
        if not bem_exist:
            prepare_subject_folder(
                mri_subject, remote_subject_dir, subjects_dir,
                {'bem': ['*-bem-sol.*']}, args, use_subject_anat_folder=True)
            bem_fname, bem_exist = locating_subject_file(bem_fname, '*-bem-sol.*')
    if not op.isfile(bem_fname) or recreate_bem_solution:
        # todo: check if the bem and src has same ico
        bem_prepared = prepare_bem_surfaces(mri_subject, remote_subject_dir, args)
        surfaces = mne.make_bem_model(mri_subject, subjects_dir=subjects_dir, ico=int(bem_ico))
        mne.write_bem_surfaces(op.join(
            subjects_dir, mri_subject, 'bem', '{}-surfaces.fif'.format(mri_subject)), surfaces)
        bem_sol = mne.make_bem_solution(surfaces)
        save_bem_solution(bem_sol, bem_fname)
    if bem_sol is None and op.isfile(bem_fname):
        bem_sol = read_bem_solution(bem_fname)
    return op.isfile(bem_fname), bem_sol


def read_bem_surfaces(mri_subject):
    from mne.io.constants import FIFF
    input_fname = op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem', '{}-surfaces.fif'.format(mri_subject))
    if op.isfile(input_fname):
        surfaces = mne.read_bem_surfaces(input_fname)
        ids = {FIFF.FIFFV_BEM_SURF_ID_BRAIN: 'Brain',
               FIFF.FIFFV_BEM_SURF_ID_SKULL: 'Skull',
               FIFF.FIFFV_BEM_SURF_ID_HEAD: 'Head'}
        print([ids[s['id']] for s in surfaces])
    else:
        print('No {}!'.format(input_fname))


def save_bem_solution(bem_sol, bem_fname):
    try:
        mne.write_bem_solution(bem_fname, bem_sol)
    except:
        try:
            mne.externals.h5io.write_hdf5(bem_fname, bem_sol)
        except:
            print(traceback.format_exc())
            print('Can\'t write the BEM solution!')


def read_bem_solution(bem_fname):
    bem_sol = None
    if utils.file_type(bem_fname) == 'fif':
        bem_sol = mne.read_bem_solution(bem_fname)
    elif utils.file_type(bem_fname) == 'h5':
        bem_sol = mne.externals.h5io.read_hdf5(bem_fname)
        bem_sol = mne.bem.ConductorModel(bem_sol)
    if bem_sol is None:
        raise Exception('Can\'t read the BEM solution! {}'.format(bem_fname))
    return bem_sol


def prepare_bem_surfaces(mri_subject, remote_subject_dir, args):
    def watershed_exist(fol):
        return np.all([op.isfile(op.join(fol, 'watershed', watershed_fname.format(mri_subject)))
                       for watershed_fname in watershed_files])

    bem_files = ['brain.surf', 'inner_skull.surf', 'outer_skin.surf', 'outer_skull.surf']
    watershed_files = ['{}_brain_surface', '{}_inner_skull_surface', '{}_outer_skin_surface',
                       '{}_outer_skull_surface']
    bem_fol = op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem')
    bem_files_exist = np.all([op.isfile(op.join(bem_fol, bem_fname)) for bem_fname in bem_files])
    if not bem_files_exist:
        prepare_subject_folder(
            mri_subject, remote_subject_dir, SUBJECTS_MRI_DIR,
            {'bem': [f for f in bem_files]}, args)
    bem_files_exist = np.all([op.isfile(op.join(bem_fol, bem_fname)) for bem_fname in bem_files])
    watershed_files_exist = watershed_exist(bem_fol)
    if not watershed_files_exist:
        remote_bem_fol = op.join(remote_subject_dir, 'bem')
        watershed_files_exist = watershed_exist(remote_bem_fol)
        if watershed_files_exist:
            utils.make_link(op.join(remote_bem_fol, 'watershed'), op.join(bem_fol, 'watershed'))
    if not bem_files_exist and not watershed_files_exist:
        # os.environ['SUBJECT'] = mri_subject
        print('Running mne_watershed_bem on {}!'.format(mri_subject))
        utils.run_script('mne_watershed_bem')
        err_msg = '''BEM files don't exist, you should create it first using mne_watershed_bem.
            For that you need to open a terminal, define SUBJECTS_DIR, SUBJECT, source MNE, and run
            mne_watershed_bem.
            cshrc: setenv SUBJECT {0}
            basrc: export SUBJECT={0}
            You can take a look here:
            http://perso.telecom-paristech.fr/~gramfort/mne/MRC/mne_anatomical_workflow.pdf '''.format(mri_subject)
        # raise Exception(err_msg)
    watershed_files_exist = watershed_exist(bem_fol)
    surfaces = [op.join(bem_fol, 'watershed', watershed_fname.format(mri_subject))
                for watershed_fname in watershed_files]
    if watershed_files_exist and (not bem_files_exist):
        # Try and read the surfaces
        from src.utils import geometry_utils as gu
        for surf_fname in surfaces:
            if op.isfile(surf_fname):
                gu.read_surface(surf_fname)

        for bem_file, watershed_file in zip(bem_files, watershed_files):
            utils.remove_file(op.join(bem_fol, bem_file))
            utils.copy_file(op.join(bem_fol, 'watershed', watershed_file.format(mri_subject)),
                            op.join(bem_fol, bem_file))
    return all([op.isfile(surf_fname) for surf_fname in surfaces])


def make_forward_solution(
        subject, mri_subject, events=None, raw_fname='', epo_fname='', evo_fname='', fwd_fname='', cor_fname='',
        bad_channels=[],
        usingMEG=True, usingEEG=True, calc_corticals=True, calc_subcorticals=True, sub_corticals_codes_file='',
        recreate_the_source_space=False, recreate_bem_solution=False, bem_ico=4, recreate_src_spacing='oct6',
        recreate_src_surface='white', overwrite_fwd=False, remote_subject_dir='', n_jobs=4, args={}):
    fwd, fwd_with_subcortical = None, None
    raw_fname = get_raw_fname(raw_fname)
    epo_fname = get_epo_fname(epo_fname)
    evo_fname = get_evo_fname(subject, evo_fname)
    fwd_fname = get_fwd_fname(fwd_fname, usingMEG, usingEEG, True)
    cor_fname = get_cor_fname(cor_fname)
    events_keys = events.keys() if events is not None else ['all']
    try:
        src = check_src(mri_subject, recreate_the_source_space, recreate_src_spacing, recreate_src_surface,
                        remote_subject_dir, n_jobs)
        check_src_ply_vertices_num(src)
        bem_exist, bem = check_bem(
            mri_subject, remote_subject_dir, bem_ico=bem_ico, args=args)
        sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
        if '{cond}' not in evo_fname:
            if calc_corticals:
                if overwrite_fwd or not op.isfile(fwd_fname):
                    fwd = _make_forward_solution(
                        mri_subject, src, raw_fname, epo_fname, evo_fname, cor_fname, bad_channels, usingMEG, usingEEG,
                        bem=bem, n_jobs=n_jobs)
                    print('Writing fwd solution to {}'.format(fwd_fname))
                    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
            if calc_subcorticals and len(sub_corticals) > 0:
                # add a subcortical volumes
                if overwrite_fwd or not op.isfile(FWD_SUB):
                    src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
                    fwd_with_subcortical = _make_forward_solution(
                        mri_subject, src_with_subcortical, raw_fname, epo_fname, evo_fname, cor_fname, bad_channels,
                        usingMEG, usingEEG, n_jobs=n_jobs)
                    mne.write_forward_solution(FWD_SUB, fwd_with_subcortical, overwrite=True)
        else:
            for cond in events_keys:
                if calc_corticals:
                    fwd_cond_fname = get_cond_fname(fwd_fname, cond)
                    if overwrite_fwd or not op.isfile(fwd_cond_fname):
                        fwd = _make_forward_solution(
                            mri_subject, src, raw_fname, get_cond_fname(epo_fname, cond),
                            get_cond_fname(evo_fname, cond), cor_fname, bad_channels, usingMEG, usingEEG, n_jobs=n_jobs)
                        mne.write_forward_solution(fwd_cond_fname, fwd, overwrite=True)
                if calc_subcorticals and len(sub_corticals) > 0:
                    # add a subcortical volumes
                    fwd_cond_fname = get_cond_fname(FWD_SUB, cond)
                    if overwrite_fwd or not op.isfile(fwd_cond_fname):
                        src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
                        fwd_with_subcortical = _make_forward_solution(
                            mri_subject, src_with_subcortical, raw_fname, get_cond_fname(epo_fname, cond),
                            get_cond_fname(evo_fname, cond), cor_fname, bad_channels, usingMEG, usingEEG, n_jobs=n_jobs)
                        mne.write_forward_solution(fwd_cond_fname, fwd_with_subcortical, overwrite=True)
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in calculating fwd solution')
        flag = False

    return flag, fwd, fwd_with_subcortical


def make_forward_solution_to_specific_subcortrical(
        events, region, bad_channels=[], usingMEG=True, usingEEG=True, n_jobs=4):
    raise Exception('make_forward_solution_to_specific_subcortrical: Need to reimplement!')
    import nibabel as nib
    aseg_fname = op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    for cond in events.keys():
        src = add_subcortical_volumes(None, [region])
        fwd = _make_forward_solution(
            subject, src, RAW, get_cond_fname(EPO, cond), get_cond_fname(EVO, cond), COR, bad_channels,
            usingMEG, usingEEG, n_jobs=n_jobs)
        mne.write_forward_solution(get_cond_fname(FWD_X, cond, region=region), fwd, overwrite=True)
    return fwd


def make_forward_solution_to_specific_points(
        subject, events, pts, region_name, epo_fname='', evo_fname='', fwd_fname='', bad_channels=[],
        usingMEG=True, usingEEG=True, n_jobs=4):
    from mne.source_space import _make_discrete_source_space

    epo_fname = EPO if epo_fname == '' else epo_fname
    fwd_fname = FWD_X if fwd_fname == '' else fwd_fname

    # Convert to meters
    pts /= 1000.
    # Set orientations
    ori = np.zeros(pts.shape)
    ori[:, 2] = 1.0
    pos = dict(rr=pts, nn=ori)

    # Setup a discrete source
    sp = _make_discrete_source_space(pos)
    sp.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                   dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                   nuse_tri=None, tris=None, type='discrete',
                   seg_name=region_name))

    src = mne.SourceSpaces([sp])
    for cond in events.keys():
        fwd = _make_forward_solution(
            subject, src, RAW, get_cond_fname(epo_fname, cond), get_cond_fname(evo_fname, cond), COR, bad_channels,
            usingMEG, usingEEG, n_jobs)
        mne.write_forward_solution(get_cond_fname(fwd_fname, cond, region=region_name), fwd, overwrite=True)
    return fwd


def _make_forward_solution(subject, src, raw_fname='', epo_fname='', evo_fname='', cor_fname='', bad_channels=[],
                           fwd_usingMEG=True, fwd_usingEEG=True, bem=None, bem_fname='', n_jobs=6):
    if bem is None:
        if bem_fname == '':
            bem_fname = BEM
        if not op.isfile(bem_fname):
            bem_fname = utils.select_one_file(op.join(
                SUBJECTS_MRI_DIR, MRI_SUBJECT, 'bem', '*-bem-sol.fif'), '', 'bem files')
            if op.isfile(bem_fname):
                utils.copy_file(bem_fname, BEM)
            else:
                raise Exception("Can't find the BEM file!")

    info = get_info(
        subject, epo_fname, evo_fname, raw_fname, bad_channels, fwd_usingEEG=fwd_usingEEG, fwd_usingMEG=fwd_usingMEG)
    if info is None:
        raise Exception("Can't find info object for make_forward_solution!")

    bem = bem_fname if bem is None else bem
    try:
        fwd = mne.make_forward_solution(
            info=info, trans=cor_fname, src=src, bem=bem, meg=fwd_usingMEG, eeg=fwd_usingEEG, mindist=5.0,
            n_jobs=n_jobs)  # , overwrite=True)
    except:
        utils.print_last_error_line()
        print('Trying to create fwd only with MEG')
        fwd = mne.make_forward_solution(
            info=info, trans=cor_fname, src=src, bem=bem_fname, meg=fwd_usingMEG, eeg=False, mindist=5.0,
            n_jobs=n_jobs)  # , overwrite=True)

    return fwd


def add_subcortical_surfaces(src, seg_labels):
    """Adds a subcortical volume to a cortical source space
    """
    from mne.source_space import _make_discrete_source_space

    # Read the freesurfer lookup table
    lut = utils.read_freesurfer_lookup_table()

    # Get the indices to the desired labels
    for label in seg_labels:
        # Get numeric index to label
        seg_name, seg_id = utils.get_numeric_index_to_label(label, lut)
        srf_file = op.join(ASEG, 'aseg_%.3d.srf' % seg_id)
        pts, _, _, _ = utils.read_srf_file(srf_file)

        # Convert to meters
        pts /= 1000.

        # Set orientations
        ori = np.zeros(pts.shape)
        ori[:, 2] = 1.0

        # Store coordinates and orientations as dict
        pos = dict(rr=pts, nn=ori)

        # Setup a discrete source
        sp = _make_discrete_source_space(pos)
        sp.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                       dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                       nuse_tri=None, tris=None, type='discrete',
                       seg_name=seg_name))

        # Combine source spaces
        src.append(sp)

    return src


def add_subcortical_volumes(org_src, seg_labels, spacing=5., use_grid=True):
    """Adds a subcortical volume to a cortical source space
    """
    # Get the subject
    import nibabel as nib
    from mne.source_space import _make_discrete_source_space

    if org_src is not None:
        src = org_src.copy()
    else:
        src = None

    # Find the segmentation file
    aseg_fname = op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, spacing, use_grid, FREESURFER_HOME)
    for pts, seg_name, seg_id in sub_cortical_generator:
        pts = utils.transform_voxels_to_RAS(aseg_hdr, pts)
        # Convert to meters
        pts /= 1000.
        # Set orientations
        ori = np.zeros(pts.shape)
        ori[:, 2] = 1.0

        # Store coordinates and orientations as dict
        pos = dict(rr=pts, nn=ori)

        # Setup a discrete source
        sp = _make_discrete_source_space(pos)
        sp.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                       dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                       nuse_tri=None, tris=None, type='discrete',
                       seg_name=seg_name))

        # Combine source spaces
        if src is None:
            src = mne.SourceSpaces([sp])
        else:
            src.append(sp)

    return src


@check_globals()
def calc_noise_cov(subject, epochs=None, noise_t_min=None, noise_t_max=0, noise_cov_fname='', args=None, raw=None,
                   use_eeg=False, use_meg=False):
    if not use_eeg and not use_meg:
        raise Exception('use_eeg and use_meg are False!')
    if noise_cov_fname == '':
        if use_eeg and use_meg:
            noise_cov_fname = NOISE_COV_MEEG
        else:
            noise_cov_fname = NOISE_COV_MEG if use_meg else NOISE_COV_EEG
    if not op.isfile(noise_cov_fname):
        noise_cov_fname = utils.select_one_file(glob.glob(
            op.join(MEG_DIR, subject, '*noise-cov.fif')), default_fname=noise_cov_fname)
    if op.isfile(noise_cov_fname):
        print('Reading noise cov: {}'.format(noise_cov_fname))
        noise_cov = mne.read_cov(noise_cov_fname)
        return noise_cov
    if epochs is None:
        noise_cov = recalc_epochs_for_noise_cov(subject, noise_t_min, noise_t_max, args, raw)
        noise_cov.save(noise_cov_fname)
        return noise_cov
    if len(epochs) > 1:
        # Check if we need to recalc the epoches...
        if noise_t_min is None or noise_t_max is None:
            noise_cov = mne.compute_covariance(epochs)
        elif epochs.tmin <= noise_t_min and epochs.tmax >= noise_t_max:
            noise_cov = mne.compute_covariance(epochs.crop(noise_t_min, noise_t_max))
        else:
            # Yes, we do...
            noise_cov = recalc_epochs_for_noise_cov(subject, noise_t_min, noise_t_max, args, raw)
    else:
        if op.isfile(EPO_NOISE):
            demi_epochs = mne.read_epochs(EPO_NOISE)
        else:
            raise Exception("You should split first your epochs into small demi epochs, see calc_demi_epoches")
        noise_cov = calc_noise_cov(subject, demi_epochs, use_eeg=use_eeg, use_meg=use_meg)
    noise_cov.save(noise_cov_fname)
    return noise_cov


def recalc_epochs_for_noise_cov(subject, noise_t_min, noise_t_max, args, raw=None):
    noise_args = utils.Bag(args.copy())
    noise_args.t_min = noise_t_min
    noise_args.t_max = noise_t_max
    noise_args.overwrite_epochs = False
    noise_args.baseline = (noise_t_min, noise_t_max)
    epo_fname = get_epo_fname(args.epo_fname)
    noise_args.epo_fname = '{}noise-epo.fif'.format(epo_fname[:-len('epo.fif')])
    _, noise_epochs = calc_epochs_wrapper_args(subject, noise_args.conditions, noise_args, raw=raw)
    noise_cov = mne.compute_covariance(noise_epochs)
    return noise_cov


@check_globals()
def get_inv_fname(inv_fname='', fwd_usingMEG=True, fwd_usingEEG=True, create_new=False, **kwargs):
    if fwd_usingMEG and fwd_usingEEG:
        inv_modal_fname = INV_MEEG
    else:
        inv_modal_fname = INV_EEG if fwd_usingEEG else INV_MEG
    if op.isfile(inv_modal_fname) and not op.isfile(inv_fname) == '':
        print('get_inv_fname: using {}'.format(inv_modal_fname))
        return inv_modal_fname
    if create_new:
        return inv_fname if inv_fname != '' else inv_modal_fname
    inv_fname, inv_exist = locating_meg_file(inv_fname, '*inv.fif')
    if op.isfile(inv_modal_fname) and not inv_exist:
        ret = input('Can\'t find {}, do you want to use {} instead? '.format(inv_fname, inv_modal_fname))
        if au.is_true(ret):
            return inv_modal_fname
    if not inv_exist:
        files = glob.glob(op.join(SUBJECT_MEG_FOLDER, '*{}*'.format(inv_fname)))
        if len(files) > 0:
            inv_fname = utils.select_one_file(files)
        raise Exception('Can\'t find the inv fname!')
    return inv_fname


def get_raw_fname(raw_fname='', include_empty=False):
    if raw_fname == '':
        raw_fname = RAW
    if op.isfile(raw_fname):
        return raw_fname
    files = glob.glob(raw_fname)
    if len(files) > 0:
        return utils.select_one_file(files, files_desc='raw')
    print('{}: looking for raw fif file...'.format(inspect.stack()[1][3]))
    exclude_pattern = '*empty*.fif' if not include_empty else ''
    raw_fname, raw_exist = locating_meg_file(raw_fname, '*raw.fif', exclude_pattern=exclude_pattern)
    if not raw_exist:
        files = glob.glob(op.join(MEG_DIR, SUBJECT, '*.fif'))
        if len(files) > 0:
            return utils.select_one_file(files, files_desc='raw')
    return raw_fname if raw_exist else ''


def get_fwd_fname(fwd_fname='', fwd_usingMEG=True, fwd_usingEEG=True, create_new=False):
    if fwd_usingMEG and fwd_usingEEG:
        fwd_modal_fname = FWD_MEEG
    else:
        fwd_modal_fname = FWD_EEG if fwd_usingEEG else FWD_MEG
    if op.isfile(fwd_modal_fname) and fwd_fname in ['', fwd_modal_fname]:
        print('get_fwd_fname: using {}'.format(fwd_modal_fname))
        return fwd_modal_fname
    if create_new:
        return fwd_fname if fwd_fname != '' else fwd_modal_fname
    fwd_fname, fwd_exist = locating_meg_file(fwd_fname, '*fwd.fif')
    if fwd_exist and op.isfile(fwd_modal_fname):
        ret = input('Can\'t find {}, do you want to use {} instead? '.format(fwd_fname, fwd_modal_fname))
        fwd_fname = fwd_modal_fname if au.is_true(ret) else fwd_fname
    return fwd_fname


def get_epo_fname(epo_fname='', load_autoreject_if_exist=False, overwrite=False):
    if overwrite and epo_fname != '':
        return epo_fname
    if epo_fname == '':
        epo_fname = EPO
    epo_exist = False
    if load_autoreject_if_exist:
        epo_fname, epo_exist = locating_meg_file(epo_fname, '*ar-epo.fif')
    if not epo_exist or not load_autoreject_if_exist:
        epo_fname, epo_exist = locating_meg_file(epo_fname, '*epo.fif')
    return epo_fname.strip()


def get_evo_fname(subject, evo_fname=''):
    if op.isfile(evo_fname):
        return evo_fname
    if evo_fname == '':
        evo_fname = EVO
    if '{subject}' in evo_fname:
        evo_fname = evo_fname.format(subject=subject)
    files = glob.glob(evo_fname)
    if len(files) > 0:
        evo_fname = utils.select_one_file(files, files_desc='evoked')
    else:
        evo_fname, epo_exist = locating_meg_file(evo_fname, '*ave.fif')
    return evo_fname.strip()


def get_cor_fname(cor_fname=''):
    cor = COR if cor_fname == '' else cor_fname
    if not op.isfile(cor):
        print('get_cor_fname: cor_fname does not exist! ({})'.format(cor))
    return cor


def get_empty_fname(empty_fname=''):
    if empty_fname == '':
        empty_fname = EMPTY_ROOM
    print('Looking for empty room fif file...')
    empty_fname, empty_exist = locating_meg_file(empty_fname, '*empty*.fif')
    if not empty_exist:
        empty_fname, empty_exist = locating_meg_file(empty_fname, '*noise*.fif')
    if not empty_exist:
        empty_fname, empty_exist = locating_meg_file(empty_fname, '*.fif')
    return empty_fname


def calc_inverse_operator(
        subject, events=None, raw_fname='', epo_fname='', evo_fname='', fwd_fname='', inv_fname='', noise_cov_fname='',
        empty_fname='', bad_channels=[], inv_loose=0.2, inv_depth=0.8, noise_t_min=None, noise_t_max=0,
        overwrite_inverse_operator=False, use_empty_room_for_noise_cov=False, use_raw_for_noise_cov=False,
        overwrite_noise_cov=False, calc_for_cortical_fwd=True, calc_for_sub_cortical_fwd=True, fwd_usingMEG=True,
        fwd_usingEEG=True, check_for_channels_inconsistency=True, calc_for_spec_sub_cortical=False,
        cortical_fwd=None, subcortical_fwd=None, spec_subcortical_fwd=None, region=None, args=None):
    raw_fname = get_raw_fname(raw_fname)
    fwd_fname = get_fwd_fname(fwd_fname, fwd_usingMEG, fwd_usingEEG)
    inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG, True)
    evo_fname = get_evo_fname(subject, evo_fname)
    epo_fname = get_epo_fname(epo_fname)
    if use_empty_room_for_noise_cov:
        empty_fname = get_empty_fname(empty_fname)
    noise_cov = None
    if noise_cov_fname == '':
        if fwd_usingEEG and fwd_usingMEG:
            noise_cov_fname = NOISE_COV_MEEG
        else:
            noise_cov_fname = NOISE_COV_MEG if fwd_usingMEG else NOISE_COV_EEG
    if not op.isfile(noise_cov_fname):
        noise_cov_fname = utils.select_one_file(glob.glob(op.join(MEG_DIR, subject, '*noise-cov.fif')))
    if not op.isfile(noise_cov_fname):
        noise_cov_fname = op.join(MEG_DIR, subject, 'noise-cov.fif')
    print('calc_inverse_operator: noise_cov_fname {}'.format(noise_cov_fname))
    if events is None:
        conds = ['all']
    else:
        conds = ['all'] if '{cond}' not in epo_fname else events.keys()
    flag = True
    for cond in conds:
        if (not overwrite_inverse_operator and
                (not calc_for_cortical_fwd or op.isfile(get_cond_fname(inv_fname, cond))) and \
                (not calc_for_sub_cortical_fwd or op.isfile(get_cond_fname(INV_SUB, cond))) and \
                (not calc_for_spec_sub_cortical or op.isfile(get_cond_fname(INV_X, cond, region=region)))):
            continue
        try:
            if op.isfile(noise_cov_fname) and not overwrite_noise_cov:
                noise_cov = read_noise_cov(noise_cov_fname)
            if noise_cov is None:
                if use_empty_room_for_noise_cov:
                    raw_empty_room = mne.io.read_raw_fif(empty_fname, allow_maxshield=True)  # , add_eeg_ref=False)
                    noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)
                    noise_cov.save(noise_cov_fname)
                elif use_raw_for_noise_cov:
                    # We might want to downfilter the raw for the noise cov calculatation, like in here:
                    # https://martinos.org/mne/stable/manual/sample_dataset.html#computing-the-noise-covariance-matrix
                    # mne_process_raw --raw sample_audvis_raw.fif --lowpass 40 --projon --savecovtag -cov --cov audvis.cov
                    if not op.isfile(raw_fname):
                        print('Can\'t find the raw file ({}) for calculating the noise cov!'.format(raw_fname))
                    raw = mne.io.read_raw_fif(raw_fname)  # preload=True # add_eeg_ref=False
                    noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)
                    noise_cov.save(noise_cov_fname)
                else:
                    epo = get_cond_fname(epo_fname, cond)
                    epochs = mne.read_epochs(epo)
                    noise_cov = calc_noise_cov(
                        subject, epochs, noise_t_min, noise_t_max, noise_cov_fname, args,
                        use_eeg=fwd_usingEEG, use_meg=fwd_usingMEG)

            # todo: should use noise_cov = calc_cov(...
            if calc_for_cortical_fwd and (not op.isfile(get_cond_fname(inv_fname, cond))
                                          or overwrite_inverse_operator):
                if cortical_fwd is None:
                    cortical_fwd = get_cond_fname(fwd_fname, cond)
                _calc_inverse_operator(
                    subject, cortical_fwd, get_cond_fname(inv_fname, cond), raw_fname, get_cond_fname(evo_fname, cond),
                    epo_fname, noise_cov, bad_channels, fwd_usingMEG, fwd_usingEEG, inv_loose, inv_depth,
                    noise_cov_fname,
                    check_for_channels_inconsistency)
            if calc_for_sub_cortical_fwd and (not op.isfile(get_cond_fname(INV_SUB, cond))
                                              or overwrite_inverse_operator):
                if subcortical_fwd is None:
                    subcortical_fwd = get_cond_fname(FWD_SUB, cond)
                _calc_inverse_operator(
                    subject, subcortical_fwd, get_cond_fname(INV_SUB, cond), raw_fname, evo_fname, epo_fname,
                    noise_cov, bad_channels, fwd_usingMEG, fwd_usingEEG,
                    check_for_channels_inconsistency=check_for_channels_inconsistency)
            if calc_for_spec_sub_cortical and (not op.isfile(get_cond_fname(INV_X, cond, region=region))
                                               or overwrite_inverse_operator):
                if spec_subcortical_fwd is None:
                    spec_subcortical_fwd = get_cond_fname(FWD_X, cond, region=region)
                _calc_inverse_operator(
                    subject, spec_subcortical_fwd, get_cond_fname(INV_X, cond, region=region), raw_fname,
                    evo_fname, epo_fname, noise_cov, bad_channels, fwd_usingMEG, fwd_usingEEG,
                    check_for_channels_inconsistency=check_for_channels_inconsistency)
            flag = True
        except:
            print(traceback.format_exc())
            print('Error in calculating inv for {}'.format(cond))
            flag = False
    return flag


@utils.tryit(None)
def read_noise_cov(noise_cov_fname):
    return mne.read_cov(noise_cov_fname)


def _calc_inverse_operator(
        subject, fwd_name, inv_name, raw_fname, evoked_fname, epochs_fname, noise_cov, bad_channels,
        fwd_usingMEG, fwd_usingEEG, inv_loose=0.2, inv_depth=0.8, noise_cov_fname='',
        check_for_channels_inconsistency=True):
    fwd = mne.read_forward_solution(fwd_name)
    info = get_info(
        subject, epochs_fname, evoked_fname, raw_fname, bad_channels,
        fwd_usingEEG=fwd_usingEEG, fwd_usingMEG=fwd_usingMEG)
    if info is None:
        raise Exception("Can't find info for calculating the inverse operator!")
    # noise_cov['bads'] = info['bads']
    noise_cov = check_noise_cov_channels(
        noise_cov, info, fwd, fwd_usingMEG, fwd_usingEEG, noise_cov_fname, check_for_channels_inconsistency)
    inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                             loose=inv_loose, depth=inv_depth)
    write_inverse_operator(inv_name, inverse_operator)


def get_bad_channels(info, bad_channels, fwd_usingEEG=True, fwd_usingMEG=True):
    channels = {c['ch_name'] for c in info['chs']}
    non_exist_bad_channels_all = set(bad_channels) - channels
    non_exist_bad_channels_eeg = {c for c in non_exist_bad_channels_all if c.startswith('EEG')}
    non_exist_bad_channels_meg = {c for c in non_exist_bad_channels_all if c.startswith('MEG')}
    non_exist_bad_channels = []
    if fwd_usingEEG:
        non_exist_bad_channels.extend(non_exist_bad_channels_eeg)
    if fwd_usingMEG:
        non_exist_bad_channels.extend(non_exist_bad_channels_meg)

    if len(non_exist_bad_channels) > 0:
        print('Non existing bad channels were set in args.bad_channels! {}'.format(non_exist_bad_channels))
        if fwd_usingEEG and any([c.startswith('EEG') for c in non_exist_bad_channels]):
            print('EEG: {}'.format([c for c in channels if c.startswith('EEG')]))
        if fwd_usingMEG and any([c.startswith('MEG') for c in non_exist_bad_channels]):
            print('MEG: {}'.format([c for c in channels if c.startswith('MEG')]))
        ret = input('Do you want to continue (y/n)? ')
        if not au.is_true(ret):
            raise Exception('Non existing bad channels')

    bad_channels = list(set(bad_channels) - non_exist_bad_channels_all)
    return bad_channels


def check_noise_cov_channels(noise_cov, info, fwd, fwd_usingMEG, fwd_usingEEG, noise_cov_fname='',
                             check_for_channels_inconsistency=True):
    fwd_sol_ch_names = fwd['sol']['row_names']
    if set([c['ch_name'] for c in info['chs']]) == set(noise_cov.ch_names) == set(fwd_sol_ch_names):
        return noise_cov

    cov_dict = utils.Bag(dict(noise_cov))
    ch0 = info['chs'][0]['ch_name']
    group, num = utils.get_group_and_number(ch0)
    if '{}{}'.format(group, num) == ch0:
        sep = ''
    else:
        sep = ch0[len(group):-len(num)]
    num_len_dict = {'MEG': 0, 'EEG': 0}
    for group_type in num_len_dict.keys():
        for c in info['chs']:
            if c['ch_name'].startswith(group_type):
                group, num = utils.get_group_and_number(c['ch_name'])
                num_len_dict[group_type] = len(num)
                break
    cov_dict.names, cov_dict.bads = [], []

    if fwd_usingEEG and fwd_usingMEG:
        sensors_set = ['EEG', 'MEG']
    elif fwd_usingEEG and not fwd_usingMEG:
        sensors_set = ['EEG']
    elif fwd_usingMEG and not fwd_usingEEG:
        sensors_set = ['MEG']
    else:
        raise Exception('both fwd_usingEEG and fwd_usingMEG are False!')

    C = [c['ch_name'] for c in info['chs'] if c['ch_name'][:3] in sensors_set]
    F = [c for c in fwd_sol_ch_names if c[:3] in sensors_set]
    setC, setF = set(C), set(F)
    for c in noise_cov.ch_names:
        group, num = get_sensor_group_num(c)
        new_ch_name = '{}{}{}'.format(group, sep, num.zfill(num_len_dict[group]))
        if new_ch_name in setC and new_ch_name in setF:
            cov_dict.names.append(new_ch_name)
    for c in noise_cov['bads']:
        group, num = get_sensor_group_num(c)
        cov_dict.bads.append('{}{}{}'.format(group, sep, num.zfill(num_len_dict[group])))
    noise_cov = get_cov_from_dict(cov_dict)

    if check_for_channels_inconsistency:
        N = [c for c in noise_cov.ch_names if c[:3] in sensors_set]
        if not len(C) == len(N) == len(F):
            print('Inconsistency in channels num: info:{}, noise:{}, fwd:{}'.format(len(C), len(N), len(F)))
            ret = input('Do you want to continue? ')
            if not au.is_true(ret):
                raise Exception('Inconsistency in channels names')
        elif not setC == set(noise_cov.ch_names) == setF:
            for k in range(len(C)):
                # if not fwd_sol_ch_names[k] == noise_cov.ch_names[k] == info['chs'][k]['ch_name']:
                if not C[k] == N[k] == F[k]:
                    print(fwd_sol_ch_names[k], noise_cov.ch_names[k], info['chs'][k]['ch_name'])
            ret = input('Inconsistency in channels names! Do you want to continue? ')
            if not au.is_true(ret):
                raise Exception('Inconsistency in channels names')

    if noise_cov_fname == '':
        if fwd_usingEEG and fwd_usingMEG:
            noise_cov_fname = NOISE_COV_MEEG
        else:
            noise_cov_fname = NOISE_COV_MEG if fwd_usingMEG else NOISE_COV_EEG
    noise_cov.save(noise_cov_fname)
    return noise_cov


def get_sensor_group_num(sensor_name):
    group, num = utils.get_group_and_number(sensor_name)
    return group, num


def get_cov_from_dict(cov_dict):
    # (data=data, dim=len(data), names=names, bads=bads, nfree=nfree, eig=eig, eigvec=eigvec, diag=diag,
    #     projs=projs, kind=FIFF.FIFFV_MNE_NOISE_COV)
    return mne.Covariance(
        cov_dict.data, cov_dict.names, cov_dict.bads, cov_dict.projs, cov_dict.nfree, cov_dict.eig, cov_dict.eigvec)


# def calc_stc(inverse_method='dSPM'):
#     snr = 3.0
#     lambda2 = 1.0 / snr ** 2
#     inverse_operator = read_inverse_operator(INV)
#     evoked = mne.read_evokeds(EVO, condition=0, baseline=(None, 0))
#     stc = apply_inverse(evoked, inverse_operator, lambda2, inverse_method,
#                         pick_ori=None)
#     stc.save(STC.format('all', inverse_method))


def calc_stc_per_condition(
        subject, events=None, task='', stc_t_min=None, stc_t_max=None, inverse_method='dSPM', baseline=(None, 0),
        apply_SSP_projection_vectors=True, add_eeg_ref=True, pick_ori=None, single_trial_stc=False,
        calc_source_band_induced_power=False, save_stc=True, snr=3.0, overwrite_stc=False, stc_template='',
        raw_fname='', epo_fname='', evo_fname='', inv_fname='', fwd_usingMEG=True, fwd_usingEEG=True,
        apply_on_raw=False, raw=None, epochs=None, modality='meg', calc_stc_for_all=False, calc_stcs_diff=True,
        atlas='aparc.DKTatlas', bands=None, calc_inducde_power_per_label=True, induced_power_normalize_proj=True,
        downsample_r=1, zero_time=None, n_jobs=6):
    # todo: If the evoked is the raw (no events), we need to seperate it into N events with different ids, to avoid memory error
    # Other options is to use calc_labels_avg_for_rest
    evo_fname = get_evo_fname(subject, evo_fname)
    inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    if pick_ori == '':
        pick_ori = None
    modality = get_modality(fwd_usingMEG, fwd_usingEEG)
    if stc_template == '':
        stc_template = STC[:-4]
        stc_hemi_template = STC_HEMI
    else:
        if stc_template.endswith('.stc'):
            stc_template = stc_template[:-4]
        stc_hemi_template = '{}{}'.format(stc_template, '-{hemi}.stc')
    all_conds = 'all' if task == '' else task
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) else [all_conds]
    stcs, stcs_num = {}, {}
    lambda2 = 1.0 / snr ** 2
    global_inverse_operator = False
    if '{cond}' not in inv_fname:
        if not op.isfile(inv_fname):
            print('No inverse operator was found! ({})'.format(inv_fname))
            return False, stcs, stcs_num
        inverse_operator = read_inverse_operator(inv_fname)
        global_inverse_operator = True
    if calc_stc_for_all or len(events_keys) == 0:
        events_keys.append(all_conds)
    flag = False
    mmvt_fol = utils.make_dir(op.join(MMVT_DIR, MRI_SUBJECT, modality))
    for cond_name in events_keys:
        stc_fname = stc_template.format(cond=cond_name, method=inverse_method, modal=modality)
        if utils.get_parent_fol(stc_fname) == '':
            stc_fname = op.join(MMVT_DIR, SUBJECT, modality, stc_fname)
        if op.isfile('{}.stc'.format(stc_fname)) and not overwrite_stc:
            stcs[cond_name] = mne.read_source_estimate(stc_fname)
            continue
        try:
            if not global_inverse_operator:
                if not op.isfile(inv_fname.format(cond=cond_name)):
                    print('No inverse operator was found!')
                    return False, stcs, stcs_num
                inverse_operator = read_inverse_operator(inv_fname.format(cond=cond_name))
            if (single_trial_stc or calc_source_band_induced_power or events_keys == ['rest']) and not apply_on_raw:
                if epochs is None:
                    # epo_fname = epo_fname.format(cond=cond_name)
                    # epo_fname = get_epo_fname(epo_fname)
                    epo_fname = get_cond_fname(epo_fname, cond_name)
                    # todo: change that!!!
                    if True:  # not op.isfile(epo_fname):
                        if single_trial_stc:
                            print('single_trial_stc=True and no epochs file was found!')
                            return False, stcs, stcs_num
                        else:
                            evoked = get_evoked_cond(
                                subject, cond_name, evo_fname, epo_fname, baseline, apply_SSP_projection_vectors,
                                add_eeg_ref)
                    else:
                        epochs = mne.read_epochs(epo_fname, apply_SSP_projection_vectors, add_eeg_ref)
                try:
                    mne.set_eeg_reference(epochs, ref_channels=None)
                except:
                    print('annot create EEG average reference projector (no EEG data found)')
                if single_trial_stc or events_keys == ['rest']:
                    stcs[cond_name] = mne.minimum_norm.apply_inverse_epochs(
                        epochs, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori, return_generator=True)
                if calc_source_band_induced_power:
                    # todo: add a flag
                    if not evoked is None:  # epochs is None and
                        C, T = evoked.data.shape
                        epochs = mne.EpochsArray(
                            evoked.data.reshape((1, C, T)), evoked.info, np.array([[0, 0, 1]]), 0, 1)
                        stc_fname = '{}-{}'.format(stc_fname, utils.namebase(evo_fname))
                    if epochs is None:
                        print('epochs are None!')
                        return False, stcs, stcs_num
                    calc_induced_power(subject, epochs, atlas, task, inverse_operator, lambda2, stc_fname,
                                       normalize_proj=induced_power_normalize_proj,
                                       overwrite_stc=overwrite_stc, modality=modality, n_jobs=n_jobs)
                    # stc files were already been saved
                    save_stc = False
                    stcs[cond_name] = None
                stcs_num[cond_name] = epochs.events.shape[0]
            if not single_trial_stc:  # So calc_source_band_induced_power can enter here also
                if apply_on_raw:
                    raw_fname = get_raw_fname(raw_fname)
                    if op.isfile(raw_fname):
                        raw = mne.io.read_raw_fif(raw_fname, preload=True)
                    else:
                        raise Exception('Can\'t find the raw data!')
                    try:
                        mne.set_eeg_reference(raw, projection=True)  # , ref_channels=None)
                    except:
                        utils.print_last_error_line()
                        print('Cannot create EEG average reference projector (no EEG data found)')
                    stcs[cond_name] = mne.minimum_norm.apply_inverse_raw(
                        raw, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori)
                else:
                    evoked = get_evoked_cond(
                        subject, cond_name, evo_fname, epo_fname, baseline, apply_SSP_projection_vectors, add_eeg_ref)
                    if stc_template == '':
                        stc_fname = '{}-{}'.format(stc_fname, utils.namebase(evo_fname))
                    stc_mmvt_fname = op.join(mmvt_fol, utils.namebase_with_ext(stc_fname))
                    if save_stc and utils.stc_exist(stc_mmvt_fname) and not overwrite_stc:
                        flag = True  # should check all condisiton...
                        print('{} already exist'.format(stc_mmvt_fname))
                        stcs_num[cond_name] = stcs_num.get(cond_name, 0) + 1
                        stcs[cond_name] = mne.read_source_estimate(stc_mmvt_fname)
                        continue
                    if evoked is None and cond_name == 'all':
                        all_evokes_fname = op.join(SUBJECT_MEG_FOLDER, '{}-all-eve.fif'.format(SUBJECT))
                        if op.isfile(all_evokes_fname):
                            evoked = mne.read_evokeds(all_evokes_fname)[0]
                    if evoked is None:
                        continue
                    if not zero_time is None:
                        if zero_time - evoked.tmin != 0:
                            evoked = evoked.shift_time(zero_time - evoked.tmin)
                    if not stc_t_min is None and not stc_t_max is None:
                        evoked = evoked.crop(stc_t_min, stc_t_max)
                    try:
                        info = evoked.info
                        # if modal == 'eeg' or (not info['custom_ref_applied'] and
                        #                       not mne.io.proj._has_eeg_average_ref_proj(info['projs'])):
                        # todo: should check if this was already done
                        mne.set_eeg_reference(evoked, projection=True)  # ref_channels=None)
                        # evoked.apply_ref
                    except:
                        print('Cannot create EEG average reference projector (no EEG data found)')
                    if isinstance(evoked, list):
                        for evk in evoked:
                            stcs[cond_name] = apply_inverse(
                                evk, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori)
                    else:
                        stcs[cond_name] = apply_inverse(
                            evoked, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori)
            # Can work only for non generator stcs
            if not isinstance(stcs[cond_name], types.GeneratorType):
                factor = 0
                if np.max(stcs[cond_name].data) < 1e-4:
                    factor = 6 if modality == 'eeg' else 12  # todo: depends on the inverse method, should check
                elif np.max(stcs[cond_name].data) > 1e8:
                    factor = -6 if modality == 'eeg' else -12  # todo: depends on the inverse method, should check
                if factor != 0:
                    stcs[cond_name] = mne.SourceEstimate(  # stcs[cond_name].data * np.power(10, factor)
                        stcs[cond_name].data * 10 ** factor, vertices=stcs[cond_name].vertices,
                        tmin=stcs[cond_name].tmin, tstep=stcs[cond_name].tstep, subject=stcs[cond_name].subject,
                        verbose=stcs[cond_name].verbose)
            if downsample_r > 1:
                # stcs[cond_name].resample(stcs[cond_name].sfreq / downsample_r)
                # Much faster if downsample_r is integer:
                data_ds = utils.downsample_2d(stcs[cond_name].data, downsample_r)
                stcs[cond_name] = mne.SourceEstimate(
                    data_ds, stcs[cond_name].vertices, stcs[cond_name].tmin, stcs[cond_name].tstep * downsample_r,
                    subject=subject)
            if save_stc and (not op.isfile(stc_fname) or overwrite_stc) and \
                    not isinstance(stcs[cond_name], types.GeneratorType):
                # MMVT reads stcs only from the 'meg' fol, need to change that
                mmvt_fol = utils.make_dir(op.join(MMVT_DIR, MRI_SUBJECT, modality_fol(modality)))
                mmvt_stc_fname = op.join(mmvt_fol, utils.namebase_with_ext(stc_fname))
                print('Saving the source estimate to {} and\n {}'.format(
                    stc_fname, mmvt_stc_fname))
                print('max: {}, min: {}'.format(np.max(stcs[cond_name].data), np.min(stcs[cond_name].data)))
                ftype = 'h5' if apply_on_raw else 'stc'
                stcs[cond_name].save(mmvt_stc_fname, ftype)
                times_fname = op.join(mmvt_fol, '{}_times.pkl'.format(utils.namebase(mmvt_stc_fname)))
                utils.save((evoked.times[0], evoked.times[-1]), times_fname)
                # stcs[cond_name].save(stc_fname)
                # if mmvt_stc_fname != stc_fname:
                #     for hemi in utils.HEMIS:
                #         utils.make_link(
                #             '{}-{}.stc'.format(stc_fname, hemi),
                #             '{}-{}.stc'.format(mmvt_stc_fname, hemi))
                # stcs[cond_name].save(op.join(mmvt_fol, utils.namebase(stc_fname)))
            flag = True
        except:
            print(traceback.format_exc())
            print('Error with {}!'.format(cond_name))
    if calc_stcs_diff and len(events) == 2:
        calc_stc_diff_both_hemis(subject, events, modality, stc_hemi_template, inverse_method, overwrite_stc)
    return flag, stcs, stcs_num


def get_stc_fname(args):
    return '{}-{}.stc'.format(
        STC[:-4].format(cond=args.conditions[0], method=args.inverse_method[0]), '{hemi}')


def calc_stc_zvals(subject, stc_name, baseline_stc_name, modality='meg', use_abs=False, from_index=None, to_index=None,
                   stc_zvals_name='', overwrite=False):
    fol = utils.make_dir(op.join(op.join(MMVT_DIR, subject, modality)))
    stc_zvals_fname = op.join(fol, stc_zvals_name if stc_zvals_name != '' else '{}-zvals'.format(stc_name))
    if utils.stc_exist(stc_zvals_fname) and not overwrite:
        print('calc_stc_zvals: {} already exist'.format(stc_zvals_fname))
        return True
    stc_template = {}
    for file_name, key in zip([stc_name, baseline_stc_name], ['stc', 'baseline']):
        stc_template[key] = op.join(MMVT_DIR, subject, modality_fol(modality), '{}-{}.stc'.format(file_name, '{hemi}'))
        if not utils.both_hemi_files_exist(stc_template[key]):
            stcs = glob.glob(op.join(MMVT_DIR, subject, modality_fol(modality), '**', '{}-?h.stc'.format(file_name)),
                             recursive=True)
            if len(stcs) == 2:
                stc_template[key] = op.join(utils.get_parent_fol(stcs[0]), '{}-{}.stc'.format(file_name, '{hemi}'))
                if not utils.both_hemi_files_exist(stc_template[key]):
                    print('Can\'t find {}!'.format(file_name))
                    return False
            else:
                if len(set([utils.namebase(f) for f in stcs])) == 2:
                    fols = sorted(list(set([utils.get_parent_fol(f) for f in stcs])))
                    print('stc files were found in more than one folder, please pick one:')
                    fol = utils.select_one_file(fols, print_title=False)
                    if fol != '':
                        stc_template[key] = op.join(fol, '{}-{}.stc'.format(file_name, '{hemi}'))
                    else:
                        return False
                else:
                    print('Can\'t find {}!'.format(file_name))
                    return False
    return calc_stc_zvals(
        subject, stc_template['stc'].format(hemi='rh'), stc_template['baseline'].format(hemi='rh'),
        stc_zvals_fname, use_abs, from_index, to_index, False, overwrite)


def calc_stc_zvals(subject, stc_rh_fname, baseline_rh_fnames, stc_zvals_fname, use_abs=False, from_index=None,
                   to_index=None,
                   no_negatives=False, overwrite=False):
    if utils.both_hemi_files_exist('{}-{}.stc'.format(stc_zvals_fname, '{hemi}')) and not overwrite:
        return True
    stc = mne.read_source_estimate(stc_rh_fname)
    print(utils.namebase(stc_rh_fname), stc.times[0], stc.times[-1])
    if not isinstance(baseline_rh_fnames, list):
        baseline_rh_fnames = [baseline_rh_fnames]
    baseline_data = calc_baseline_data_from_files(baseline_rh_fnames, from_index, to_index)
    baseline_std = np.std(baseline_data * pow(10, -15), axis=1, keepdims=True) * pow(10, 15)
    baseline_mean = np.mean(baseline_data * pow(10, -15), axis=1, keepdims=True) * pow(10, 15)
    if any(np.isinf(baseline_std)):
        print('std of baseline is inf!')
        return False
    elif any(baseline_std == 0):
        print('std of baseline is 0!')
        return False
    zvals = (stc.data - baseline_mean) / baseline_std
    if use_abs:
        zvals = np.abs(zvals)
    if no_negatives:
        zvals[np.where(zvals < 0)] = 0
    stc_zvals = mne.SourceEstimate(zvals, stc.vertices, stc.tmin, stc.tstep, subject=subject)
    print('stc_zvals: max: {}, min: {}'.format(np.max(stc_zvals.data), np.min(stc_zvals.data)))
    print('Saving zvals stc to {}'.format(stc_zvals_fname))
    stc_zvals.save(stc_zvals_fname)
    return utils.both_hemi_files_exist('{}-{}.stc'.format(stc_zvals_fname, '{hemi}'))


def calc_baseline_data_from_files(baseline_fnames, from_index=None, to_index=None):
    baseline_data = np.array([
        mne.read_source_estimate(baseline_fname).data for baseline_fname in tqdm(baseline_fnames)])
    N, V, T = baseline_data.shape
    baseline_data = np.reshape(baseline_data, (V, N * T))
    from_index = 0 if from_index is None else from_index
    to_index = baseline_data.shape[1] if to_index is None else to_index
    return baseline_data[:, from_index:to_index]


def get_fwd_flags(modality):
    if modality == 'meg':
        fwd_usingMEG, fwd_usingEEG = True, False
    elif modality == 'eeg':
        fwd_usingMEG, fwd_usingEEG = False, True
    else:
        fwd_usingMEG, fwd_usingEEG = True, True
    return fwd_usingMEG, fwd_usingEEG


def modality_fol(modality):
    return 'eeg' if modality == 'eeg' else 'meg'


def calc_spatio_temporal_clusters_ttest():
    import scipy.stats
    from mne.stats import spatio_temporal_cluster_1samp_test

    # Load both conditions stcs
    pass
    adjacency = mne.spatial_src_adjacency(src)
    # Run ttest
    p_threshold = 0.001
    t_threshold = -scipy.stats.distributions.t.ppf(p_threshold / 2., data.shape[0] - 1)
    print('Clustering.')
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(X, adjacency=adjacency, n_jobs=1,
                                           threshold=t_threshold, buffer_size=None,
                                           verbose=True)

    results_file_name = op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results',
                                '{}_{}_{}'.format(patient, cond_name, inverse_method))
    np.savez(results_file_name, T_obs=T_obs, clusters=clusters, cluster_p_values=cluster_p_values, H0=H0)
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print('good_cluster_inds: {}'.format(good_cluster_inds))


@check_globals()
def plot_max_labels_data(subject, atlas, labels_data_template='', t_min=None, t_max=None, task='', extract_mode=['mean_flip'],
                         inverse_method=['dSPM'], modality='meg', moving_avg_w=50, fig_fname='',
                         include=None, threshold=None, stc_name=''):
    import matplotlib.pyplot as plt
    # include = ['parstriangularis', 'parsopercularis', 'supermarginal_2', 'supermarginal_3', 'supermarginal_4',
    #            'supermarginal_5', 'superiortemporal_1', 'superiortemporal_2']  # Wernicke and Broca
    if labels_data_template == '':
        labels_data_template = get_labels_data_template(subject, modality)
    for em, im in product(extract_mode, inverse_method):
        if not utils.both_hemi_files_exist(get_labels_data_fname(
                subject, modality, '{hemi}', labels_data_template, im, task, atlas, em)):
            print('Not all the labels data files exist for {} and {}'.format(em, im))
            continue
        times_fname = op.join(MMVT_DIR, subject, modality_fol(modality), '{}_times.pkl'.format(stc_name))
        if t_min is None or t_max is None:
            if op.isfile(times_fname):
                t_min, t_max = utils.load(times_fname)
        labels_data = {}
        for hemi in utils.HEMIS:
            labels_data_fname = get_labels_data_fname(
                subject, modality, hemi, labels_data_template, im, task, atlas, em)
            labels_data[hemi] = np.load(labels_data_fname)
        figures_num = 3 if labels_data['rh']['conditions'] == 2 else 2
        f, axs = plt.subplots(figures_num, sharex=True, sharey=False)
        diffs = {}
        for ax_ind, (hemi, ax) in enumerate(zip(utils.HEMIS, axs[:2])):
            data = labels_data[hemi]['data']  # LxTxC
            conds = list(labels_data[hemi]['conditions'])
            names = list(labels_data[hemi]['names'])
            t_axis = np.linspace(t_min, t_max, data.shape[1])
            if include is not None:
                inds = [k for k, c in enumerate(names) if any([inc in c for inc in include])]
                data = data[inds]
            if moving_avg_w != 0:
                new_data = np.empty((data.shape[0], data.shape[1] - (moving_avg_w - 1), data.shape[2]))
                new_data.fill(np.nan)
                for c in range(data.shape[2]):
                    new_data[:, :, c] = utils.moving_avg(data[:, :, c], moving_avg_w)
                data = new_data
                t_axis = t_axis[int(moving_avg_w / 2):-int(moving_avg_w / 2) + 1]
            ax.plot(t_axis, np.max(data, 0)) # T, C np.percentile(data, 99, axis=0)
            if threshold is not None:
                baseline = threshold[hemi] if isinstance(threshold, dict) else threshold
                ax.axhline(y=baseline, linestyle='--', color='r', label='baseline')
            ax.axvline(x=0, linestyle='--', color='r')
            if data.ndim == 3:
                data_diff = np.squeeze(data[:, :, 0] - data[:, :, 1])
                ax.plot(t_axis, np.max(data_diff, 0), 'r')
                diffs[hemi] = np.max(data_diff, 0)
            # ax.plot(t_axis, np.argmax(data_diff, 0))
            # if ax_ind == 0:
            if len(conds) == 2:
                ax.legend([*conds, '{}-{}'.format(*conds)], loc='upper right')
            ax.set_title('{} hemisphere activation'.format('Right' if hemi == 'rh' else 'Left'))
            # ax.set_ylim(bottom=0)
        max_ylims = max([ax.get_ylim()[1] for ax in axs])
        for ax in axs[:2]:
            ax.set_ylim((0, max_ylims))
        if len(diffs) > 0:
            for hemi in utils.HEMIS:
                axs[2].plot(t_axis, diffs[hemi], label=hemi)
            axs[2].legend(loc='upper right')
            axs[2].set_title('Hemispheres conditions differences')
        plt.tight_layout()
        # figs_fol = '/autofs/space/frieda_003/users/valia/epilepsy_clin/Auditory_language/figures'
        # figs_fol = op.join(MMVT_DIR, subject, modality_fol(modality))
        # fig_fname = op.join(figs_fol, '{}.jpg'.format(fig_name[:-3]))
        if fig_fname != '':
            print('Saving figure in {}'.format(fig_fname))
            plt.savefig(fig_fname)
            plt.close()
        else:
            plt.show()


def plot_max_stc(subject, stc_name, modality='meg', use_abs=True, do_plot=True, return_stc=False,
                 t_min=-2, t_max=5, fig_fname='', mean_baseline=None, stc=None):
    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        # if mmvt_agent is not None:
        #     mmvt_agent.play.set_current_t(int(event.xdata))

    import matplotlib.pyplot as plt
    from src.utils import scripts_utils as su

    # mmvt_agent = su.get_mmvt_object(subject)
    # if mmvt_agent is not None:
    #     print('We got the mmvt object ({})!'.format(list(mmvt_agent._proxy_agent.connections.keys())[0]))
    #     mmvt_agent.play.set_current_t(0)

    if '{subject}' in stc_name:
        stc_name = stc_name.format(subject=subject)
    if op.isfile(stc_name):
        stc_fname = stc_name
    else:
        stc_fname = op.join(MMVT_DIR, subject, modality_fol(modality), '{}-lh.stc'.format(stc_name))
    stc_name = utils.namebase(stc_fname).replace('-lh', '').replace('-rh', '')
    if not op.isfile(stc_fname):
        raise Exception("Can't find the stc file! ({}-lh.stc)".format(stc_name))
    times_fname = op.join(MMVT_DIR, subject, modality_fol(modality), '{}_times.pkl'.format(stc_name))
    if op.isfile(times_fname):
        t_min, t_max = utils.load(times_fname)
    print('Reading {}'.format(stc_fname))
    if stc is None:
        stc = mne.read_source_estimate(stc_fname, subject)
    both_hemi_data_max = np.max(np.abs(stc.data) if use_abs else stc.data, axis=0)
    f, axs = plt.subplots(2, sharex=True, sharey=True)
    t_axis = np.linspace(t_min, t_max, len(both_hemi_data_max))
    for hemi, ax in zip(utils.HEMIS, axs):
        hemi_data = stc.rh_data if hemi == 'rh' else stc.lh_data
        # hemi_data_max = np.max(np.abs(hemi_data) if use_abs else hemi_data, axis=0).squeeze()
        hemi_data_max = np.percentile(np.abs(hemi_data) if use_abs else hemi_data, 99, axis=0).squeeze()
        max_ind = np.argmax(hemi_data_max)
        print('{} max at {}'.format(hemi, max_ind))
        # if evokes_fname != '' and op.isfile(evokes_fname):
        # fig.canvas.mpl_connect('button_press_event', onclick)
        if do_plot:
            ax.plot(t_axis, hemi_data_max)
            if mean_baseline is not None:
                baseline = mean_baseline[hemi] if isinstance(mean_baseline, dict) else mean_baseline
                ax.axhline(y=baseline, linestyle='--', color='r', label='baseline')
            ax.set_title('{} {}'.format(stc_name, hemi))
    plt.tight_layout()
    print('Saving figure in {}'.format(fig_fname))
    if fig_fname != '':
        plt.savefig(fig_fname)
        plt.close()
    elif do_plot:
        plt.show()
    return both_hemi_data_max if return_stc else True


def plot_evoked(subject, evoked_fname, evoked_key=None, pick_meg=True, pick_eeg=True, pick_eog=False, ssp_proj=False,
                spatial_colors=True, window_title='', hline=None, exclude='bads', save_fig=False, fig_fname='',
                overwrite=False):
    if not op.isfile(evoked_fname):
        print('plot_evoked: Can\'t find {}!'.format(evoked_fname))
        return False, None

    if fig_fname == '':
        fig_fname = op.join(MMVT_DIR, subject, 'figures', '{}_evoked.jpg'.format(utils.namebase(evoked_fname)))
    if save_fig and op.isfile(fig_fname) and not overwrite:
        return True, None

    evokes = mne.read_evokeds(evoked_fname)
    evoked = evokes[evoked_key] if evoked_key is not None else evokes[0]
    if len(exclude) == 0:
        exclude = exclude[0]
    picks = mne.pick_types(evoked.info, meg=pick_meg, eeg=pick_eeg, eog=pick_eog, exclude=exclude)
    if len(set([evoked.info['ch_names'][k] for k in picks]).intersection(set(exclude))) > 0:
        print('Not all the bad channels were excluded!')
    fig = evoked.plot(
        picks=picks, proj=ssp_proj, hline=hline, window_title=window_title, spatial_colors=spatial_colors,
        selectable=True, show=not save_fig)
    fig.tight_layout()
    if save_fig:
        import matplotlib.pyplot as plt
        plt.savefig(fig_fname, dpi=300)
        plt.close()
    return True, fig


def plot_topomap(subject, evoked_fname, evoked_key=None, times=[], find_peaks=False, same_peaks=True,
                 ch_types=['mag', 'grad', 'eeg'], proj=False, average=None, n_peaks=5, title='',
                 bad_channels='bads', save_fig=False, fig_fname=''):
    # times : float | array of floats | "auto" | "peaks" | "interactive"
    # ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
    # proj : bool | 'interactive'
    # title : str | None
    # average: float | None. 0.01 would translate into window that starts 5 ms before
    #         and ends 5 ms after a given time point

    import matplotlib.pyplot as plt
    from mne.viz.utils import _find_peaks

    if not op.isfile(evoked_fname):
        print('plot_evoked: Can\'t find {}!'.format(evoked_fname))
        return False, None

    if fig_fname == '':
        fig_fname = op.join(MMVT_DIR, subject, 'figures', '{}.jpg'.format(utils.namebase(evoked_fname)))
    if save_fig and op.isfile(fig_fname):
        return True, None

    evokes = mne.read_evokeds(evoked_fname)
    evoked = evokes[evoked_key] if evoked_key is not None else evokes[0]
    if find_peaks:
        ch_peaks = {}
        if same_peaks:
            peaks = _find_peaks(evoked, n_peaks)
            for ch_type in ch_types:
                ch_peaks[ch_type] = peaks
        else:
            for ch_type in ch_types:
                if ch_type == 'eeg':
                    _evoked = evoked.copy().pick_types(meg=False, eeg=True, exclude=bad_channels)
                elif ch_type in ['mag', 'grad', 'planar1', 'planar2']:
                    _evoked = evoked.copy().pick_types(meg=ch_type, eeg=False, exclude=bad_channels)
                else:
                    raise Exception('Wrong channel type!')
                ch_peaks[ch_type] = _find_peaks(_evoked, n_peaks)
        for t, ch_type in product(times, ch_types):
            ch_peaks[ch_type] = np.insert(ch_peaks[ch_type], ch_peaks[ch_type].searchsorted(t), t)
        n_peaks += len(times)
        times = ch_peaks
    else:
        n_peaks = len(times)
    fig, ax = plt.subplots(len(ch_types), n_peaks)
    for ind, ch_type in enumerate(ch_types):
        ch_times = times[ch_type] if isinstance(times, dict) else times
        ch_type = ch_type if ch_type != 'meg' else True
        evoked.plot_topomap(axes=ax[ind], times=ch_times, ch_type=ch_type, proj=proj, average=average, show=False)
        ax[ind][-2].text(0.5, 0.5, ch_type.upper(), fontsize=12)
    plt.suptitle(title)
    if not save_fig:
        plt.show()
    if save_fig:
        plt.savefig(fig_fname, dpi=300)
    return True, fig


def calc_morlet_freqs(epochs, n_cycles=2, max_high_gamma=120):
    from mne.time_frequency import morlet
    import math
    min_f = math.floor((epochs.info['sfreq'] * n_cycles * 2) / len(epochs.times))
    if min_f < 1:
        min_f = 1
    # freqs = np.concatenate([np.arange(min_f, 30), np.arange(31, 60, 3), np.arange(60, max_high_gamma + 5, 5)])
    freqs = np.arange(min_f, max_high_gamma + 1)
    ws = morlet(epochs.info['sfreq'], freqs, n_cycles=n_cycles, zero_mean=False)
    too_long = any([len(w) > len(epochs.times) for w in ws])
    while too_long:
        print('At least one of the wavelets is longer than the signal. ' +
              'Consider padding the signal or using shorter wavelets.')
        min_f += 1
        print('Increasing min_f to {}'.format(min_f))
        # freqs = np.concatenate([np.arange(min_f, 30), np.arange(31, 60, 3), np.arange(60, max_high_gamma + 5, 5)])
        freqs = np.arange(min_f, max_high_gamma + 1)
        ws = morlet(epochs.info['sfreq'], freqs, n_cycles=n_cycles, zero_mean=False)
        too_long = any([len(w) > len(epochs.times) for w in ws])

    # if min_f <= 30:
    #     freqs = np.concatenate([np.arange(min_f, 30), np.arange(31, 60, 3), np.arange(60, max_high_gamma + 5, 5)])
    # elif min_f <= 60:
    #     freqs = np.concatenate([np.arange(31, 60, 3), np.arange(60, max_high_gamma + 5, 5)])
    # elif min_f <= max_high_gamma:
    #     freqs = np.concatenate([np.arange(60, max_high_gamma + 5, 5)])
    # else:
    #     raise Exception('min_f > max_high_gamma! ({}>{})'.format(min_f, max_high_gamma))
    print('morlet freqs: {}'.format(freqs))
    return freqs


def calc_induced_power(subject, epochs, atlas, task, inverse_operator, lambda2, stc_fname,
                       normalize_proj=True, overwrite_stc=False,
                       modality='meg', df=1, n_cycles=2, downsample=2, n_jobs=6):
    # https://martinos.org/mne/stable/auto_examples/time_frequency/plot_source_space_time_frequency.html
    from mne.minimum_norm import source_band_induced_power
    if epochs is None:
        print('epochs are None!!')
        return False
    # if bands is None or bands == '':
    # min_delta = 1 if n_cycles <= 2 else 2
    max_high_gamma = 120  # 300
    # bands = dict(delta=[min_delta, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, max_high_gamma])
    # freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, max_high_gamma + 5, 5)])
    freqs = calc_morlet_freqs(epochs, n_cycles, max_high_gamma)
    # ret = check_bands(epochs, bands, df, n_cycles)
    # if not ret:
    #     return False
    if normalize_proj:
        epochs.info.normalize_proj()
    # if calc_inducde_power_per_label:
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    fol = utils.make_dir(op.join(root_dir, '{}-induced_power'.format(stc_fname)))
    labels = lu.read_labels(subject, SUBJECTS_MRI_DIR, atlas)
    if len(labels) == 0:
        raise Exception('No labels found for {}!'.format(atlas))
    now = time.time()
    for ind, label in enumerate(labels):
        if 'unknown' in label.name:
            continue
        powers_fname = op.join(fol, '{}_{}_induced_power.npy'.format(task, label.name))
        # exist = all([utils.both_hemi_files_exist(op.join(fol, '{}_{}_{}_induced_power-{}.stc'.format(
        #     task, label.name, band, '{hemi}'))) for band in bands.keys()])
        exist = op.join(powers_fname)
        if exist and not overwrite_stc:
            print('Files already exist for {}'.format(label.name))
            continue
        print('Calculating source_band_induced_power for {}'.format(label.name))
        utils.time_to_go(now, ind, len(labels), runs_num_to_print=1)

        # On a normal computer, you might want to set n_jobs to 1 (memory...)
        # !!! We changed the mne-python implementation, to return the powers !!!
        # todo: copy the function instead of chaging it
        # stcs, powers = source_band_induced_power(
        #     epochs, inverse_operator, bands, label, n_cycles=n_cycles, use_fft=False, lambda2=lambda2,
        #     pca=True, df=df, n_jobs=n_jobs)
        import mne.minimum_norm.time_frequency
        # vertno, src_sel = mne.minimum_norm.inverse.label_src_vertno_sel(label, inv['src'])
        powers, _, _ = mne.minimum_norm.time_frequency._source_induced_power(
            epochs, inverse_operator, freqs, label=label, lambda2=lambda2,
            method='dSPM', n_cycles=n_cycles, n_jobs=n_jobs)
        if powers.shape[2] % 2 == 1:
            powers = powers[:, :, :-1]
        if downsample > 1:
            powers = utils.downsample_3d(powers, downsample)
        powers_db = 10 * np.log10(powers)  # dB/Hz should be baseline corrected!!!
        print('Saving powers to {}'.format(powers_fname))
        np.save(powers_fname, powers_db.astype(np.float16))

        # for band, stc_band in stcs.items():
        # print('Saving the {} source estimate to {}.stc'.format(label.name, stc_fname))
        # band_stc_fname = op.join(fol, '{}_{}_{}_induced_power'.format(task, label.name, band))
        # print('Saving {}'.format(band_stc_fname))
        # stc_band.save(band_stc_fname)
    # params = [(subject, atlas, task, band, stc_fname, labels, modality, fol, overwrite_stc)
    #           for band in bands.keys()]
    # results = utils.run_parallel(_combine_labels_stc_files_parallel, params, len(bands))
    # ret = all(results)
    ret = True
    # else:
    #     stcs = source_band_induced_power(
    #         epochs, inverse_operator, bands, n_cycles=n_cycles, use_fft=False, lambda2=lambda2, pca=True,
    #         df=df, n_jobs=n_jobs)
    #     ret = True
    #     for band, stc_band in stcs.items():
    #         # print('Saving the {} source estimate to {}.stc'.format(label.name, stc_fname))
    #         band_stc_fname = '{}_induced_power_{}'.format(stc_fname, band)
    #         print('Saving {}'.format(band_stc_fname))
    #         stc_band.save(band_stc_fname)
    #         ret = ret and utils.both_hemi_files_exist('{}-{}.stc'.format(band_stc_fname, '{hemi}'))
    return ret


def check_bands(epochs, bands, df=1, n_cycles=2):
    from mne.time_frequency import morlet
    freqs = np.concatenate([np.arange(band[0], band[1] + df / 2.0, df) for _, band in bands.items()])
    ws = morlet(epochs.info['sfreq'], freqs, n_cycles=n_cycles, zero_mean=False)
    too_long = any([len(w) > len(epochs.times) for w in ws])
    if too_long:
        print('At least one of the wavelets is longer than the signal. ' +
              'Consider padding the signal or using shorter wavelets.')
        return False
    else:
        return True


def _combine_labels_stc_files_parallel(p):
    subject, atlas, task, band, stc_fname, labels, modality, fol, overwrite_stc = p
    stc_namebase = utils.namebase(stc_fname) if stc_fname.endswith('.stc') \
        else utils.namebase_with_ext(stc_fname)
    ret = combine_labels_stc_files(
        subject, atlas, fol, '{}_{}'.format(stc_namebase, band), labels, '{}_'.format(task),
        '_{}_induced_power'.format(band), modality, overwrite_stc)
    return ret


def combine_labels_stc_files(subject, atlas, folder, stc_output_name, labels=None, pre_identifier='',
                             post_identifier='', modality='meg', overwrite=False):
    stcs_fol = utils.make_dir(op.join(MMVT_DIR, subject, modality_fol(modality), folder))
    combined_stcs_fol = utils.make_dir(op.join(MMVT_DIR, subject, modality))
    output_fname = op.join(combined_stcs_fol, stc_output_name)
    if utils.stc_exist(output_fname, include_subdirs=True) and not overwrite:
        return True
    if not op.isdir(stcs_fol):
        print('The folder {} could not be found!'.format(stcs_fol))
        return False
    if labels is None:
        labels = lu.read_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas)
    if labels is None or len(labels) == 0:
        print('No labels found for {}!'.format(atlas))
        return False
    vertices, vertices_data = defaultdict(list), defaultdict(list)
    for label in labels:
        stc_label_fname = op.join(stcs_fol, '{}{}{}-lh.stc'.format(pre_identifier, label.name, post_identifier))
        if not op.isfile(stc_label_fname):
            if 'unknown' in label.name:
                continue
            else:
                print('Can\'t find the stc files for {}! ({})'.format(label.name, stc_label_fname))
                return False
        label_stc = mne.read_source_estimate(stc_label_fname)
        verts_data = label_stc.lh_data if label.hemi == 'lh' else label_stc.rh_data
        verts_no = label_stc.lh_vertno if label.hemi == 'lh' else label_stc.rh_vertno
        vertices[label.hemi].extend(verts_no)
        vertices_data[label.hemi].extend(verts_data)
    combined_stc = creating_stc_obj(vertices_data, vertices, subject)
    print('Saving {}'.format(output_fname))
    combined_stc.save(output_fname)
    return utils.both_hemi_files_exist('{}-{}.stc'.format(output_fname, '{hemi}'))


def calc_stc_diff_both_hemis(subject, events, modality, stc_hemi_template, inverse_method, overwrite_stc=False):
    if events is None or len(events) < 2:  # (not isinstance(events, dict)
        return False
    stcs_fol = op.join(MMVT_DIR, subject, modality_fol(modality))
    stc_hemi_template = op.join(stcs_fol, utils.namebase_with_ext(stc_hemi_template.format(
        cond='{cond}', method=inverse_method, hemi='{hemi}', modal=modality)))
    conds = list(events.keys())
    if all([utils.both_hemi_files_exist(
            stc_hemi_template.format(cond=cond, hemi='{hemi}'))
        for cond in events.keys()]) and len(glob.glob(stc_hemi_template.format(cond='*', hemi='*'))) >= 4:
        if len(conds) == 2:
            times_fname = stc_hemi_template.format(cond=conds[0], hemi='').replace('-.stc', '_times.pkl')
            if op.isfile(times_fname):
                times = utils.load(times_fname)
                diff_times_fname = stc_hemi_template.format(cond='{}-{}'.format(
                    conds[0], conds[1]), hemi='').replace('-.stc', '_times.pkl')
                utils.save(times, diff_times_fname)
            for hemi in utils.HEMIS:
                diff_fname = stc_hemi_template.format(cond='{}-{}'.format(conds[0], conds[1]), hemi=hemi)
                if op.isfile(diff_fname) and not overwrite_stc:
                    continue
                calc_stc_diff(
                    stc_hemi_template.format(cond=conds[0], hemi=hemi),
                    stc_hemi_template.format(cond=conds[1], hemi=hemi), diff_fname, modality)
    return utils.both_hemi_files_exist(stc_hemi_template.format(cond='{}-{}'.format(conds[0], conds[1]), hemi='{hemi}'))


def dipoles_fit(subject, dipoles_times, dipoloes_title, evokes=None, noise_cov_fname='', evo_fname='',
                head_to_mri_trans_mat_fname='', min_dist=5., use_meg=True, use_eeg=False, vol_atlas_fname='',
                vol_atlas_lut_fname='', mask_roi='', do_plot=False,
                n_jobs=6):
    from mne.forward import make_forward_dipole
    import nibabel as nib
    if do_plot:
        import matplotlib.pyplot as plt

    if noise_cov_fname == '':
        if use_eeg and use_meg:
            noise_cov_fname = NOISE_COV_MEEG
        else:
            noise_cov_fname = NOISE_COV_MEG if use_meg else NOISE_COV_EEG
    if head_to_mri_trans_mat_fname == '':
        head_to_mri_trans_mat_fname = COR
    evo_fname = get_evo_fname(subject, evo_fname)
    if vol_atlas_fname == '':
        vol_atlas_fname = op.join(op.join(MMVT_DIR, MRI_SUBJECT, 'freeview', 'aparc.DKTatlas+aseg.mgz'))
    if vol_atlas_lut_fname == '':
        vol_atlas_lut_fname = op.join(op.join(MMVT_DIR, MRI_SUBJECT, 'freeview', 'aparc.DKTatlasColorLUT.txt'))

    if not op.isfile(noise_cov_fname):
        print("The noise covariance cannot be found in {}!".format(noise_cov_fname))
        return False
    if not op.isfile(BEM):
        print("The BEM solution cannot be found in {}!".format(BEM))
        return False
    if not op.isfile(head_to_mri_trans_mat_fname):
        print("The MEG-to-head-trans matrix cannot be found in {}!".format(COR))
        return False
    if evokes is None:
        evokes = mne.read_evokeds(evo_fname)
    if not isinstance(evokes, Iterable):
        evokes = [evokes]

    for evoked in evokes:
        evoked.pick_types(meg=use_meg, eeg=use_eeg)
        for (t_min, t_max), dipole_title in zip(dipoles_times, dipoloes_title):
            dipole_fname = op.join(SUBJECT_MEG_FOLDER, 'dipole_{}_{}.pkl'.format(evoked.comment, dipole_title))
            dipole_fix_output_fname = op.join(SUBJECT_MEG_FOLDER,
                                              'dipole_fix_{}_{}.pkl'.format(evoked.comment, dipole_title))
            dipole_stc_fwd_fname = op.join(SUBJECT_MEG_FOLDER,
                                           'dipole_{}_fwd_stc_{}.pkl'.format(evoked.comment, dipole_title))
            dipole_location_figure_fname = op.join(SUBJECT_MEG_FOLDER,
                                                   'dipole_{}_{}.png'.format(evoked.comment, dipole_title))
            if not op.isfile(dipole_fname):
                evoked_t = evoked.copy()
                evoked_t.crop(t_min, t_max)
                dipole, residual = mne.fit_dipole(
                    evoked_t, noise_cov_fname, BEM, head_to_mri_trans_mat_fname, min_dist, n_jobs)
                utils.save((dipole, residual), dipole_fname)
            else:
                dipole, residual = utils.load(dipole_fname)
            if do_plot:
                dipole.plot_locations(head_to_mri_trans_mat_fname, 'DC', SUBJECTS_MRI_DIR, mode='orthoview')
            dipole_pos_vox = dipole_pos_to_vox(dipole, head_to_mri_trans_mat_fname)
            if op.isfile(vol_atlas_fname) and op.isfile(vol_atlas_lut_fname) and mask_roi != '':
                mask = nib.load(vol_atlas_fname).get_data()
                lut = utils.read_freesurfer_lookup_table(lut_fname=vol_atlas_lut_fname, return_dict=True,
                                                         reverse_dict=True)
                _lut = utils.read_freesurfer_lookup_table(lut_fname=vol_atlas_lut_fname, return_dict=True)
                mask_code = lut[mask_roi]
            #
            #     print('asdf')
            if not op.isfile(dipole_fix_output_fname):
                # Estimate the time course of a single dipole with fixed position and orientation
                # (the one that maximized GOF) over the entire interval
                best_idx = np.argmax(dipole.gof)
                dipole_fixed, residual_fixed = mne.fit_dipole(
                    evoked, noise_cov_fname, BEM, head_to_mri_trans_mat_fname, min_dist, pos=dipole.pos[best_idx],
                    ori=dipole.ori[best_idx],
                    n_jobs=n_jobs)
                utils.save((dipole_fixed, residual_fixed), dipole_fix_output_fname)
            else:
                dipole_fixed, residual_fixed = utils.load(dipole_fix_output_fname)
            if not op.isfile(dipole_stc_fwd_fname):
                dipole_fwd, dipole_stc = make_forward_dipole(dipole, BEM, evoked.info, head_to_mri_trans_mat_fname,
                                                             n_jobs=n_jobs)
                utils.save((dipole_fwd, dipole_stc), dipole_stc_fwd_fname)
            else:
                dipole_fwd, dipole_stc = utils.load(dipole_stc_fwd_fname)
            if not op.isfile(dipole_location_figure_fname) and do_plot:
                dipole.plot_locations(head_to_mri_trans_mat_fname, MRI_SUBJECT, SUBJECTS_MRI_DIR, mode='orthoview')
                plt.savefig(dipole_location_figure_fname)

            if do_plot:
                # save_diploe_loc(dipole, COR)
                best_idx = np.argmax(dipole.gof)
                best_time = dipole.times[best_idx]
                # plot_predicted_dipole(evoked, dipole_fwd, dipole_stc, best_time)
                dipole_fixed.plot()


def find_dipole_cortical_locations(atlas, cond, dipole_title, grow_label=True, label_r=5,
                                   inv_fname='', stc_hemi_template='', extract_mode='mean_flip', n_jobs=6):
    from mne.transforms import _get_trans, apply_trans
    from scipy.spatial.distance import cdist

    if inv_fname == '':
        inv_fname = INV
    dipole_fname = op.join(SUBJECT_MEG_FOLDER, 'dipole_{}_{}.pkl'.format(cond, dipole_title))
    vertices_labels_lookup = lu.create_vertices_labels_lookup(MRI_SUBJECT, atlas)
    dipole, _ = utils.load(dipole_fname)
    trans = _get_trans(COR, fro='head', to='mri')[0]
    scatter_points = apply_trans(trans['trans'], dipole.pos) * 1e3

    vertices, vertices_dist = {}, {}
    for hemi_ind, hemi in enumerate(['lh', 'rh']):
        verts, _ = utils.read_ply_file(op.join(MMVT_DIR, MRI_SUBJECT, 'surf', '{}.pial.ply'.format(hemi)))
        dists = cdist(scatter_points, verts)
        vertices[hemi_ind] = np.argmin(dists, 1)
        vertices_dist[hemi_ind] = np.min(dists, 1)
    hemis = np.argmin(np.vstack((vertices_dist[0], vertices_dist[1])), 0)
    sort_inds = np.argsort(dipole.gof)[::-1]
    print('****** {} ********'.format(cond))
    dipole_vert = -1
    for ind in sort_inds:
        hemi_ind = hemis[ind]
        hemi = ['lh', 'rh'][hemi_ind]
        dist = vertices_dist[hemi_ind][ind]
        vert = vertices[hemi_ind][ind]
        label = vertices_labels_lookup[hemi][vert]
        gof = dipole.gof[ind]
        if label.startswith('precentral'):
            print(ind, hemi, dist, label, gof)
            if dipole_vert == -1:
                dipole_vert = vert
                dipole_hemi = hemi
                print(cond, gof)
                break
    label_data = None
    if grow_label and dipole_vert != -1:
        label_name = 'precentral_dipole_{}'.format(cond)
        label_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'labels', '{}-{}.label'.format(label_name, dipole_hemi))
        if not op.isfile(label_fname):
            dipole_label = lu.grow_label(
                MRI_SUBJECT, dipole_vert, dipole_hemi, label_name, label_r, n_jobs)
        else:
            dipole_label = mne.read_label(label_fname)
        inverse_operator = read_inverse_operator(inv_fname.format(cond=cond))
        src = inverse_operator['src']
        stc = mne.read_source_estimate(stc_hemi_template.format(cond=cond, hemi=dipole_hemi))
        label_data = stc.extract_label_time_course(dipole_label, src, mode=extract_mode, allow_empty=True)
        label_data = np.squeeze(label_data)
    return dipole_vert, label_data, label_fname, dipole_hemi


def plot_predicted_dipole(evoked, dipole_fwd, dipole_stc, best_time):
    from mne.simulation import simulate_evoked

    noise_cov = mne.read_cov(NOISE_COV)
    pred_evoked = simulate_evoked(dipole_fwd, dipole_stc, evoked.info, cov=noise_cov)  # , nave=np.inf)

    # first plot the topography at the time of the best fitting (single) dipole
    plot_params = dict(times=best_time, ch_type='mag', outlines='skirt',
                       colorbar=True)  # , vmin=vmin, vmax=vmin)
    evoked.plot_topomap(**plot_params)
    # compare this to the predicted field
    pred_evoked.plot_topomap(**plot_params)
    print('Comparison of measured and predicted fields at {:.0f} ms'.format(best_time * 1000.))


def dipole_pos_to_vox(dipole, trans):
    from mne.transforms import _get_trans, apply_trans

    trans_fname = op.join(MMVT_DIR, MRI_SUBJECT, 't1_trans.npz')
    if not op.isfile(trans_fname):
        from src.preproc import anatomy as anat
        anat.save_subject_orig_trans(MRI_SUBJECT)
    orig_trans = utils.Bag(np.load(trans_fname))
    trans = _get_trans(trans, fro='head', to='mri')[0]
    scatter_points = apply_trans(trans['trans'], dipole.pos) * 1e3
    dipole_locs_vox = apply_trans(orig_trans.ras_tkr2vox, scatter_points)
    dipole_locs_vox = np.rint(np.array(dipole_locs_vox)).astype(int)
    return dipole_locs_vox


def save_diploe_loc(dipole, trans):
    from mne.transforms import _get_trans, apply_trans

    best_idx = np.argmax(dipole.gof)
    best_point = dipole_locs[best_idx]
    dipole_xyz = np.rint(best_point).astype(int)
    dipole_ori = apply_trans(trans['trans'], dipole.ori, move=False)
    print(dipole_xyz, dipole_ori)


def get_evoked_cond(subject, cond_name, evo_fname='', epo_fname='', baseline=(None, 0),
                    apply_SSP_projection_vectors=True,
                    add_eeg_ref=True):
    evo_fname = get_evo_fname(subject, evo_fname)
    evoked = None
    if '{cond}' not in evo_fname:
        # if not op.isfile(evo_fname):
        #     print('get_evoked_cond: No evoked file found! ({})'.format(evo_fname))
        #     return None
        try:
            if op.isfile(evo_fname):
                evoked = mne.read_evokeds(evo_fname, condition=cond_name, baseline=baseline)
        except:
            print('No evoked data with the condition {}'.format(cond_name))
            try:
                evoked = mne.read_evokeds(evo_fname)
            except:
                print(traceback.format_exc())
                evoked = None
    if evoked is None:
        evo_cond = get_cond_fname(evo_fname, cond_name)
        if op.isfile(evo_cond):
            evoked = mne.read_evokeds(evo_cond, baseline=baseline)[0]
        else:
            print('No evoked file, trying to use epo file')
            epo_fname = get_epo_fname(epo_fname)
            if not op.isfile(epo_fname):
                print('No epochs were found!')
                return None
            if '{cond}' not in epo_fname:
                epochs = mne.read_epochs(epo_fname, apply_SSP_projection_vectors, add_eeg_ref)
                evoked = epochs[cond_name].average()
            else:
                epo_cond = get_cond_fname(epo_fname, cond_name)
                epochs = mne.read_epochs(epo_cond, apply_SSP_projection_vectors, add_eeg_ref)
                evoked = epochs.average()
            mne.write_evokeds(evo_cond, evoked)
    if isinstance(evoked, list) and len(evoked) > 1:
        ret = -1
        while not au.is_int(ret) or ret < 0 or ret >= len(evoked):
            print('More than one evoked were found:')
            for ind, ev in enumerate(evoked):
                print('{}) {}'.format(ind, ev.comment))
            ret = input(('For cond {}, which evoked do you want to choose? '.format(cond_name)))
            if au.is_int(ret) and 0 <= int(ret) < len(evoked):
                return evoked[int(ret)]
            else:
                print('You should pick a number')
    else:
        return evoked[0] if isinstance(evoked, list) else evoked


def get_cond_fname(fname, cond, **kargs):
    if '{cond}' in fname:
        kargs['cond'] = cond
    return fname.format(**kargs)


def calc_sub_cortical_activity(events, sub_corticals_codes_file=None, inverse_method='dSPM', pick_ori=None,
                               evoked=None, epochs=None, regions=None, inv_include_hemis=True, n_dipoles=0):
    lut = utils.read_freesurfer_lookup_table()
    evoked_given = not evoked is None
    epochs_given = not epochs is None
    if regions is not None:
        sub_corticals = utils.lut_labels_to_indices(regions, lut)
    else:
        if not sub_corticals_codes_file is None:
            sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    if len(sub_corticals) == 0:
        return

    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    global_evo = False
    if '{cond}' not in EVO:
        global_evo = True
        if not evoked_given:
            evoked = {event: mne.read_evokeds(EVO, condition=event, baseline=(None, 0)) for event in events.keys()}
        inv_fname = INV_SUB if len(sub_corticals) > 1 else INV_X.format(region=regions[0])
        if op.isfile(inv_fname):
            inverse_operator = read_inverse_operator(inv_fname)
        else:
            print('The Inverse operator file does not exist! {}'.format(inv_fname))
            return False

    for event in events.keys():
        sub_corticals_activity = {}
        if not global_evo:
            evo = get_cond_fname(EVO, event)
            if not evoked_given:
                evoked = {event: mne.read_evokeds(evo, baseline=(None, 0))[0]}
            inv_fname = get_cond_fname(INV_SUB, event) if len(sub_corticals) > 1 else \
                get_cond_fname(INV_X, event, region=regions[0])
            inverse_operator = read_inverse_operator(inv_fname)
        if inverse_method in ['lcmv', 'dics', 'rap_music']:
            if not epochs_given:
                epochs = mne.read_epochs(get_cond_fname(EPO, event))
            fwd_fname = get_cond_fname(FWD_SUB, event) if len(sub_corticals) > 1 else get_cond_fname(FWD_X, event,
                                                                                                     region=regions[0])
            forward = mne.read_forward_solution(fwd_fname)
        if inverse_method in ['lcmv', 'rap_music']:
            noise_cov = calc_cov(get_cond_fname(NOISE_COV, event), event, epochs, None, 0)

        if inverse_method == 'lcmv':
            from mne.beamformer import lcmv
            data_cov = calc_cov(get_cond_fname(DATA_COV, event), event, epochs, 0.0, 1.0)
            # pick_ori = None | 'normal' | 'max-power'
            stc = lcmv(evoked[event], forward, noise_cov, data_cov, reg=0.01, pick_ori='max-power')

        elif inverse_method == 'dics':
            from mne.beamformer import dics
            from mne.time_frequency import compute_epochs_csd
            data_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=0.0, tmax=2.0,
                                          fmin=6, fmax=10)
            noise_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=-0.5, tmax=0.0,
                                           fmin=6, fmax=10)
            stc = dics(evoked[event], forward, noise_csd, data_csd)

        elif inverse_method == 'rap_music':
            if len(sub_corticals) > 1:
                print('Need to do more work here for len(sub_corticals) > 1')
            else:
                from mne.beamformer import rap_music
                noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))
                n_dipoles = len(sub_corticals) if n_dipoles == 0 else n_dipoles
                dipoles = rap_music(evoked[event], forward, noise_cov, n_dipoles=n_dipoles,
                                    return_residual=False, verbose=True)
                for sub_cortical_ind, sub_cortical_code in enumerate(sub_corticals):
                    amp = dipoles[sub_cortical_ind].amplitude
                    amp = amp.reshape((1, amp.shape[0]))
                    sub_corticals_activity[sub_cortical_code] = amp
                    print(set(
                        [tuple(dipoles[sub_cortical_ind].pos[t]) for t in range(len(dipoles[sub_cortical_ind].times))]))
        else:
            stc = apply_inverse(evoked[event], inverse_operator, lambda2, inverse_method, pick_ori=pick_ori)

        if inverse_method not in ['rap_music']:
            # todo: maybe to flip?
            # stc.extract_label_time_course(label, src, mode='mean_flip')
            read_vertices_from = len(stc.vertices[0]) + len(stc.vertices[1]) if inv_include_hemis else 0
            # 2 becasue the first two are the hemispheres
            sub_cortical_indices_shift = 2 if inv_include_hemis else 0
            for sub_cortical_ind, sub_cortical_code in enumerate(sub_corticals):
                if len(sub_corticals) > 1:
                    vertices_to_read = len(stc.vertices[sub_cortical_ind + sub_cortical_indices_shift])
                else:
                    vertices_to_read = len(stc.vertices)
                sub_corticals_activity[sub_cortical_code] = stc.data[
                                                            read_vertices_from: read_vertices_from + vertices_to_read]
                read_vertices_from += vertices_to_read

        subs_fol = op.join(SUBJECT_MEG_FOLDER, 'subcorticals', inverse_method)
        utils.make_dir(subs_fol)
        for sub_cortical_code, activity in sub_corticals_activity.items():
            sub_cortical, _ = utils.get_numeric_index_to_label(sub_cortical_code, lut)
            np.save(op.join(subs_fol, '{}-{}-{}.npy'.format(event, sub_cortical, inverse_method)), activity.mean(0))
            np.save(op.join(subs_fol, '{}-{}-{}-all-vertices.npy'.format(event, sub_cortical, inverse_method)),
                    activity.mean(0))
            # np.save(op.join(subs_fol, '{}-{}-{}'.format(event, sub_cortical, inverse_method)), activity.mean(0))
            # np.save(op.join(subs_fol, '{}-{}-{}-all-vertices'.format(event, sub_cortical, inverse_method)), activity)


def calc_cov(cov_fname, cond, epochs, from_t, to_t, method='empirical', overwrite=False):
    cov_cond_fname = get_cond_fname(cov_fname, cond)
    if not op.isfile(cov_cond_fname) or overwrite:
        cov = mne.compute_covariance(epochs.crop(from_t, to_t), method=method)
        cov.save(cov_cond_fname)
    else:
        cov = mne.read_cov(cov_cond_fname)
    return cov


def calc_csd(csd_fname, cond, epochs, from_t, to_t, mode='multitaper', fmin=6, fmax=10, overwrite=False):
    from mne.time_frequency import compute_epochs_csd
    csd_cond_fname = get_cond_fname(csd_fname, cond)
    if not op.isfile(csd_cond_fname) or overwrite:
        csd = compute_epochs_csd(epochs, mode, tmin=from_t, tmax=to_t, fmin=fmin, fmax=fmax)
        utils.save(csd, csd_cond_fname)
    else:
        csd = utils.load(csd_cond_fname)
    return csd


def calc_specific_subcortical_activity(region, inverse_methods, events, plot_all_vertices=False,
                                       overwrite_fwd=False, overwrite_inv=False, overwrite_activity=False,
                                       inv_loose=0.2, inv_depth=0.8):
    raise Exception('calc_specific_subcortical_activity: Needed to be reimplemented!!!')
    if not x_opertor_exists(FWD_X, region, events) or overwrite_fwd:
        make_forward_solution_to_specific_subcortrical(events, region, bad_channels)
    if not x_opertor_exists(INV_X, region, events) or overwrite_inv:
        # todo: send subject as parameter!!
        calc_inverse_operator(subject, events, '', '', '', '', '', '', '', inv_loose, inv_depth,
                              False, False, False, False, False, True, region=region)
    for inverse_method in inverse_methods:
        files_exist = np.all([op.isfile(op.join(SUBJECT_MEG_FOLDER, 'subcorticals',
                                                '{}-{}-{}.npy'.format(cond, region, inverse_method))) for cond in
                              events.keys()])
        if not files_exist or overwrite_activity:
            calc_sub_cortical_activity(events, None, inverse_method=inverse_method, pick_ori=pick_ori,
                                       regions=[region], inv_include_hemis=False)
        plot_sub_cortical_activity(events, None, inverse_method=inverse_method,
                                   regions=[region], all_vertices=plot_all_vertices)


def x_opertor_exists(operator, region, events):
    if not '{cond}' in operator:
        exists = op.isfile(op.join(SUBJECT_MEG_FOLDER, operator.format(region=region)))
    else:
        exists = np.all([op.isfile(op.join(SUBJECT_MEG_FOLDER,
                                           get_cond_fname(operator, cond, region=region))) for cond in events.keys()])
    return exists


def plot_sub_cortical_activity(events, sub_corticals_codes_file, inverse_method='dSPM', regions=None,
                               all_vertices=False):
    import matplotlib.pyplot as plt
    lut = utils.read_freesurfer_lookup_table()
    if sub_corticals_codes_file is None:
        sub_corticals = regions
    else:
        sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    for label in sub_corticals:
        sub_cortical, _ = utils.get_numeric_index_to_label(label, lut)
        print(sub_cortical)
        activity = {}
        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
        fig_name = '{} ({}): {}, {}, contrast {}'.format(sub_cortical, inverse_method,
                                                         events.keys()[0], events.keys()[1],
                                                         '-all-vertices' if all_vertices else '')
        ax1.set_title(fig_name)
        for event, ax in zip(events.keys(), [ax1, ax2]):
            data_file_name = '{}-{}-{}{}.npy'.format(event, sub_cortical, inverse_method,
                                                     '-all-vertices' if all_vertices else '')
            activity[event] = np.load(op.join(SUBJECT_MEG_FOLDER, 'subcorticals', data_file_name))
            ax.plot(activity[event].T)
        ax3.plot(activity[events.keys()[0]].T - activity[events.keys()[1]].T)
        f.subplots_adjust(hspace=0.2)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        fol = op.join(SUBJECT_MEG_FOLDER, 'figures')
        if not op.isdir(fol):
            os.mkdir(fol)
        plt.savefig(op.join(fol, fig_name))
        plt.close()


def save_subcortical_activity_to_blender(sub_corticals_codes_file, events, stat, inverse_method='dSPM',
                                         norm_by_percentile=True, norm_percs=(1, 99), do_plot=False):
    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure()

    sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    first_time = True
    names_for_blender = []
    lut = utils.read_freesurfer_lookup_table()
    for ind, sub_cortical_ind in enumerate(sub_corticals):
        sub_cortical_name, _ = utils.get_numeric_index_to_label(sub_cortical_ind, lut)
        names_for_blender.append(sub_cortical_name)
        for cond_id, cond in enumerate(events.keys()):
            data_fname = op.join(SUBJECT_MEG_FOLDER, 'subcorticals', inverse_method,
                                 '{}-{}-{}.npy'.format(cond, sub_cortical_name, inverse_method))
            if op.isfile(data_fname):
                x = np.load(op.join(SUBJECT_MEG_FOLDER, 'subcorticals', inverse_method,
                                    '{}-{}-{}.npy'.format(cond, sub_cortical_name, inverse_method)))
                if first_time:
                    first_time = False
                    T = len(x)
                    data = np.zeros((len(sub_corticals), T, len(events.keys())))
                data[ind, :, cond_id] = x[:T]
            else:
                print('The file {} does not exist!'.format(data_fname))
                return
        if do_plot:
            plt.plot(data[ind, :, 0] - data[ind, :, 1], label='{}-{} {}'.format(
                events.keys()[0], events.keys()[1], sub_cortical_name))

    stat_data = utils.calc_stat_data(data, stat)
    # Normalize
    # todo: I don't think we should normalize stat_data
    # stat_data = utils.normalize_data(stat_data, norm_by_percentile, norm_percs)
    data = utils.normalize_data(data, norm_by_percentile, norm_percs)
    data_max, data_min = utils.get_data_max_min(stat_data, norm_by_percentile, norm_percs, symmetric=True)
    # if stat == STAT_AVG:
    #     colors = utils.mat_to_colors(stat_data, data_min, data_max, colorsMap=colors_map)
    # elif stat == STAT_DIFF:
    #     data_minmax = max(map(abs, [data_max, data_min]))
    #     colors = utils.mat_to_colors_two_colors_maps(stat_data, threshold=threshold,
    #         x_max=data_minmax,x_min = -data_minmax, cm_big=cm_big, cm_small=cm_small,
    #         default_val=1, flip_cm_big=flip_cm_big, flip_cm_small=flip_cm_small)

    np.savez(op.join(MMVT_SUBJECT_FOLDER, 'subcortical_meg_activity'), data=data,
             names=names_for_blender, conditions=list(events.keys()), data_minmax=data_max)

    if do_plot:
        plt.legend()
        plt.show()


# def plotActivationTS(stcs_meg):
#     import matplotlib.pyplot as plt
#
#     plt.close('all')
#     plt.figure(figsize=(8, 6))
#     name = 'MEG'
#     stc = stcs_meg
#     plt.plot(1e3 * stc.times, stc.data[::150, :].T)
#     plt.ylabel('%s\ndSPM value' % str.upper(name))
#     plt.xlabel('time (ms)')
#     plt.show()
#
#
# def plot3DActivity(stc=None):
#     import matplotlib.pyplot as plt
#     if (stc is None):
#         stc = read_source_estimate(STC)
#     # Plot brain in 3D with PySurfer if available. Note that the subject name
#     # is already known by the SourceEstimate stc object.
#     brain = stc.plot(surface='inflated', hemi='rh', subjects_dir=SUBJECT_MRI_DIR, subject=SUBJECT)
#     brain.scale_data_colormap(fmin=8, fmid=12, fmax=15, transparent=True)
#     brain.show_view('lateral')
#
#     # use peak getter to move vizualization to the time point of the peak
#     vertno_max, time_idx = stc.get_peak(hemi='rh', time_as_index=True)
#     brain.set_data_time_index(time_idx)
#
#     # draw marker at maximum peaking vertex
#     brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
#                     scale_factor=0.6)
# #     brain.save_movie(SUBJECT_FOLDER)
#     brain.save_image(getPngName('dSPM_map'))


# def morphTOTlrc(method='MNE'):
#     # use dSPM method (could also be MNE or sLORETA)
#     epoches = loadEpoches()
#     epo1 = epoches[CONDS[0]]
#     epo2 = epoches[CONDS[1]]
#
#     snr = 3.0
#     lambda2 = 1.0 / snr ** 2
#
#     inverse_operator = read_inverse_operator(INV)
#
#     #    Let's average and compute inverse, resampling to speed things up
#     evoked1 = epo1.average()
#     evoked1.resample(50)
#     condition1 = apply_inverse(evoked1, inverse_operator, lambda2, method)
#     evoked2 = epo2.average()
#     evoked2.resample(50)
#     condition2 = apply_inverse(evoked2, inverse_operator, lambda2, method)
#
#     cond1tlrc = mne.morph_data(SUBJECT, 'fsaverage', condition1, subjects_dir=SUBJECTS_DIR, n_jobs=4)
#     cond2tlrc = mne.morph_data(SUBJECT, 'fsaverage', condition2, subjects_dir=SUBJECTS_DIR, n_jobs=4)
#     cond1tlrc.save(op.join(SUBJECT_FOLDER, '{}_tlrc_{}'.format(method, CONDS[0])))
#     cond2tlrc.save(op.join(SUBJECT_FOLDER, '{}_tlrc_{}'.format(method, CONDS[1])))


# def morph_stc(subject_to, cond='all', grade=None, n_jobs=6, inverse_method='dSPM'):
#     stc = mne.read_source_estimate(STC.format(cond, inverse_method))
#     vertices_to = mne.grade_to_vertices(subject_to, grade=grade)
#     stc_to = mne.morph_data(SUBJECT, subject_to, stc, n_jobs=n_jobs, grade=vertices_to)
#     fol_to = op.join(MEG_DIR, TASK, subject_to)
#     if not op.isdir(fol_to):
#         os.mkdir(fol_to)
#     stc_to.save(STC_MORPH.format(subject_to, cond, inverse_method))


def calc_stc_for_all_vertices(stc, subject='', morph_to_subject='', n_jobs=6):
    subject = MRI_SUBJECT if subject == '' else subject
    morph_to_subject = subject if morph_to_subject == '' else morph_to_subject
    # grade = None if morph_to_subject != 'fsaverage' else 5
    # vertices_to = mne.grade_to_vertices(morph_to_subject, grade)
    # return mne.morph_data(subject, morph_to_subject, stc, n_jobs=n_jobs, grade=vertices_to
    morph_obj = mne.compute_source_morph(stc, subject, morph_to_subject, SUBJECTS_MRI_DIR, spacing=None)
    return morph_obj.apply(stc)


@utils.files_needed({'surf': ['lh.sphere.reg', 'lh.sphere.reg']})
def calc_source_morph_mat(subject_from, subject_to, src_vertices=None, zooms=5, niter_affine=(100, 100, 10),
                          niter_sdr=(5, 5, 3), spacing=5, overwrite=False):
    import mne.morph
    output_fname = op.join(MMVT_DIR, subject_from, 'smooth_map_to_{}.pkl'.format(subject_to))
    if op.isfile(output_fname) and not overwrite:
        morph = utils.load(output_fname)
        return True, morph
    if not utils.both_hemi_files_exist(op.join(SUBJECTS_MRI_DIR, subject_from, 'surf', '{hemi}.sphere.reg')):
        print('Can\'t find {}!'.format(op.join(SUBJECTS_MRI_DIR, subject_from, 'surf', '{hemi}.sphere.reg')))
        return False, None
    if src_vertices is None:
        stc_files = glob.glob(op.join(MMVT_DIR, subject_from, 'meg', '*.stc')) + \
                    glob.glob(op.join(MEG_DIR, subject_from, '*.stc'))
        if len(stc_files) == 0:
            print('Can\'t find any stc files!')
            return False, None
        stc = mne.read_source_estimate(stc_files[0])
        src_vertices = stc.vertices

    src_data = dict(vertices_from=copy.deepcopy(src_vertices))
    vertices_from = src_data['vertices_from']
    pial = utils.get_pial_vertices(subject_to, MMVT_DIR)
    vertices_to = [np.arange(len(pial['lh'])), np.arange(len(pial['rh']))]
    morph_mat = mne.morph._compute_morph_matrix(
        subject_from=subject_from, subject_to=subject_to,
        vertices_from=vertices_from, vertices_to=vertices_to,
        subjects_dir=SUBJECTS_MRI_DIR, smooth=None,
        xhemi=False)
    n_verts = sum(len(v) for v in vertices_to)
    assert morph_mat.shape[0] == n_verts
    morph = mne.SourceMorph(
        subject_from, subject_to, 'surface', zooms, niter_affine, niter_sdr, spacing, None, False,
        morph_mat, vertices_to, None, None, None, None, src_data)
    utils.save(morph, output_fname)
    return op.isfile(output_fname), morph


# def create_stc_t(stc, t, subject=''):
#     from mne import SourceEstimate
#     subject = MRI_SUBJECT if subject == '' else MRI_SUBJECT
#     data = np.concatenate([stc.lh_data[:, t:t + 1], stc.rh_data[:, t:t + 1]])
#     vertices = [stc.lh_vertno, stc.rh_vertno]
#     stc_t = SourceEstimate(data, vertices, stc.tmin + t * stc.tstep, stc.tstep, subject=subject, verbose=stc.verbose)
#     return stc_t


def create_stc_t(stc, t, subject=''):
    subject = MRI_SUBJECT if subject == '' else subject
    C = max([stc.rh_data.shape[0], stc.lh_data.shape[0]])
    stc_lh_data = stc.lh_data[:, t:t + 1] if stc.lh_data.shape[0] > 0 else np.zeros((C, 1))
    stc_rh_data = stc.rh_data[:, t:t + 1] if stc.rh_data.shape[0] > 0 else np.zeros((C, 1))
    data = np.concatenate([stc_lh_data, stc_rh_data])
    vertno = max([len(stc.lh_vertno), len(stc.rh_vertno)])
    lh_vertno = stc.lh_vertno if len(stc.lh_vertno) > 0 else np.arange(0, vertno)
    rh_vertno = stc.rh_vertno if len(stc.rh_vertno) > 0 else np.arange(0, vertno) + max(lh_vertno) + 1
    vertices = [lh_vertno, rh_vertno]
    stc_t = mne.SourceEstimate(data, vertices, stc.tmin + t * stc.tstep, stc.tstep, subject=subject,
                               verbose=stc.verbose)
    return stc_t


def create_stc_t_from_data(subject, rh_data, lh_data, tstep=0.001):
    lh_vertno, rh_vertno = len(lh_data), len(rh_data)
    data = np.concatenate([lh_data, rh_data])
    if np.max(data) > 1e-6:
        data /= np.power(10, 9)
    data = np.reshape(data, (data.shape[0], 1))
    rh_verts, _ = utils.read_pial(subject, MMVT_DIR, 'rh')
    lh_verts, _ = utils.read_pial(subject, MMVT_DIR, 'lh')
    lh_vertno = np.arange(0, lh_vertno)
    rh_vertno = np.arange(0, rh_vertno) + max(lh_vertno) + 1
    vertices = [lh_vertno, rh_vertno]
    stc_t = mne.SourceEstimate(data, vertices, 0, tstep, subject=subject)
    return stc_t


@utils.files_needed({'surf': ['lh.sphere.reg', 'rh.sphere.reg']})
def morph_stc(subject, events, morph_to_subject, inverse_method='dSPM', grade=5, smoothing_iterations=None,
              modality='meg', overwrite=False, n_jobs=6):
    ret = True
    # if utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'meg', '{}-{}.stc'.format(stc_name, '{hemi}'))):
    #         output_fname = op.join(MMVT_DIR, morph_to_subject stc_name.replace(subject, morph_to_subject)
    for ind, cond in enumerate(events.keys()):
        input_fname = stc_fname = STC_HEMI.format(cond=cond, method=inverse_method, hemi='rh')
        if not op.isfile(input_fname):
            stcs_files = list(set([f[:-len('-rh.stc')] for f in glob.glob(op.join(SUBJECT_MEG_FOLDER, '*.stc'))]))
            stcs_files += list(
                set([f[:-len('-rh.stc')] for f in glob.glob(op.join(MMVT_DIR, subject, 'meg', '*.stc'))]))
            stc_fname = utils.select_one_file(stcs_files)
            input_fname = '{}-rh.stc'.format(stc_fname)
            output_fname = stc_fname.replace(subject, morph_to_subject)
        else:
            output_fname = STC_HEMI_SAVE.format(cond=cond, method=inverse_method).replace(SUBJECT, morph_to_subject)
        if not op.isfile(input_fname):
            print("Can't find {}!".format(input_fname))
            continue
        if utils.both_hemi_files_exist('{}-{}.stc'.format(output_fname, '{hemi}')) and not overwrite:
            continue
        utils.make_dir(utils.get_parent_fol(output_fname))
        stc = mne.read_source_estimate(stc_fname)
        stc_morphed = mne.morph_data(
            subject, morph_to_subject, stc, grade=grade, smooth=smoothing_iterations, n_jobs=n_jobs)
        stc_morphed.save(output_fname)
        print('Morphed stc file was saves in {}'.format(output_fname))
        ret = ret and utils.both_hemi_files_exist(output_fname)
    # diff_template = op.join(SUBJECT_MEG_FOLDER, '{cond}-{hemi}.stc').replace(SUBJECT, morph_to_subject)
    diff_template = STC_HEMI.format(cond='{cond}', hemi='{hemi}', method=inverse_method).replace(SUBJECT,
                                                                                                 morph_to_subject)
    ret = ret and calc_stc_diff_both_hemis(subject, events, modality, diff_template, inverse_method, overwrite)
    return ret


@utils.files_needed({'surf': ['lh.sphere.reg', 'rh.sphere.reg']})
def morph_stc_file(
        subject, stc_fname, morph_to_subject, grade=5, modality='meg', overwrite=False):
    if '{hemi}' in stc_fname:
        stc_fname = stc_fname.replace('{hemi}', 'rh')
    if not op.isdir(op.join(SUBJECTS_MRI_DIR, morph_to_subject)):
        print('You should have {} in you SUBJECTS_DIR ({})'.format(morph_to_subject, SUBJECTS_MRI_DIR))
        return False
    if not utils.both_hemi_files_exist(op.join(
            SUBJECTS_MRI_DIR, morph_to_subject, 'surf', '{hemi}.sphere.reg')):
        print('lh.sphere.reg and rh.sphere.reg cannot be found under {}'.format(
            SUBJECTS_MRI_DIR, morph_to_subject, 'surf'))
        return False
    output_fol = utils.make_dir(op.join(MMVT_DIR, morph_to_subject, modality))
    stc_name = utils.namebase(stc_fname).replace('-rh.stc', '').replace('-lh.stc', '')
    output_fname = op.join(output_fol, '{}-morphed-to-{}'.format(stc_name, morph_to_subject))
    if utils.both_hemi_files_exist(output_fname) and not overwrite:
        print('The ouptut file already exist: {}.\n If you wish to overwrite, set overwrite to True '
              '(--overwrite_stc 1).'.format(output_fname))
        return True
    stc = mne.read_source_estimate(stc_fname)
    ret, morph_map = calc_source_morph_mat(
        subject, morph_to_subject, stc.vertices, spacing=grade, overwrite=overwrite)
    if not ret or morph_map is None:
        print('Can\'t calculate the morping map between {} and {}!')
        return False
    stc_morphed = morph_map.apply(stc)
    stc_morphed.save(output_fname)
    print('Morphed stc file was saves in {}'.format(output_fname))
    return utils.both_hemi_files_exist(output_fname)


# @utils.timeit
def smooth_stc(events, stcs_conds=None, inverse_method='dSPM', t=-1, morph_to_subject='', n_jobs=6):
    try:
        stcs = {}
        for ind, cond in enumerate(events.keys()):
            output_fname = STC_HEMI_SMOOTH_SAVE.format(cond=cond, method=inverse_method)
            if morph_to_subject != '':
                output_fname = '{}-{}'.format(output_fname, morph_to_subject)
            if stcs_conds is not None:
                stc = stcs_conds[cond]
            else:
                # Can read only for the 'rh', it'll also read the second file for 'lh'. Strange...
                stc = mne.read_source_estimate(STC_HEMI.format(cond=cond, method=inverse_method, hemi='rh'))
            if t != -1:
                stc = create_stc_t(stc, t)
                output_fname = '{}-t{}'.format(output_fname, t)
            stc_smooth = calc_stc_for_all_vertices(stc, MRI_SUBJECT, morph_to_subject, n_jobs)
            check_stc_with_ply(stc_smooth, cond, morph_to_subject)
            stc_smooth.save(output_fname)
            stcs[cond] = stc_smooth
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in calculating inv for {}'.format(cond))
        flag = False

    return flag, stcs


def check_stc_with_ply(stc, cond_name='', subject=''):
    mmvt_surf_fol = op.join(MMVT_DIR, MRI_SUBJECT if subject == '' else subject, 'surf')
    verts = {}
    for hemi in HEMIS:
        stc_vertices = stc.rh_vertno if hemi == 'rh' else stc.lh_vertno
        print('{} {} stc vertices: {}'.format(hemi, cond_name, len(stc_vertices)))
        verts[hemi], _ = utils.read_ply_file(op.join(mmvt_surf_fol, '{}.pial.ply'.format(hemi)))
        print('{} {} ply vertices: {}'.format(hemi, cond_name, verts[hemi].shape[0]))
        if len(stc_vertices) != verts[hemi].shape[0]:
            raise Exception('check_stc_with_ply: Wrong number of vertices!')
    return verts


# def get_pial_vertices(subject):
#     mmvt_surf_fol = op.join(MMVT_DIR, MRI_SUBJECT if subject == '' else subject, 'surf')
#     verts = {}
#     for hemi in HEMIS:
#         verts[hemi], _ = utils.read_ply_file(op.join(mmvt_surf_fol, '{}.pial.ply'.format(hemi)))
#     return verts


def save_activity_map(events, stat, stcs_conds=None, inverse_method='dSPM', smoothed_stc=True, morph_to_subject='',
                      stc_t=-1, norm_by_percentile=False, norm_percs=(1, 99), plot_cb=False):
    try:
        if stat not in [STAT_DIFF, STAT_AVG]:
            raise Exception('stat not in [STAT_DIFF, STAT_AVG]!')
        stcs = get_stat_stc_over_conditions(
            events, stat, stcs_conds, inverse_method, smoothed_stc, morph_to_subject, stc_t)
        if stc_t == -1:
            save_activity_map_minmax(stcs, events, stat, stcs_conds, inverse_method, morph_to_subject,
                                     norm_by_percentile, norm_percs, plot_cb)
        subject = MRI_SUBJECT if morph_to_subject == '' else morph_to_subject
        for hemi in HEMIS:
            verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
            data = stcs[hemi]
            if verts.shape[0] != data.shape[0]:
                raise Exception('save_activity_map: wrong number of vertices!')
            else:
                print('Both {}.pial.ply and the stc file have {} vertices'.format(hemi, data.shape[0]))
            fol = '{}'.format(ACT.format(hemi))
            if morph_to_subject != '':
                fol = fol.replace(MRI_SUBJECT, morph_to_subject)
            if stc_t == -1:
                utils.delete_folder_files(fol)
                now = time.time()
                T = data.shape[1]
                for t in range(T):
                    utils.time_to_go(now, t, T, runs_num_to_print=10)
                    np.save(op.join(fol, 't{}'.format(t)), data[:, t])
            else:
                utils.make_dir(fol)
                np.save(op.join(fol, 't{}'.format(stc_t)), data)
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in save_activity_map')
        flag = False
    return flag


def save_activity_map_minmax(stcs=None, events=None, stat=STAT_DIFF, stcs_conds=None, inverse_method='dSPM',
                             morph_to_subject='', norm_by_percentile=False, norm_percs=(1, 99), plot_cb=False):
    from src.utils import color_maps_utils as cp
    from src.utils import figures_utils as figu

    subject = MRI_SUBJECT if morph_to_subject == '' else morph_to_subject
    output_fname = op.join(MMVT_DIR, subject, 'meg_activity_map_minmax.pkl')
    if stcs is None:
        if stat not in [STAT_DIFF, STAT_AVG]:
            raise Exception('stat not in [STAT_DIFF, STAT_AVG]!')
        stcs = get_stat_stc_over_conditions(events, stat, stcs_conds, inverse_method, False)
        if stcs is None and morph_to_subject != '':
            stcs = get_stat_stc_over_conditions(events, stat, stcs_conds, inverse_method, False, morph_to_subject)
        if stcs is None:
            print("Can't find the stc files!")
    data_max, data_min = utils.get_activity_max_min(stcs, norm_by_percentile, norm_percs)
    data_minmax = utils.get_max_abs(data_max, data_min)
    print('Saving data minmax, min: {}, max: {} to {}'.format(-data_minmax, data_minmax, output_fname))
    utils.save((-data_minmax, data_minmax), output_fname)
    if plot_cb:
        # todo: create colors map according to the parameters
        colors_map = cp.create_BuPu_YlOrRd_cm()
        figures_fol = op.join(MMVT_DIR, MRI_SUBJECT, 'figures')
        figu.plot_color_bar(data_minmax, -data_minmax, colors_map, fol=figures_fol)
    return op.isfile(output_fname)


# def calc_activity_significance(events, inverse_method, stcs_conds=None):
#     from mne import spatial_tris_connectivity, grade_to_tris
#     from mne.stats import (spatio_temporal_cluster_1samp_test)
#     from mne import bem
#     from scipy import stats as stats
#
#     paired_constart_fname = op.join(SUBJECT_MEG_FOLDER, 'paired_contrast.npy')
#     n_subjects = 1def calc_activity_significance(events, inverse_method, stcs_conds=None):
#     from mne import spatial_tris_connectivity, grade_to_tris
#     from mne.stats import (spatio_temporal_cluster_1samp_test)
#     from mne import bem
#     from scipy import stats as stats
#
#     paired_constart_fname = op.join(SUBJECT_MEG_FOLDER, 'paired_contrast.npy')
#     n_subjects = 1
#     if not op.isfile(paired_constart_fname):
#         stc_template = STC_HEMI_SMOOTH
#         if stcs_conds is None:
#             stcs_conds = {}
#             for cond_ind, cond in enumerate(events.keys()):
#                 # Reading only the rh, the lh will be read too
#                 print('Reading {}'.format(stc_template.format(cond=cond, method=inverse_method, hemi='lh')))
#                 stcs_conds[cond] = mne.read_source_estimate(stc_template.format(cond=cond, method=inverse_method, hemi='lh'))
#
#         # Let's only deal with t > 0, cropping to reduce multiple comparisons
#         for cond in events.keys():
#             stcs_conds[cond].crop(0, None)
#         conds = sorted(list(events.keys()))
#         tmin = stcs_conds[conds[0]].tmin
#         tstep = stcs_conds[conds[0]].tstep
#         n_vertices_sample, n_times = stcs_conds[conds[0]].data.shape
#         X = np.zeros((n_vertices_sample, n_times, n_subjects, 2))
#         X[:, :, :, 0] += stcs_conds[conds[0]].data[:, :, np.newaxis]
#         X[:, :, :, 1] += stcs_conds[conds[1]].data[:, :, np.newaxis]
#         X = np.abs(X)  # only magnitude
#         X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast
#         #    Note that X needs to be a multi-dimensional array of shape
#         #    samples (subjects) x time x space, so we permute dimensions
#         X = np.transpose(X, [2, 1, 0])
#         np.save(paired_constart_fname, X)
#     else:
#         X = np.load(paired_constart_fname)
#
#     #    To use an algorithm optimized for spatio-temporal clustering, we
#     #    just pass the spatial connectivity matrix (instead of spatio-temporal)
#     print('Computing connectivity.')
#     # tris = get_subject_tris()
#     connectivity = None # spatial_tris_connectivity(tris)
#     #    Now let's actually do the clustering. This can take a long time...
#     #    Here we set the threshold quite high to reduce computation.
#     p_threshold = 0.2
#     t_threshold = -stats.distributions.t.ppf(p_threshold / 2., 10 - 1)
#     print('Clustering.')
#     T_obs, clusters, cluster_p_values, H0 = \
#         spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=6,
#             threshold=t_threshold)
#     #    Now select the clusters that are sig. at p < 0.05 (note that this value
#     #    is multiple-comparisons corrected).
#     good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
#     # utils.save((clu, good_cluster_inds), op.join(SUBJECT_MEG_FOLDER, 'spatio_temporal_ttest.npy'))
#     np.savez(op.join(SUBJECT_MEG_FOLDER, 'spatio_temporal_ttest'), T_obs=T_obs, clusters=clusters,
#              cluster_p_values=cluster_p_values, H0=H0, good_cluster_inds=good_cluster_inds)
#     print('good_cluster_inds: {}'.format(good_cluster_inds))


def get_subject_tris():
    from mne import read_surface
    _, tris_lh = read_surface(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'surf', 'lh.white'))
    _, tris_rh = read_surface(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'surf', 'rh.white'))
    # tris =  [tris_lh, tris_rh]
    tris = np.vstack((tris_lh, tris_rh))
    return tris


def save_vertex_activity_map(events, stat, stcs_conds=None, inverse_method='dSPM', number_of_files=100):
    try:
        if stat not in [STAT_DIFF, STAT_AVG]:
            raise Exception('stat not in [STAT_DIFF, STAT_AVG]!')
        stcs = get_stat_stc_over_conditions(events, stat, stcs_conds, inverse_method, smoothed=True)
        for hemi in HEMIS:
            verts, faces = utils.read_pial(MRI_SUBJECT, MMVT_DIR, hemi)
            data = stcs[hemi]
            if verts.shape[0] != data.shape[0]:
                raise Exception('save_vertex_activity_map: wrong number of vertices!')
            else:
                print('Both {}.pial.ply and the stc file have {} vertices'.format(hemi, data.shape[0]))

            data_hash = defaultdict(list)
            fol = '{}_verts'.format(ACT.format(hemi))
            utils.delete_folder_files(fol)
            look_up = np.zeros((data.shape[0], 2), dtype=np.int)
            for vert_ind in range(data.shape[0]):
                file_num = vert_ind % number_of_files
                data_hash[file_num].append(data[vert_ind, :])
                look_up[vert_ind] = [file_num, len(data_hash[file_num]) - 1]
                if vert_ind % 10000 == 0:
                    print('{}: {} out of {}'.format(hemi, vert_ind, data.shape[0]))

            np.save('{}_verts_lookup'.format(ACT.format(hemi)), look_up)
            for file_num in range(number_of_files):
                file_name = op.join(fol, str(file_num))
                x = np.array(data_hash[file_num])
                np.save(file_name, x)
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in save_vertex_activity_map')
        flag = False
    return flag


def get_stat_stc_over_conditions(events, stat, stcs_conds=None, inverse_method='dSPM', smoothed=False,
                                 morph_to_subject='', stc_t=-1):
    stcs = {}
    stc_template = STC_HEMI if not smoothed else STC_HEMI_SMOOTH
    for cond_ind, cond in enumerate(events.keys()):
        if stcs_conds is None:
            # Reading only the rh, the lh will be read too
            input_fname = stc_template.format(cond=cond, method=inverse_method, hemi='lh')
            if morph_to_subject != '':
                input_fname = '{}-{}-lh.stc'.format(input_fname[:-len('-lh.stc')], morph_to_subject)
            if stc_t != -1:
                input_fname = '{}-t{}-lh.stc'.format(input_fname[:-len('-lh.stc')], stc_t)
            if op.isfile(input_fname):
                print('Reading {}'.format(input_fname))
                stc = mne.read_source_estimate(input_fname)
            else:
                print('No such file {}!'.format(input_fname))
                return None
        else:
            stc = stcs_conds[cond]
        for hemi in HEMIS:
            data = stc.rh_data if hemi == 'rh' else stc.lh_data
            if hemi not in stcs:
                stcs[hemi] = np.zeros((data.shape[0], data.shape[1], len(events)))
            stcs[hemi][:, :, cond_ind] = data
    for hemi in HEMIS:
        if stat == STAT_AVG:
            # Average over the conditions
            stcs[hemi] = stcs[hemi].mean(2)
        elif stat == STAT_DIFF:
            # Calc the diff of the conditions
            stcs[hemi] = np.squeeze(np.diff(stcs[hemi], axis=2))
        else:
            raise Exception('Wrong value for stat, should be STAT_AVG or STAT_DIFF')
    return stcs


def rename_activity_files():
    fol = '/homes/5/npeled/space3/MEG/ECR/mg79/activity_map_rh'
    files = glob.glob(op.join(fol, '*.npy'))
    for file in files:
        name = '{}.npy'.format(file.split('/')[-1].split('-')[0])
        os.rename(file, op.join(fol, name))


# def calc_labels_avg(parc, hemi, surf_name, stc=None):
#     if stc is None:
#         stc = mne.read_source_estimate(STC)
#     labels = mne.read_labels_from_annot(SUBJECT, parc, hemi, surf_name)
#     inverse_operator = read_inverse_operator(INV)
#     src = inverse_operator['src']
#
#     plt.close('all')
#     plt.figure()
#
#     for ind, label in enumerate(labels):
#         # stc_label = stc.in_label(label)
#         mean_flip = stc.extract_label_time_course(label, src, mode='mean_flip')
#         mean_flip = np.squeeze(mean_flip)
#         if ind==0:
#             labels_data = np.zeros((len(labels), len(mean_flip)))
#         labels_data[ind, :] = mean_flip
#         plt.plot(mean_flip, label=label.name)
#
#     np.savez(LBL.format('all'), data=labels_data, names=[l.name for l in labels])
#     plt.legend()
#     plt.xlabel('time (ms)')
#     plt.show()


def morph_labels_from_fsaverage(atlas='aparc250', fs_labels_fol='', sub_labels_fol='', n_jobs=6):
    lu.morph_labels_from_fsaverage(MRI_SUBJECT, SUBJECTS_MRI_DIR, MMVT_DIR, atlas, fs_labels_fol, sub_labels_fol,
                                   n_jobs)


def labels_to_annot(parc_name, subject='', mri_subject='', labels_fol='', hemi='both', labels=[], overwrite=True):
    subject = SUBJECT if subject == '' else subject
    mri_subject = subject if mri_subject == '' else mri_subject
    ret = lu.labels_to_annot(mri_subject, SUBJECTS_MRI_DIR, parc_name, labels_fol, overwrite, hemi=hemi,
                             labels=labels)
    return op.join(SUBJECTS_MRI_DIR, mri_subject, 'label', '{}.{}.annot'.format(hemi, parc_name))


def calc_single_trial_labels_per_condition(atlas, events, stcs, extract_modes=('mean_flip'), src=None):
    global_inverse_operator = False
    if '{cond}' not in INV:
        global_inverse_operator = True
        if src is None:
            inverse_operator = read_inverse_operator(INV)
            src = inverse_operator['src']

    for extract_mode in extract_modes:
        for (cond_name, cond_id), stc in zip(events.items(), stcs.values()):
            if not global_inverse_operator:
                if src is None:
                    inverse_operator = read_inverse_operator(INV.format(cond=cond_name))
                    src = inverse_operator['src']
            labels = lu.read_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas)
            labels_ts = mne.extract_label_time_course(stcs[cond_name], labels, src, mode=extract_mode,
                                                      return_generator=False, allow_empty=True)
            np.save(op.join(SUBJECT_MEG_FOLDER, 'labels_ts_{}_{}'.format(cond_name, extract_mode)), np.array(labels_ts))


def get_stc_conds(subject, events, inverse_method, stc_hemi_template, modality):
    stcs = {}
    hemi = 'lh'  # both will be loaded
    for cond in events.keys():
        stc_fname = stc_hemi_template.format(cond=cond, method=inverse_method, hemi=hemi, modal=modality)
        if not utils.stc_exist(stc_fname.replace('-lh', '')):
            print('Finding stc file for the condition {}'.format(cond))
            if len(events.keys()) == 1:
                template_mmvt = '{}.{}'.format(stc_hemi_template.format(
                    cond='*', method=inverse_method, hemi='*-{}'.format(hemi), modal=modality), '{type}')
                stc_fname = utils.select_one_file(
                    glob.glob(template_mmvt.format(type='stc')) + glob.glob(template_mmvt.format(type='h5')))
            else:
                template_meg_stc = op.join(SUBJECT_MEG_FOLDER, '*-{}.stc'.format(hemi))
                template_meg_h5 = op.join(SUBJECT_MEG_FOLDER, '*-stc.h5'.format(hemi))
                template_mmvt_stc = op.join(MMVT_DIR, subject, modality, '*-{}.stc'.format(hemi))
                template_mmvt_h5 = op.join(MMVT_DIR, subject, modality, '*-stc.h5'.format(hemi))
                stc_fname = utils.select_one_file(
                    glob.glob(template_meg_stc) + glob.glob(template_meg_h5) + glob.glob(template_mmvt_stc) +
                    glob.glob(template_mmvt_h5), template='stc/h5', files_desc='STC', print_title=True)
        else:
            stc_fname = '{}.stc'.format(stc_fname)
        if not op.isfile(stc_fname):
            return None
        print('Read {}'.format(stc_fname))
        stcs[cond] = mne.read_source_estimate(stc_fname)
    return stcs, stc_fname


def calc_labels_avg_per_cluster(
        subject, atlas, events, inverse_method, stc_names, extract_method, modality='meg',
        labels_output_fname_template='', task='', modalitiy='meg'):
    if labels_output_fname_template == '':
        labels_output_fname_template = LBL
    labels_data = {hemi: {} for hemi in utils.HEMIS}
    labels_names = defaultdict(list)
    for hemi in utils.HEMIS:
        labels_data[hemi][extract_method] = None
    conditions = list(events.keys())
    for stc_ind, stc_name in enumerate(stc_names):
        clusters_fname = utils.make_dir(op.join(
            MMVT_DIR, subject, modalitiy, 'clusters', 'clusters_labels_{}.pkl'.format(stc_name)))
        clusters = utils.load(clusters_fname)
        cond, inv_method, lfreq, hfreq = utils.namebase(clusters_fname)[len('clusters_labels_'):].split('-')
        cond_id = conditions.index(cond)
        clusters_info = [info for info in clusters.values if info.label_data is not None]
        for hemi in utils.HEMIS:
            if labels_data[hemi][extract_method] is None:
                labels_data[hemi][extract_method] = np.zeros(
                    (len(clusters_info), len(clusters_info[0].label_data), len(stc_names), len(events)))
        for ind, info in enumerate(clusters_info):
            labels_data[info.hemi][extract_method][ind, :, cond_id, stc_ind] = info.label_data
            label_name = '{name}_max_{max:.2f}_size_{size}'.format(**info)
            labels_names[info.hemi].append(label_name)
    for hemi in utils.HEMIS:
        labels_output_fname = get_labels_data_fname(
            subject, modality, hemi, labels_output_fname_template, inverse_method, task, atlas, extract_method)
        lables_mmvt_fname = op.join(MMVT_DIR, MRI_SUBJECT, modalitiy, op.basename(labels_output_fname))
        np.savez(labels_output_fname, data=labels_data[hemi][extract_method],
                 names=labels_names, conditions=conditions)
        utils.copy_file(labels_output_fname, lables_mmvt_fname)


def check_all_lables_data_exist(
        subject, modality, hemi, labels_data_template, task, atlas, extract_modes, inverse_methods):
    return all([op.isfile(
        get_labels_data_fname(
            subject, modality, hemi, labels_data_template, im, task, atlas, em))
        for em, im in product(extract_modes, inverse_methods)])


@check_globals()
def get_labels_data_template(subject, modality):
    return op.join(
            MMVT_DIR, subject, modality_fol(modality), 'labels_data_{}_{}_{}_{}_{}.npz')


@check_globals()
def calc_labels_avg_per_condition(
        subject, atlas, hemi, events=None, surf_name='pial', labels_fol='', stcs=None, stcs_num={},
        inverse_method=['dSPM'], extract_modes=['mean_flip'], positive=False, moving_average_win_size=0,
        labels_data_template='', src=None, factor=1, inv_fname='', fwd_usingMEG=True, fwd_usingEEG=True,
        read_only_from_annot=True, task='', mri_subject='', stc_fname='', stc_hemi_template='', overwrite=False,
        stc=None, do_plot=False, n_jobs=1):
    mri_subject = subject if mri_subject == '' else mri_subject
    modality = get_modality(fwd_usingMEG, fwd_usingEEG)
    if isinstance(inverse_method, str):
        inverse_method = [inverse_method]
    if labels_data_template == '':
        labels_data_template = get_labels_data_template(mri_subject, modality)

    if do_plot:
        import matplotlib.pyplot as plt

    if stcs is None:
        stcs = {}
        if stc is not None:
            stcs['all'] = stc
        elif stc_fname != '':
            stcs['all'] = mne.read_source_estimate(stc_fname)
    # if stc_hemi_template == '':
    #     stc_hemi_template = op.join(MMVT_DIR, MRI_SUBJECT, modality_fol(modality), utils.namebase(STC_HEMI))
    # if stc_name != '':
    #     labels_data_template = op.join(utils.get_parent_fol(
    #         stc_name), 'labels_data_{}'.format(utils.namebase(stc_name)[:-3]))
    #     labels_data_template += '_{}_{}_{}_{}_{}.npz' #task, atlas, inverse_method, em, hemi
    labels_data = {}
    if events is None or len(events) == 0:
        events = dict(all=0)
    conditions = list(events.keys())
    all_files_exist = check_all_lables_data_exist(
        subject, modality, hemi, labels_data_template, task, atlas, extract_modes, inverse_method)
    if all_files_exist and not overwrite:
        return True
    try:
        labels = lu.read_labels(
            mri_subject, SUBJECTS_MRI_DIR, atlas, hemi=hemi, surf_name=surf_name,
            labels_fol=labels_fol, read_only_from_annot=read_only_from_annot, n_jobs=n_jobs)
        if len(labels) == 0:
            print('No labels were found for {} atlas!'.format(atlas))
            return False

        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
        global_inverse_operator = False
        if '{cond}' not in inv_fname:
            if not op.isfile(inv_fname):
                print('No inverse operator found!')
                return False
            global_inverse_operator = True
            if src is None:
                inverse_operator = read_inverse_operator(inv_fname)
                src = inverse_operator['src']

        if do_plot:
            utils.make_dir(op.join(SUBJECT_MEG_FOLDER, 'figures'))

        # todo: check why those lines were removed
        if stcs is None or len(stcs) == 0:
            stcs = get_stc_conds(subject, events, inverse_method, stc_hemi_template, modality)
        conds_incdices = {cond_id: ind for ind, cond_id in zip(range(len(stcs)), events.values())}

        if not check_source_and_labels_interestion(src, labels):
            return False

        for (cond_name, cond_id), stc_cond in zip(events.items(), stcs.values()):
            if do_plot:
                plt.figure()
            if not global_inverse_operator:
                if src is None:
                    if not op.isfile(inv_fname.format(cond=cond_name)):
                        print('No inverse operator found!')
                        return False
                    inverse_operator = read_inverse_operator(inv_fname.format(cond=cond_name))
                    src = inverse_operator['src']

            if isinstance(stc_cond, types.GeneratorType):
                stc_cond_num = stcs_num[cond_name]
            else:
                stc_cond = [stc_cond]
                stc_cond_num = 1
            for stc_ind, stc in enumerate(stc_cond):
                for em in extract_modes:
                    for ind, label in enumerate(labels):
                        label_data = stc.extract_label_time_course(label, src, mode=em, allow_empty=True)
                        label_data = np.squeeze(label_data)
                        # Set flip to be always positive
                        # mean_flip *= np.sign(mean_flip[np.argmax(np.abs(mean_flip))])
                        if em not in labels_data:
                            T = len(stc.times)
                            labels_data[em] = np.zeros((len(labels), T, len(stcs), stc_cond_num))
                        labels_data[em][ind, :, conds_incdices[cond_id], stc_ind] = label_data
                        if do_plot:
                            plt.plot(labels_data[em][ind, :, conds_incdices[cond_id]], label=label.name)

            if do_plot:
                plt.xlabel('time (ms)')
                plt.title('{}: {} {}'.format(cond_name, hemi, atlas))
                plt.legend()
                # plt.show()
                plt.savefig(op.join(SUBJECT_MEG_FOLDER, 'figures', '{}: {} {}.png'.format(cond_name, hemi, atlas)))

        save_labels_data(
            subject, modality, labels_data, hemi, labels, atlas, conditions, extract_modes, inverse_method,
            labels_data_template, task, factor, positive, moving_average_win_size)
        flag = True  # todo: check_all_files_exist(labels_data_template)
    except:
        print(traceback.format_exc())
        print('Error in calc_labels_avg_per_condition inv')
        flag = False
    return flag


def check_source_and_labels_interestion(src, labels):
    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]
    labels_with_no_vertices = 0
    for label in labels:
        if label.hemi == 'both':
            # handle BiHemiLabel
            sub_labels = [label.lh, label.rh]
        else:
            sub_labels = [label]
        this_vertidx = list()
        for slabel in sub_labels:
            if slabel.hemi == 'lh':
                this_vertno = np.intersect1d(vertno[0], slabel.vertices)
                vertidx = np.searchsorted(vertno[0], this_vertno)
            elif slabel.hemi == 'rh':
                this_vertno = np.intersect1d(vertno[1], slabel.vertices)
                vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)
            else:
                raise ValueError('label %s has invalid hemi' % label.name)
            this_vertidx.append(vertidx)

        this_vertidx = np.concatenate(this_vertidx)
        if len(this_vertidx) == 0:
            print('source space does not contain any vertices for label {}!'.format(label.name))
            labels_with_no_vertices += 1
        else:
            print('label {} contain {} source space vertices'.format(label.name, len(this_vertidx)))
    if labels_with_no_vertices > 0:
        print('source space does not contain any vertices for {} labels!'.format(labels_with_no_vertices))
        ret = input('Do you wish to continue? ')
        if not au.is_true(ret):
            return False
    return True


def save_labels_data(
        subject, modality, labels_data, hemi, labels_names, atlas, conditions, extract_modes, inverse_method,
        labels_data_template, task='', factor=1, positive=False, moving_average_win_size=0):
    if not isinstance(labels_names[0], str):
        labels_names = [utils.to_str(l.name) for l in labels_names]
    for em, im in product(extract_modes, inverse_method):
        labels_data[em] = labels_data[em].squeeze()
        if np.max(labels_data[em]) < 1e-4:
            labels_data[em] *= np.power(10, factor)
        if positive or moving_average_win_size > 0:
            labels_data[em] = utils.make_evoked_smooth_and_positive(
                labels_data[em], conditions, positive, moving_average_win_size)
        labels_output_fname = get_labels_data_fname(
            subject, modality, hemi, labels_data_template, im, task, atlas, em)
        print('Saving to {}'.format(labels_output_fname))
        utils.make_dir(utils.get_parent_fol(labels_output_fname))
        # If labels_data is per ephoch: labels_num x time x conds_num x epoches_num
        np.savez(labels_output_fname, data=labels_data[em], names=labels_names, conditions=conditions)


def calc_power_spectrum(subject, events, args, fwd_usingEEG=True, fwd_usingMEG=True, modality='meg', do_plot=False):
    if do_plot:
        import matplotlib.pyplot as plt

    info = get_info(subject, args.epo_fname, args.evo_fname, args.raw_fname, args.info_fname,
                    fwd_usingEEG=fwd_usingEEG, fwd_usingMEG=fwd_usingMEG)
    if info is None:
        print('calc_power_spectrum: Can\'t find the MEG info file!')
        return False
    if args.labels_data_template == '':
        args.labels_data_template = LBL
    output_template = op.join(MMVT_DIR, subject, modality_fol(modality),
                              'labels_data_power_spectrum_{}_{}_{}_{}_{}.npz'.format(
                                  args.task, args.atlas, '{em}', '{im}', '{hemi}'))
    events_ids = [e - min(events.values()) for e in events.values()]
    for hemi, em, im in product(utils.HEMIS, args.extract_mode, args.inverse_method):
        output_fname = output_template.format(hemi=hemi, em=em, im=im)
        if op.isfile(output_fname) and not args.overwrite_labels_power_spectrum:
            continue
        labels_fname = get_labels_fname(subject, hemi, args)
        if not op.isfile(labels_fname):
            print('Can\'t find the labels file! {}'.format(labels_fname))
            continue
        labels_dict = utils.Bag(np.load(labels_fname))
        data = labels_dict.data[em] if isinstance(labels_dict.data, dict) else labels_dict.data
        if data.ndim == 3 and data.shape[2] > len(events) or args.calc_spectrum_with_no_windows:
            data = data.reshape((data.shape[0], -1))
        first_time = True
        now, N = time.time(), len(labels_dict.names) * len(events)
        for cond_id, label_ind in product(events_ids, range(len(labels_dict.names))):
            utils.time_to_go(now, label_ind, N, 10)
            label_data = data[label_ind, :, cond_id] if data.ndim == 3 else data[label_ind, :]
            frequencies, linear_spectrum = utils.power_spectrum(label_data, info.sfreq)
            if first_time:
                power_spectrum = np.zeros((data.shape[0], len(frequencies), len(events)))
                first_time = False
            power_spectrum[label_ind, :, cond_id] = linear_spectrum
            if do_plot:
                plt.figure()
                plt.plot(frequencies, linear_spectrum)
                plt.xlabel('frequency [Hz]')
                plt.ylabel('Linear spectrum [V RMS]')
                plt.title('Power spectrum (scipy.signal.welch)')
                plt.show()
        np.savez(output_fname, data=power_spectrum, frequencies=frequencies, names=labels_dict.names,
                 conditions=labels_dict.conditions)
    return np.all([utils.both_hemi_files_exist(output_template.format(em=em, im=im, hemi='{hemi}'))
                   for em, im in product(args.extract_mode, args.inverse_method)])


def get_labels_fname(subject, hemi, args, modality='meg'):
    labels_fnames = glob.glob(op.join(MMVT_DIR, subject, modality_fol(modality), 'labels_data*{}*{}_{}.npz'.format(
        args.atlas, args.extract_mode, hemi)))
    if args.task != '':
        labels_task_fnames = [l for l in labels_fnames if args.task.lower() in utils.namebase(l.lower())]
        if len(labels_task_fnames) > 0:
            labels_fnames = labels_task_fnames
    labels_fname = utils.select_one_file(labels_fnames)
    return labels_fname


@check_globals()
def get_info(subject, epochs_fname='', evoked_fname='', raw_fname='', bad_channels=[], info_fname='',
             fwd_usingEEG=True, fwd_usingMEG=True):
    info = None
    epochs_fname = get_epo_fname(epochs_fname)
    evoked_fname = get_evo_fname(subject, evoked_fname)
    raw_fname = get_raw_fname(raw_fname)
    info_fname, info_exist = get_info_fname(subject, info_fname)
    if info_exist:
        try:
            info = utils.load(info_fname)
            if not info_consist(info):
                print('Info from pkl file info_consist is False!')
                info = None
        except:
            utils.print_last_error_line()
            info = None
    if info is None and op.isfile(evoked_fname):
        evoked = mne.read_evokeds(evoked_fname)
        if isinstance(evoked, list):
            evoked = evoked[0]
        info = evoked.info
        if not info_consist(info):
            info = None
    if info is None and op.isfile(epochs_fname):
        epochs = mne.read_epochs(epochs_fname)
        info = epochs.info
        if not info_consist(info):
            info = None
    if info is None and op.isfile(raw_fname):
        info = read_info_from_raw(raw_fname)
    if not info_consist(info):
        info = None
    if info is None:
        print('Can\'t find info object in:')
        print('epochs: {}\nevoked: {}\nraw: {}'.format(epochs_fname, evoked_fname, raw_fname))
        return None
    if set(info['bads']) == set(bad_channels):
        return info

    bad_channels = get_bad_channels(info, bad_channels, fwd_usingEEG, fwd_usingMEG)
    if set(info['bads']) != set(bad_channels):
        info['bads'] = list((set(info['bads']).union(set(bad_channels))))
        utils.save(info, info_fname)
    #     info = utils.Bag(info)
    return info


def read_info_from_raw(raw_fname):
    # Maybe the raw isn't really a raw file
    try:
        raw = mne.io.read_raw_fif(raw_fname)
        return raw.info
    except:
        info = None
    try:
        evoked = mne.read_evokeds(raw_fname)
        if isinstance(evoked, list):
            evoked = evoked[0]
        return evoked.info
    except:
        info = None
    try:
        epochs = mne.read_epochs(raw_fname)
        return epochs.info
    except:
        info = None
    return None


def info_consist(info):
    try:
        info._check_consistency()
        return True
    except:
        return False


@check_globals()
def get_info_fname(subject, info_fname=''):
    # The subject variable is needed if the function is called outside this module
    if info_fname == '':
        info_fname = INFO
    info_fname, info_exist = locating_meg_file(info_fname, '*-info.pkl')
    return info_fname, info_exist


@check_globals()
def read_sensors_layout(mri_subject, args=None, pick_meg=True, pick_eeg=False, overwrite_sensors=False,
                        raw_template='', trans_file='', info_fname='', info=None, read_info_file=True,
                        raw_fname=''):
    from mne.io import _loc_to_coil_trans
    from mne.forward import _create_meg_coils
    from mne.viz._3d import _sensor_shape
    from mne.transforms import apply_trans

    if not op.isfile(raw_fname):
        raw_fname = get_raw_fname(raw_fname)
    if pick_meg:
        all_exist = all([op.isfile(op.join(
            MMVT_DIR, mri_subject, 'meg', 'meg_{}_sensors_positions.npz'.format(sensor_type)))
            for sensor_type in ['mag', 'planar1', 'planar2']])
    elif pick_eeg:
        all_exist = op.isfile(op.join(MMVT_DIR, mri_subject, 'eeg', 'eeg_sensors_positions.npz'))
    if all_exist and not overwrite_sensors:
        return True

    if pick_eeg and pick_meg or (not pick_meg and not pick_eeg):
        raise Exception('read_sensors_layout: You should pick only meg or eeg!')
    try:
        if isinstance(trans_file, str) and not op.isfile(trans_file):
            remote_subject_dir = args.remote_subject_dir if args is not None else ''
            trans_file = find_trans_file(trans_file, remote_subject_dir, mri_subject, SUBJECTS_MRI_DIR)
        else:
            ok_trans_files = filter_trans_files([trans_file] if isinstance(trans_file, str) else trans_file)
            if len(ok_trans_files) == 1:
                trans_file = ok_trans_files[0]
            else:
                print('Wrong trans file!')
                return False
    except:
        utils.print_last_error_line()
        return False
    if not op.isfile(trans_file):
        print('read_sensors_layout: No trans files!')
        return False
    if pick_meg:
        utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg'))
        output_fname_template = op.join(MMVT_DIR, mri_subject, 'meg', 'meg_{sensors_type}_sensors_positions.npz')
    else:
        utils.make_dir(op.join(MMVT_DIR, mri_subject, 'eeg'))
        output_fname_template = op.join(MMVT_DIR, mri_subject, 'eeg', 'eeg_sensors_positions.npz')

    if info is None:
        info_fname, info_exist = get_info_fname(info_fname)
        if not info_exist or not read_info_file:
            if not op.isfile(raw_fname):
                raw_fname, raw_exist = locating_meg_file(RAW, raw_template)
                if not raw_exist:
                    print('No raw or raw info file!')
                    return False
            info = read_info_from_raw(raw_fname)
            # raw = mne.io.read_raw_fif(raw_fname)
            # info = raw.info
            utils.save(info, info_fname)
        else:
            info = utils.load(info_fname)
    if pick_meg:
        # ref_meg = ?
        sensors_picks = {sensor_type: mne.io.pick.pick_types(info, meg=sensor_type, exclude=[]) for sensor_type in
                         # , exclude='bads'
                         ['mag', 'planar1', 'planar2']}
    else:
        sensors_picks = {
            sensor_type: mne.io.pick.pick_types(info, meg=pick_meg, eeg=pick_eeg, exclude=[])  # , exclude='bads'
            for sensor_type in ['eeg']}

    trans = mne.transforms.read_trans(trans_file)
    head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
    dev_head_t = info['dev_head_t']
    meg_trans = mne.transforms.combine_transforms(dev_head_t, head_mri_t, 'meg', 'mri')

    for sensors_type, picks in sensors_picks.items():
        output_fname = output_fname_template.format(sensors_type=sensors_type)
        if op.isfile(output_fname) and not overwrite_sensors:
            continue
        if pick_meg:
            coil_transs = [_loc_to_coil_trans(info['chs'][pick]['loc']) for pick in picks]
            coils = _create_meg_coils([info['chs'][pick] for pick in picks], acc='normal')
            offset = 0
            meg_tris, meg_rrs = [], []
            for coil, coil_trans in zip(coils, coil_transs):
                rrs, tris = _sensor_shape(coil)
                rrs = apply_trans(coil_trans, rrs)
                meg_rrs.append(np.mean(rrs, axis=0))  # rrs
                # meg_tris.append(tris + offset)
                # offset += len(meg_rrs[-1])
            if len(meg_rrs) == 0:
                print('MEG sensors not found. Cannot plot MEG locations.')
            else:
                sensors_pos = apply_trans(meg_trans, meg_rrs)  # np.concatenate(meg_rrs, axis=0))
                # meg_tris = np.concatenate(meg_tris, axis=0)
        elif pick_eeg:
            sensors_pos = np.array([info['chs'][k]['loc'][:3] for k in picks])
            sensors_pos = apply_trans(head_mri_t, sensors_pos)
        if len(sensors_pos) == 0:
            print('{}: No sensors found!'.format(sensors_type))
            continue
        sensors_names = np.array([info['ch_names'][k].replace(' ', '') for k in picks])
        if 'Event' in sensors_names:
            event_ind = np.where(sensors_names == 'Event')[0]
            sensors_names = np.delete(sensors_names, event_ind)
            sensors_pos = np.delete(sensors_pos, event_ind)
        head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')

        sensors_pos *= 1000
        print('Saving sensors pos in {}'.format(output_fname))
        np.savez(output_fname, pos=sensors_pos, names=sensors_names, picks=picks)
    return True


@utils.tryit()
def create_helmet_mesh(subject, excludes=[], overwrite_faces_verts=True, sensors_type='mag', modality='meg',
                       overwrite=False):
    from scipy.spatial import Delaunay
    from src.utils import trig_utils
    mesh_ply_fname = op.join(MMVT_DIR, subject, modality_fol(modality), '{}_helmet.ply'.format(modality))
    if op.isfile(mesh_ply_fname) and not overwrite:
        print('{} mesh already exist!'.format(modality))
        return True
    input_file = op.join(MMVT_DIR, subject, modality_fol(modality), '{}_{}sensors_positions.npz'.format(
        modality, '{}_'.format(sensors_type) if modality == 'meg' else ''))
    if not op.isfile(input_file):
        print('Can\'t find {}! Run the read_sensors_layout function first'.format(input_file))
        return False
    faces_verts_out_fname = op.join(MMVT_DIR, subject, modality_fol(modality), '{}_faces_verts.npy'.format(modality))
    f = np.load(input_file)
    verts = f['pos']
    # excluded_inds = [np.where(f['names'] == e)[0] for e in excludes]
    # verts = np.delete(verts, excluded_inds, 0)
    verts_tup = [(x, y, z) for x, y, z in verts]
    tris = Delaunay(verts_tup)
    faces = tris.convex_hull
    areas = [trig_utils.poly_area(verts[poly]) for poly in tris.convex_hull]
    inds = [k for k, s in enumerate(areas) if s > np.percentile(areas, 97)]
    faces = np.delete(faces, inds, 0)
    utils.write_ply_file(verts, faces, mesh_ply_fname, True)
    utils.calc_ply_faces_verts(verts, faces, faces_verts_out_fname, overwrite_faces_verts,
                               utils.namebase(faces_verts_out_fname))
    np.savez(input_file, pos=f['pos'], names=f['names'], tri=faces, excludes=excludes)
    return True


# @check_globals()
def find_trans_file(trans_file='', remote_subject_dir='', subject='', subjects_dir='', silent=False):
    subject = MRI_SUBJECT if subject == '' else subject
    subjects_dir = SUBJECTS_MRI_DIR if subjects_dir == '' else subjects_dir
    # trans_file = COR if trans_file == '' else trans_file
    if not op.isfile(trans_file):
        # trans_files = glob.glob(op.join(subject_dir, subject, '**', '*COR*.fif'), recursive=True)
        trans_files = glob.glob(op.join(subjects_dir, subject, 'mri', 'T1-neuromag', 'sets', '*.fif'))
        if len(trans_files) == 0:
            trans_files += utils.find_recursive(op.join(subjects_dir, subject), '*COR*.fif')
        if len(trans_files) == 0 and remote_subject_dir != '':
            # trans_files = glob.glob(op.join(remote_subject_dir, '**', '*COR*.fif'), recursive=True)
            trans_files = utils.find_recursive(op.join(remote_subject_dir), '*COR*.fif')
        if len(trans_files) == 0:
            # trans_files = glob.glob(op.join(utils.get_parent_fol(trans_file), '**', '*COR*.fif'), recursive=True)
            trans_files = utils.find_recursive(op.join(utils.get_parent_fol(trans_file)), '*COR*.fif')
        # bem_trans_files = glob.glob(op.join(subjects_dir, subject, 'bem', '*-head.fif'))
        if len(trans_files) == 0:
            trans_files = utils.find_recursive(op.join(subjects_dir, subject, 'bem'), '*-head.fif')
        if len(trans_files) == 0:
            # trans_files += glob.glob(op.join(remote_subject_dir, 'bem', '*-head.fif'), recursive=True)
            trans_files = utils.find_recursive(op.join(remote_subject_dir, 'bem'), '*-head.fif')
        ok_trans_files = filter_trans_files(trans_files)
        trans_file = utils.select_one_file(
            ok_trans_files, template='*COR*.fif', files_desc='MRI-Head transformation',
            file_func=lambda fname: read_trans(fname))
    if not silent:
        if op.isfile(trans_file):
            print('trans file was found in {}'.format(trans_file))
        else:
            raise Exception('trans file wasn\'t found!')
    return trans_file


def filter_trans_files(trans_files):
    ok_trans_files = []
    for trans_file in trans_files:
        try:
            trans = mne.transforms.read_trans(trans_file)
            head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
            if trans is not None and not np.all(head_mri_t['trans'] == np.eye(4)):
                ok_trans_files.append(trans_file)
        except:
            pass
    return ok_trans_files


def read_trans(fname):
    try:
        print(mne.transforms.read_trans(fname))
    except:
        print('Not a trans file')


# def plot_labels_data(plot_each_label=False):
#     import matplotlib.pyplot as plt
#     plt.close('all')
#     for hemi in HEMIS:
#         plt.figure()
#         d = np.load(LBL.format(hemi))
#         for cond_id, cond_name in enumerate(d['conditions']):
#             figures_fol = op.join(SUBJECT_MEG_FOLDER, 'figures', hemi, cond_name)
#             if not op.isdir(figures_fol):
#                 os.makedirs(figures_fol)
#             for name, data in zip(d['names'], d['data'][:,:,cond_id]):
#                 if plot_each_label:
#                     plt.figure()
#                 plt.plot(data, label=name)
#                 if plot_each_label:
#                     plt.title('{}: {} {}'.format(cond_name, hemi, name))
#                     plt.xlabel('time (ms)')
#                     plt.savefig(op.join(figures_fol, '{}.jpg'.format(name)))
#                     plt.close()
#             # plt.legend()
#             if not plot_each_label:
#                 plt.title('{}: {}'.format(cond_name, hemi))
#                 plt.xlabel('time (ms)')
#                 plt.show()


def check_both_hemi_in_stc(events, inverse_method):
    for ind, cond in enumerate(events.keys()):
        stcs = {}
        for hemi in HEMIS:
            stcs[hemi] = mne.read_source_estimate(STC_HEMI.format(cond=cond, method=inverse_method, hemi=hemi))
        print(np.all(stcs['rh'].rh_data == stcs['lh'].rh_data))
        print(np.all(stcs['rh'].lh_data == stcs['lh'].lh_data))


def check_labels():
    import matplotlib.pyplot as plt

    data, names = [], []
    for hemi in HEMIS:
        # todo: What?
        f = np.load('/homes/5/npeled/space3/visualization_blender/fsaverage/pp003_Fear/labels_data_{}.npz'.format(hemi))
        data.append(f['data'])
        names.extend(f['names'])
    d = np.vstack((d for d in data))
    plt.plot((d[:, :, 0] - d[:, :, 1]).T)
    t_range = range(0, 1000)
    dd = d[:, t_range, 0] - d[:, t_range, 1]
    print(dd.shape)
    dd = np.sqrt(np.sum(np.power(dd, 2), 1))
    print(dd)
    objects_to_filtter_in = np.argsort(dd)[::-1][:2]
    print(objects_to_filtter_in)
    print(dd[objects_to_filtter_in])
    return objects_to_filtter_in, names


def test_labels_coloring(subject, atlas):
    T = 2500
    labels_fnames = glob.glob(op.join(SUBJECTS_MRI_DIR, subject, 'label', atlas, '*.label'))
    labels_names = defaultdict(list)
    for label_fname in labels_fnames:
        label = mne.read_label(label_fname)
        labels_names[label.hemi].append(label.name)

    for hemi in HEMIS:
        L = len(labels_names[hemi])
        data, data_no_t = np.zeros((L, T)), np.zeros((L))
        for ind in range(L):
            data[ind, :] = (np.sin(np.arange(T) / 100 - np.random.rand(1) * 100) +
                            np.random.randn(T) / 100) * np.random.rand(1)
            data_no_t[ind] = data[ind, 0]
        colors = utils.mat_to_colors(data)
        colors_no_t = utils.arr_to_colors(data_no_t)[:, :3]
        np.savez(op.join(MMVT_SUBJECT_FOLDER, 'meg', 'meg_labels_coloring_{}.npz'.format(hemi)),
                 data=data, colors=colors, names=labels_names[hemi])
        np.savez(op.join(MMVT_SUBJECT_FOLDER, 'meg', 'meg_labels_coloring_no_t{}.npz'.format(hemi)),
                 data=data_no_t, colors=colors_no_t, names=labels_names[hemi])
        # plt.plot(range(T), data.T)
        # plt.show()


def misc():
    # check_labels()
    # Morph and move to mg79
    # morph_stc('mg79', 'all')
    # initGlobals('mg79')
    # readLabelsData()
    # plot3DActivity()
    # plot3DActivity()
    # morphTOTlrc()
    # stc = read_source_estimate(STC)
    # plot3DActivity(stc)
    # permuationTest()
    # check_both_hemi_in_stc(events)
    # lut = utils.read_freesurfer_lookup_table(FREESURFER_HOME)
    pass


def get_fname_format_args(args):
    return get_fname_format(
        args.task, args.fname_format, args.fname_format_cond, args.conditions, args.get_task_defaults)


def get_fname_format(task, fname_format='', fname_format_cond='', args_conditions='', get_task_defaults=True):
    conditions = None
    if get_task_defaults:
        if task == 'MSIT':
            if fname_format_cond == '':
                fname_format_cond = '{subject}_msit_{cleaning_method}_{contrast}_{cond}_1-15-{ana_type}.{file_type}'
            if fname_format == '':
                fname_format = '{subject}_msit_{cleaning_method}_{contrast}_1-15-{ana_type}.{file_type}'
            conditions = dict(interference=1,
                              neutral=2)  # dict(congruent=1, incongruent=2), events = dict(Fear=1, Happy=2)
        elif task == 'ECR':
            if fname_format_cond == '':
                fname_format_cond = '{subject}_ecr_{cond}_15-{ana_type}.{file_type}'
            if fname_format == '':
                fname_format = '{subject}_ecr_15-{ana_type}.{file_type}'
            # conditions = dict(Fear=1, Happy=2) # or dict(congruent=1, incongruent=2)
            conditions = dict(C=1, I=2)
            # event_digit = 3
        elif task == 'ARC':
            if fname_format_cond == '':
                fname_format_cond = '{subject}_arc_rer_{cleaning_method}_{cond}-{ana_type}.{file_type}'
            if fname_format == '':
                fname_format = '{subject}_arc_rer_{cleaning_method}-{ana_type}.{file_type}'
            conditions = dict(low_risk=1, med_risk=2, high_risk=3)
        elif task == 'audvis':
            if fname_format_cond == '':
                fname_format_cond = '{subject}_audvis_{cond}_{ana_type}.{file_type}'
            if fname_format == '':
                fname_format = '{subject}_audvis_{ana_type}.{file_type}'
            conditions = dict(LA=1, RA=2, LV=3, RV=4, smiley=5, button=32)
        elif task == 'rest':
            if fname_format == '' or fname_format == '{subject}-{ana_type}.{file_type}':
                fname_format = fname_format_cond = '{subject}_{cleaning_method}-rest-{ana_type}.{file_type}'
            conditions = dict(rest=1)
        elif task == 'epilepsy':
            if fname_format == '' or fname_format == '{subject}-{ana_type}.{file_type}':
                fname_format = fname_format_cond = '{subject}_{cleaning_method}-epilepsy-{ana_type}.{file_type}'
            conditions = dict(epilepsy=1)
        elif task == 'tms':
            if fname_format == '' or fname_format == '{subject}-{ana_type}.{file_type}':
                fname_format = fname_format_cond = '{subject}_-EEG-TMS-{ana_type}.{file_type}'
            conditions = dict(tms=255)
    if conditions is None:
        if fname_format == '' or fname_format_cond == '':
            raise Exception('Empty fname_format and/or fname_format_cond!')
        # raise Exception('Unkown task! Known tasks are MSIT/ECR/ARC')
        # print('Unkown task! Known tasks are MSIT/ECR/ARC.')
        args_conditions = ['all'] if args_conditions is None or args_conditions == '' else args_conditions
        conditions = dict((cond_name, cond_id + 1) for cond_id, cond_name in enumerate(args_conditions))
    # todo: what if we want to set something like LV=3, RV=4 from the args?
    if args_conditions is not None and args_conditions != '':
        conditions = {}
        for cond_id, cond_name in enumerate(args_conditions):
            if ':' in cond_name:
                conds_tup = cond_name.split(':')
                conditions[conds_tup[0]] = int(conds_tup[1])
            else:
                conditions[cond_name] = int(cond_name) if au.is_int(cond_name) else cond_id + 1
    return fname_format, fname_format_cond, conditions


def prepare_subject_folder(subject, remote_subject_dir, local_subjects_dir, necessary_files, sftp_args,
                           use_subject_anat_folder=False):
    local_subject_dir = op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT) if use_subject_anat_folder else SUBJECT_MEG_FOLDER
    return utils.prepare_subject_folder(
        necessary_files, subject, remote_subject_dir, local_subjects_dir,
        sftp_args.sftp, sftp_args.sftp_username, sftp_args.sftp_domain, sftp_args.sftp_password,
        False, sftp_args.print_traceback, local_subject_dir=local_subject_dir)


def get_meg_files(subject, necessary_fnames, args, events=None):
    fnames = []
    events_keys = events.keys() if events is not None and isinstance(events, dict) else ['all']
    for necessary_fname in necessary_fnames:
        fname = os.path.basename(necessary_fname)
        if '{cond}' in fname:
            fnames.extend([get_cond_fname(fname, event) for event in events_keys])
        else:
            fnames.append(fname)
    # local_fol = op.join(MEG_DIR, args.task)
    prepare_subject_folder(subject, args.remote_subject_meg_dir, args.meg_dir, {'.': fnames}, args)


def calc_fwd_inv_wrapper(subject, args, conditions=None, flags={}, mri_subject=''):
    if not utils.should_run(args, 'make_forward_solution') and not utils.should_run(args, 'calc_inverse_operator'):
        return flags
    if mri_subject == '':
        mri_subject = subject
    inv_fname = get_inv_fname(args.inv_fname, args.fwd_usingMEG, args.fwd_usingEEG, True)
    fwd_fname = get_fwd_fname(args.fwd_fname, args.fwd_usingMEG, args.fwd_usingEEG, True)
    get_meg_files(subject, [inv_fname], args, conditions)
    # todo: do something smarter
    bad_channels_fname = op.join(MMVT_DIR, subject, 'meg', 'bad_channels.pkl')
    utils.save(args.bad_channels, bad_channels_fname)
    if args.overwrite_inv or args.overwrite_fwd or not op.isfile(inv_fname) or \
            (args.inv_calc_subcorticals and not op.isfile(INV_SUB)):
        if utils.should_run(args, 'make_forward_solution') and (not op.isfile(fwd_fname) or args.overwrite_fwd):
            # prepare_subject_folder(
            #     mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
            #     {op.join('mri', 'T1-neuromag', 'sets'): ['COR.fif']}, args)
            cor_fname = get_cor_fname(args.cor_fname) if args.cor_fname != '' else ''
            trans_file = find_trans_file(cor_fname, args.remote_subject_dir)
            if trans_file is None:
                flags['make_forward_solution'] = False
                flags['calc_inverse_operator'] = False
                return flags
            if op.isfile(trans_file):
                trans_fol = utils.make_dir(op.join(SUBJECTS_MRI_DIR, subject, 'mri', 'T1-neuromag', 'sets'))
                local_cor_fname = op.join(trans_fol, utils.namebase_with_ext(trans_file))
                if not op.isfile(local_cor_fname) and utils.get_parent_fol(trans_file) != trans_fol:
                    # if trans_file != trans_fol:
                    #     utils.copy_file(trans_file, trans_fol)
                    if trans_file != COR:
                        utils.copy_file(trans_file, local_cor_fname)
                args.cor_fname = local_cor_fname
            src_dic = dict(bem=['*-{}-{}-src.fif'.format(
                args.recreate_src_spacing[:3], args.recreate_src_spacing[-1])])
            src_dic_ast = dict(bem=['*-{}-{}*-src.fif'.format(
                args.recreate_src_spacing[:3], args.recreate_src_spacing[-1])])
            create_src_dic = dict(
                surf=['lh.{}'.format(args.recreate_src_surface), 'rh.{}'.format(args.recreate_src_surface),
                      'lh.sphere', 'rh.sphere'])
            for nec_file in [src_dic, src_dic_ast, create_src_dic]:
                file_exist, _ = prepare_subject_folder(
                    mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
                    nec_file, args)
                if file_exist:
                    break
            evo_fname = get_evo_fname(subject, args.evo_fname)
            epo_fname = get_epo_fname(args.epo_fname)
            get_meg_files(subject, [evo_fname], args, conditions)
            sub_corticals_codes_file = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
            raw_fname = get_raw_fname(args.raw_fname)
            flags['make_forward_solution'], fwd, fwd_subs = make_forward_solution(
                subject, mri_subject, conditions, raw_fname, epo_fname, evo_fname, fwd_fname, args.cor_fname,
                args.bad_channels,
                args.fwd_usingMEG, args.fwd_usingEEG, args.fwd_calc_corticals, args.fwd_calc_subcorticals,
                sub_corticals_codes_file, args.fwd_recreate_source_space, args.recreate_bem_solution, args.bem_ico,
                args.recreate_src_spacing, args.recreate_src_surface, args.overwrite_fwd, args.remote_subject_dir,
                args.n_jobs, args)
        else:
            flags['make_forward_solution'] = True

        if utils.should_run(args, 'calc_inverse_operator') and flags.get('make_forward_solution', True):
            epo_fname = get_epo_fname(args.epo_fname)
            evo_fname = get_evo_fname(subject, args.evo_fname)
            get_meg_files(subject, [epo_fname, fwd_fname], args, conditions)
            raw_fname = get_raw_fname(args.raw_fname)
            if args.noise_cov_fname == '':
                if args.fwd_usingEEG and args.fwd_usingMEG:
                    noise_cov_fname = NOISE_COV_MEEG
                else:
                    noise_cov_fname = NOISE_COV_MEG if args.fwd_usingMEG else NOISE_COV_EEG
            else:
                noise_cov_fname = args.noise_cov_fname
            flags['calc_inverse_operator'] = calc_inverse_operator(
                subject, conditions, raw_fname, epo_fname, evo_fname, fwd_fname, inv_fname, noise_cov_fname,
                args.empty_fname, args.bad_channels, args.inv_loose, args.inv_depth, args.noise_t_min,
                args.noise_t_max, args.overwrite_inv, args.use_empty_room_for_noise_cov, args.use_raw_for_noise_cov,
                args.overwrite_noise_cov, args.inv_calc_cortical, args.inv_calc_subcorticals,
                args.fwd_usingMEG, args.fwd_usingEEG, args.check_for_channels_inconsistency, args=args)
        else:
            flags['calc_inverse_operator'] = True
    else:
        flags['make_forward_solution'] = True
        flags['calc_inverse_operator'] = True
    return flags


def calc_evokes_wrapper(subject, conditions, args, flags={}, raw=None, mri_subject=''):
    if mri_subject == '':
        mri_subject = subject
    evoked, epochs = None, None
    if utils.should_run(args, 'calc_epochs'):
        necessary_files = calc_epochs_necessary_files(args)
        get_meg_files(subject, necessary_files, args, conditions)
        flags['calc_epochs'], epochs = calc_epochs_wrapper_args(subject, conditions, args, raw)
    if utils.should_run(args, 'calc_epochs') and not flags['calc_epochs']:
        return flags, evoked, epochs

    if utils.should_run(args, 'calc_evokes'):
        flags['calc_evokes'], evoked = calc_evokes(
            epochs, conditions, mri_subject, args.normalize_data, args.epo_fname, args.evo_fname,
            args.norm_by_percentile, args.norm_percs, args.modality, args.calc_max_min_diff,
            args.calc_evoked_for_all_epoches, args.overwrite_evoked, args.task,
            args.set_eeg_reference, args.average_per_event, bad_channels=args.bad_channels)

    return flags, evoked, epochs


def get_stc_hemi_template(stc_template):
    if stc_template.endswith('.stc'):
        stc_template = stc_template[:-4]
    return '{}{}'.format(stc_template, '-{hemi}.stc')


def calc_stc_per_condition_wrapper(subject, conditions, inverse_method, args, flags={}, raw=None, epochs=None):
    stcs_conds, stcs_num = None, {}
    if utils.should_run(args, 'calc_stc'):
        stc_hemi_template = get_stc_hemi_template(args.stc_template)
        if conditions is None or len(conditions) == 0:
            conditions = ['all']
        stc_exist = all([utils.both_hemi_files_exist(stc_hemi_template.format(
            cond=cond, method=im, hemi='{hemi}')) for (im, cond) in product(args.inverse_method, conditions)])
        if stc_exist and not args.overwrite_stc:
            print('stc exist! ({})'.format(','.join([stc_hemi_template.format(
                cond=cond, method=im, hemi='{hemi}') for (im, cond) in product(args.inverse_method, conditions)])))
            return flags, None, {}
        if isinstance(inverse_method, Iterable) and not isinstance(inverse_method, str):
            inverse_method = inverse_method[0]
        inv_fname = get_inv_fname(args.inv_fname, args.fwd_usingMEG, args.fwd_usingEEG)
        evo_fname = get_evo_fname(subject, args.evo_fname)
        get_meg_files(subject, [inv_fname, evo_fname], args, conditions)
        flags['calc_stc'], stcs_conds, stcs_num = calc_stc_per_condition(
            subject, conditions, args.task, args.stc_t_min, args.stc_t_max, inverse_method, args.baseline,
            args.apply_SSP_projection_vectors, args.add_eeg_ref, args.pick_ori, args.single_trial_stc,
            args.calc_source_band_induced_power, args.save_stc, args.snr, args.overwrite_stc, args.stc_template,
            args.raw_fname, args.epo_fname, evo_fname, inv_fname, args.fwd_usingMEG, args.fwd_usingEEG,
            args.apply_on_raw, raw, epochs, args.modality, args.calc_stc_for_all, args.calc_stc_diff,
            args.atlas, args.bands, args.calc_inducde_power_per_label, args.induced_power_normalize_proj,
            args.downsample_r, args.zero_time, args.n_jobs)
    return flags, stcs_conds, stcs_num


def calc_labels_avg_for_rest_wrapper(args, raw=None):
    return calc_labels_avg_for_rest(
        args.subject, args.atlas, args.inverse_method, raw, args.pick_ori, args.extract_mode, args.snr, args.raw_fname,
        args.inv_fname, args.labels_data_template, args.overwrite_stc, args.overwrite_labels_data,
        args.fwd_usingMEG, args.fwd_usingEEG, cond_name='all', positive=False, moving_average_win_size=0,
        save_data_files=True, n_jobs=args.n_jobs)


def calc_labels_avg_for_rest(
        subject, atlas, inverse_method, raw=None, pick_ori=None, extract_modes=['mean_flip'], snr=1, raw_fname='',
        inv_fname='', labels_data_template='', overwrite_stc=False, overwrite_labels_data=False, fwd_usingMEG=True,
        fwd_usingEEG=True, cond_name='all', positive=False, moving_average_win_size=0, save_data_files=True,
        do_plot_time_series=True, modality='meg', n_jobs=6):
    def collect_parallel_results(indices, results, labels_num):
        labels_data_hemi = {}
        for indices_chunk, labels_data_chunk in zip(indices, results):
            for em in extract_modes:
                if em not in labels_data_hemi:
                    labels_data_hemi[em] = np.zeros((labels_num, labels_data_chunk[em].shape[1]))
                labels_data_hemi[em][indices_chunk] = labels_data_chunk[em]
        return labels_data_hemi

    labels_output_fol_template = op.join(
        SUBJECT_MEG_FOLDER, 'rest_{}_labels_data_{}'.format(atlas, '{extract_mode}'))
    if labels_data_template == '':
        labels_data_template = LBL
    min_max_output_template = get_labels_minmax_template(labels_data_template)

    labels_num = lu.get_labels_num(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas)
    labels_files_exist = all([len(glob.glob(op.join(
        SUBJECT_MEG_FOLDER, labels_output_fol_template.format(extract_mode=em), '*.npy'))) == labels_num
                              for em in extract_modes])
    labels_data_exist = all([utils.both_hemi_files_exist(labels_data_template.format('rest', atlas, em, '{hemi}'))
                             for em in extract_modes])
    if (not labels_files_exist and not labels_data_exist) or overwrite_labels_data:
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
        if raw is None:
            raw_fname = get_raw_fname(raw_fname)
            if op.isfile(raw_fname):
                mne.io.read_raw_fif(raw_fname)
            else:
                raise Exception("Can't find the raw file! ({})".format(raw_fname))
        if not isinstance(inverse_method, str) and isinstance(inverse_method, Iterable):
            inverse_method = inverse_method[0]
        inverse_operator = read_inverse_operator(inv_fname.format(cond=cond_name))
        src = inverse_operator['src']
        lambda2 = 1.0 / snr ** 2
        labels_data = {}
        for hemi in utils.HEMIS:
            labels = lu.read_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi=hemi)
            indices = np.array_split(np.arange(len(labels)), n_jobs)
            chunks = [([labels[ind] for ind in indices_chunk], raw, src, inverse_operator, lambda2, inverse_method,
                       extract_modes, pick_ori, save_data_files, labels_output_fol_template, overwrite_stc,
                       do_plot_time_series) for indices_chunk in indices]
            results = utils.run_parallel(calc_stc_labels_parallel, chunks, n_jobs)
            labels_data[hemi] = collect_parallel_results(indices, results, len(labels))

    elif (not labels_data_exist) or overwrite_labels_data:
        labels_data = {}
        for hemi in utils.HEMIS:
            labels_names = lu.get_labels_names(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi)
            indices = np.array_split(np.arange(len(labels_names)), n_jobs)
            chunks = [([labels_names[ind] for ind in indices_chunk], extract_modes, labels_output_fol_template,
                       do_plot_time_series) for indices_chunk in indices]
            results = utils.run_parallel(_load_labels_data_parallel, chunks, n_jobs)
            labels_data[hemi] = collect_parallel_results(indices, results, len(labels_names))

    for em in extract_modes:
        data_max = max([np.max(labels_data[hemi][em]) for hemi in utils.HEMIS])
        data_min = min([np.min(labels_data[hemi][em]) for hemi in utils.HEMIS])
        # data_minmax = utils.get_max_abs(data_max, data_min)
        # factor = -int(utils.ceil_floor(np.log10(data_minmax)))
        factor = 6 if modality == 'eeg' else 12  # micro V for EEG, fT (Magnetometers) and fT/cm (Gradiometers) for MEG
        min_max_output_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'meg', min_max_output_template.format('rest', atlas, em))
        np.savez(min_max_output_fname, labels_minmax=[data_min, data_max])
        if (not labels_data_exist) or overwrite_labels_data:
            for hemi in utils.HEMIS:
                labels_names = lu.get_labels_names(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi)
                save_labels_data(
                    subject, modality, labels_data[hemi], hemi, labels_names, atlas, ['all'], extract_modes,
                    labels_data_template, 'rest', factor, positive, moving_average_win_size)

    output_fol = op.join(MMVT_DIR, MRI_SUBJECT, 'meg')
    flag = all([op.isfile(op.join(output_fol, op.basename(labels_data_template.format('rest', atlas, em, hemi)))) for \
                em, hemi in product(extract_modes, utils.HEMIS)]) and \
           all([op.isfile(op.join(output_fol, min_max_output_template.format('rest', atlas, em))) for em in
                extract_modes])
    return flag


def calc_stc_labels_parallel(p):
    (labels, raw, src, inverse_operator, lambda2, inverse_method, extract_modes, pick_ori,
     save_data_files, labels_output_fol_template, overwrite, do_plot_time_series) = p
    labels_data = {}
    for ind, label in enumerate(labels):
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator, lambda2, inverse_method, label=label, pick_ori=pick_ori)
        for em in extract_modes:
            # label_data = stc.extract_label_time_course(label, src, mode=em, allow_empty=True)
            if em != 'mean_flip':
                print("{} isn't implemented yet for rest data!".format(em))
                continue
            labels_output_fol = labels_output_fol_template.format(extract_mode=em)
            utils.make_dir(labels_output_fol)
            label_data = extract_label_data(label, src, stc)
            label_data = np.squeeze(label_data)
            if do_plot_time_series:
                plot_label_data(label_data, label.name, em, labels_output_fol)
            if save_data_files:
                # fol = utils.make_dir(op.join(SUBJECT_MEG_FOLDER, 'rest_labels_data_{}'.format(em)))
                label_fname = op.join(labels_output_fol, '{}-{}.npy'.format(label.name, em))
                if not op.isfile(label_fname) and not overwrite:
                    np.save(label_fname, label_data)
            if em not in labels_data:
                T = len(stc.times)
                labels_data[em] = np.zeros((len(labels), T))
            labels_data[em][ind, :] = label_data
    return labels_data


def _load_labels_data_parallel(p):
    labels, extract_modes, labels_output_fol_template, do_plot_time_series = p
    labels_data = {}
    for em in extract_modes:
        labels_output_fol = labels_output_fol_template.format(extract_mode=em)
        for ind, label in enumerate(labels):
            label_fname = op.join(labels_output_fol, '{}-{}.npy'.format(label, em))
            label_data = np.load(label_fname).squeeze()
            if do_plot_time_series:
                plot_label_data(label_data, label, em, labels_output_fol)
            if em not in labels_data:
                labels_data[em] = np.zeros((len(labels), len(label_data)))
            try:
                labels_data[em][ind, :] = label_data
            except:
                print('error in {}'.format(label))
                raise Exception(traceback.format_exc())
    return labels_data


def plot_label_data(label_data, label, em, labels_output_fol):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(label_data)
    plt.title('{} {}'.format(label, em))
    plt.savefig(op.join(labels_output_fol, '{}-{}.png'.format(label, em)))
    plt.close()


def extract_label_data(label, src, stc):
    label_flip = mne.label_sign_flip(label, src)[:, None].squeeze()
    label_flip = np.tile(label_flip, (stc.data.shape[1], 1)).T
    label_tc = np.mean(label_flip * stc.data, axis=0)
    return label_tc


@check_globals()
def calc_labels_avg_per_condition_wrapper(
        subject, conditions, atlas, inverse_method, stcs_conds, args, flags={}, stcs_num={}, raw=None, epochs=None,
        mri_subject='', modality='meg'):
    if args.labels_data_template == '':
        args.labels_data_template = op.join(
            MMVT_DIR, mri_subject, modality_fol(modality), 'labels_data_{}_{}_{}_{}_{}.npz')
    labels_data_exist = True
    for em, im in product(args.extract_mode, args.inverse_method):
        labels_data_exist = labels_data_exist and all([check_all_lables_data_exist(
            subject, modality, hemi, args.labels_data_template, args.task, atlas, em, im)
            for hemi in utils.HEMIS])

    if utils.should_run(args, 'calc_labels_avg_per_condition'):
        if labels_data_exist and not args.overwrite_labels_data:
            if utils.should_run(args, 'calc_labels_min_max'):
                flags['calc_labels_min_max'] = calc_labels_minmax(
                    subject, modality, atlas, args.inverse_method, args.extract_mode, args.task,
                    args.labels_data_template, args.overwrite_labels_data)
            return flags

        mri_subject = subject if mri_subject == '' else mri_subject
        if conditions is None or len(conditions) == 0:
            conditions = {'all': 1} if args.task == '' else {args.task: 1}
        conditions_keys = conditions.keys()
        if isinstance(inverse_method, Iterable) and not isinstance(inverse_method, str):
            inverse_method = inverse_method[0]
        args.inv_fname = get_inv_fname(args.inv_fname, args.fwd_usingMEG, args.fwd_usingEEG)
        if args.stc_template == '':
            args.stc_hemi_template = op.join(MMVT_DIR, mri_subject, modality_fol(modality), utils.namebase(STC_HEMI))
        else:
            args.stc_hemi_template = get_stc_hemi_template(args.stc_template)
        stc_fnames = [args.stc_hemi_template.format(cond='{cond}', method=inverse_method, hemi=hemi, modal=modality)
                      for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames + [args.inv_fname], args, conditions)
        stcs_conds, stc_fname = get_stc_conds(
            subject, conditions, inverse_method, args.stc_hemi_template, modality)
        if stcs_conds is None or len(stcs_conds) == 0:
            print('Can\'t find the STCs files! template: {}, conditions: {}'.format(
                args.stc_hemi_template, conditions))
            flags['calc_labels_avg_per_condition'] = False
            return flags
        factor = 6 if modality == 'eeg' else 12  # micro V for EEG, fT (Magnetometers) and fT/cm (Gradiometers) for MEG
        for hemi_ind, hemi in enumerate(HEMIS):
            flags['calc_labels_avg_per_condition_{}'.format(hemi)] = calc_labels_avg_per_condition(
                subject, args.atlas, hemi, conditions, extract_modes=args.extract_mode,
                positive=args.evoked_flip_positive, inverse_method=args.inverse_method,
                moving_average_win_size=args.evoked_moving_average_win_size,
                labels_data_template=args.labels_data_template, task=args.task,
                stcs=stcs_conds, factor=factor, inv_fname=args.inv_fname,
                fwd_usingMEG=args.fwd_usingMEG, fwd_usingEEG=args.fwd_usingEEG,
                stcs_num=stcs_num, read_only_from_annot=args.read_only_from_annot,
                mri_subject=mri_subject, stc_hemi_template=args.stc_hemi_template, overwrite=args.overwrite_labels_data,
                n_jobs=args.n_jobs)
            if stcs_conds and isinstance(stcs_conds[list(conditions_keys)[0]], types.GeneratorType) and hemi_ind == 0:
                # Create the stc generator again for the second hemi
                _, stcs_conds, stcs_num = calc_stc_per_condition_wrapper(
                    subject, conditions, inverse_method, args, flags, raw=raw, epochs=epochs)

    if utils.should_run(args, 'calc_labels_min_max'):
        flags['calc_labels_min_max'] = calc_labels_minmax(
            subject, modality, atlas, args.inverse_method, args.extract_mode, args.task, args.labels_data_template,
            args.overwrite_labels_data)
    return flags


def calc_labels_minmax(subject, modality, atlas, inverse_method, extract_modes, task='', labels_data_template='',
                       overwrite_labels_data=False):
    if labels_data_template == '':
        labels_data_template = LBL
    if isinstance(extract_modes, str):
        extract_modes = [extract_modes]
    min_max_output_template = get_labels_minmax_template(labels_data_template)
    for em, im in product(extract_modes, inverse_method):
        min_max_output_fname = get_minmax_fname(min_max_output_template, im, task, atlas, em)
        min_max_mmvt_output_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'meg', utils.namebase_with_ext(min_max_output_fname))
        print('Saving {} labels minmax to {}'.format(em, min_max_mmvt_output_fname))
        if op.isfile(min_max_output_fname) and op.isfile(min_max_mmvt_output_fname) and not overwrite_labels_data:
            continue
        template = get_labels_data_fname(
            subject, modality, '{hemi}', labels_data_template, im, task, atlas, em)
        if utils.both_hemi_files_exist(template):
            calc_labels_data_minmax(template, im, min_max_output_fname, em)
        else:
            print("Can't find {}!".format(template))
    return np.all([op.isfile(
        op.join(MMVT_DIR, MRI_SUBJECT, 'meg', get_minmax_fname(
            min_max_output_template, im, task, atlas, em))) for em, im in product(extract_modes, inverse_method)])


def calc_labels_data_minmax(labels_data_fname_template='', inverse_method='dSPM', min_max_output_fname='', task='',
                            atlas='', em='mean-flip'):
    if labels_data_fname_template == '':
        labels_data_fname_template = LBL
    if min_max_output_fname == '':
        min_max_output_template = get_labels_minmax_template(labels_data_fname_template)
        min_max_output_fname = get_minmax_fname(min_max_output_template, inverse_method, task, atlas, em)
    labels_data = [np.load(labels_data_fname_template.format(hemi=hemi)) for hemi in utils.HEMIS]
    labels_min, labels_max = _calc_labels_data_minmax(labels_data)
    if labels_data[0]['data'].ndim > 2:
        labels_diff_min = min([np.min(np.diff(d['data'])) for d in labels_data])
        labels_diff_max = max([np.max(np.diff(d['data'])) for d in labels_data])
    else:
        labels_diff_min, labels_diff_max = labels_min, labels_max
    np.savez(min_max_output_fname, labels_minmax=[labels_min, labels_max],
             labels_diff_minmax=[labels_diff_min, labels_diff_max])
    print('{}: min: {}, max: {}'.format(em, labels_min, labels_max))
    print('{}: labels diff min: {}, labels diff max: {}'.format(em, labels_diff_min, labels_diff_max))


def _calc_labels_data_minmax(hemis_data):
    hemis_data = hemis_data.values() if isinstance(hemis_data, dict) else hemis_data
    labels_min = min([np.min(d['data']) for d in hemis_data])
    labels_max = max([np.max(d['data']) for d in hemis_data])
    return labels_min, labels_max


def get_labels_data_fname(
        subject, modality, hemi, labels_data_template='', inverse_method='dSPM', task='', atlas='dkt', em='mean_flip'):
    if labels_data_template == '':
        labels_data_template = get_labels_data_template(subject, modality)
    _task = task.lower() if task.lower() not in labels_data_template else ''
    _atlas = atlas if atlas not in labels_data_template else ''
    _inverse_method = inverse_method if inverse_method not in labels_data_template else ''
    _em = em if em not in labels_data_template else ''
    labels_data_fname = labels_data_template.format(
        _task, _atlas, _inverse_method, _em, hemi).replace('__', '_')
    return labels_data_fname


def get_minmax_fname(min_max_output_template, inverse_method, task, atlas, em):
    return min_max_output_template.format(task.lower(), atlas, inverse_method, em).replace('__', '_')


def get_labels_minmax_template(labels_data_template):
    return '{}_minmax.npz'.format(labels_data_template[:-7])


def calc_stc_diff(stc1_fname, stc2_fname, output_name, modality='meg'):
    stc1 = mne.read_source_estimate(stc1_fname)
    stc2 = mne.read_source_estimate(stc2_fname)
    stc_diff = stc1 - stc2
    output_name = output_name[:-len('-lh.stc')]
    stc_diff.save(output_name)
    mmvt_fname = utils.make_dir(op.join(MMVT_DIR, MRI_SUBJECT, modality_fol(modality), utils.namebase(output_name)))
    utils.make_dir(utils.get_parent_fol(output_name))
    for hemi in utils.HEMIS:
        if output_name != mmvt_fname:
            utils.copy_file('{}-{}.stc'.format(output_name, hemi),
                            '{}-{}.stc'.format(mmvt_fname, hemi))
            print('Saving to {}'.format('{}-{}.stc'.format(mmvt_fname, hemi)))


def calc_labels_diff(labels_data1_fname, labels_data2_fname, output_fname, inverse_method='dSPM',
                     new_conditions='', norm_data=False):
    labels_data_diff = {}
    for hemi in utils.HEMIS:
        d1 = utils.Bag(np.load(labels_data1_fname.format(hemi=hemi)))
        d2 = utils.Bag(np.load(labels_data2_fname.format(hemi=hemi)))
        if not all(d1.names == d2.names):
            raise Exception('Both labels data has to have the same labels names!')
        if d1.data.shape != d2.data.shape:
            raise Exception('Both labels data has to have the same data dims!')
        if len(d1.conditions) != len(d2.conditions):
            raise Exception('Both labels data has to have the same number of conditions!')
        default_conditions = ['{}-{}'.format(cond1, cond2) for cond1, cond2 in zip(d1.conditions, d2.conditions)]
        if new_conditions == '':
            new_conditions = default_conditions
        else:
            if isinstance(new_conditions, str):
                new_conditions = [new_conditions]
            if len(new_conditions) != len(d1.conditions):
                print('The new conditions has to have the same length like the labels conditions! ({})'.format(
                    len(d1.conditions)))
        labels_data_diff[hemi] = dict(data=d1.data - d2.data, names=d1.names, conditions=new_conditions)
    if norm_data:
        labels_min, labels_max = _calc_labels_data_minmax(labels_data_diff)
        _, labels_minmax = utils.calc_minmax_abs_from_minmax(labels_min, labels_max)
        for hemi in utils.HEMIS:
            labels_data_diff[hemi]['data'] /= labels_minmax
        output_fname = '{}_norm_{}.npz'.format(output_fname[:-len('_{hemi}.npz')], '{hemi}')
    utils.make_dir(utils.get_parent_fol(output_fname.format(hemi='rh')))
    for hemi in utils.HEMIS:
        hemi_output_fname = output_fname.format(hemi=hemi)
        np.savez(hemi_output_fname, **labels_data_diff[hemi])
    min_max_output_fname = '{}_minmax.npz'.format(output_fname.format(hemi='lh')[:-len('_lh.npz')])
    calc_labels_data_minmax(output_fname, inverse_method, min_max_output_fname)
    return utils.both_hemi_files_exist(output_fname) and op.isfile(min_max_output_fname)


def calc_labels_func(subject, task, atlas, inv_method, em, func=None, tmin=None, tmax=None, times=None, time_dim=1,
                     labels_data_output_name='', precentiles=(1, 99), func_name='', norm_data=True, overwrite=False):
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
    labels_data_template_fname = op.join(
        MMVT_DIR, subject, 'meg', 'labels_data_{}_{}_{}_{}_{}{}.npz'.format(
            task.lower(), atlas, inv_method, em, 'norm_' if norm_data else '', '{hemi}'))
    labels_data = {hemi: utils.Bag(np.load(labels_data_template_fname.format(hemi=hemi))) for hemi in utils.HEMIS}
    if func is None:
        func = utils.wrapped_partial(np.mean, axis=1)
    if func_name == '':
        func_name = func.__name__ if hasattr(func, '__name__') else ''
    if labels_data_output_name == '':
        conds = labels_data['rh'].conditions
        conds = '{}-{}'.format(conds[0], conds[1]) if len(conds) == 2 else conds[0]
        labels_data_output_name = '{}_{}_{}{}'.format(conds, atlas, func_name, '_norm' if norm_data else '')
    output_fname = op.join(fol, '{}.npz'.format(labels_data_output_name))
    if op.isfile(output_fname) and not overwrite:
        return True

    data, labels = [], []
    for hemi in utils.HEMIS:
        d = labels_data[hemi]
        hemi_data = d.data
        if times is not None and len(times) == 2 and times[1] > times[0]:
            dt = (times[1] - times[0]) / (hemi_data.shape[time_dim] - 1)
            t_axis = np.arange(times[0], times[1], dt)
            if tmin is not None:
                tmin_ind = np.where(t_axis > tmin)[0][0] - 1
            if tmax is not None:
                tmax_ind = np.where(t_axis > tmax)[0][0] - 1
            if tmin is not None and tmax is not None:
                hemi_data = hemi_data[:, tmin_ind:tmax_ind] if time_dim == 1 else hemi_data[tmin_ind:tmax_ind]
            elif tmin is not None:
                hemi_data = hemi_data[:, tmin:] if time_dim == 1 else hemi_data[tmin:]
            elif tmax is not None:
                hemi_data = hemi_data[:, :tmax] if time_dim == 1 else hemi_data[:tmax]
        data = func(hemi_data) if len(data) == 0 else np.concatenate((data, func(hemi_data)))
        labels.extend(d.names)
    if data.ndim > 1 or data.shape[0] != len(labels):
        print('data ({}) should have one dim, and same length as the labels ({})'.format(data.shape, len(labels)))
        return
    data_minmax = utils.calc_abs_minmax(data, precentiles)
    print('calc_labels_func minmax: {}, {}'.format(-data_minmax, data_minmax))
    title = '{} {}'.format(d.conditions[0], func_name).replace('_', ' ')
    np.savez(output_fname, names=np.array(labels), atlas=atlas, data=data, title=title,
             data_min=-data_minmax, data_max=data_minmax, cmap='BuPu-RdOrYl')
    return op.isfile(output_fname)


def calc_labels_power_bands_from_timeseries(
        subject, task, atlas, inv_method, em, tmin, tmax, precentiles=(1, 99), func_name='power', norm_data=False,
        bands=dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200]),
        overwrite=False):
    labels_data_template_fname = op.join(
        MMVT_DIR, subject, 'meg', 'labels_data_{}_{}_{}_{}_{}.npz'.format(task, atlas, inv_method, em, '{hemi}'))
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
    labels_data = {hemi: utils.Bag(np.load(labels_data_template_fname.format(hemi=hemi))) for hemi in utils.HEMIS}
    conds = labels_data['rh'].conditions
    conds = '{}-{}'.format(conds[0], conds[1]) if len(conds) == 2 else conds[0]
    files_exist = all([op.isfile(op.join(fol, '{}_{}_{}.npz'.format(conds, func_name, band))) for band in bands.keys()])
    if files_exist and not overwrite:
        return True

    data, labels = defaultdict(list), []
    for hemi in utils.HEMIS:
        d = labels_data[hemi]
        dt = (tmax - tmin) / (d.data.shape[1] - 1)
        bands_power = defaultdict(list)
        for label_data in d.data:
            power = utils.calc_bands_power(label_data, dt, bands)
            for band, band_power in power.items():
                bands_power[band].append(band_power)
        for band in bands:
            data[band] = bands_power[band] if len(data[band]) == 0 else \
                np.concatenate((data[band], np.array(bands_power[band])))
        labels.extend(d.names)
    for band in bands.keys():
        labels_data_output_name = '{}_{}_{}'.format(d.conditions[0], func_name, band)
        if data[band].ndim > 1 or data[band].shape[0] != len(labels):
            print('data ({}) should have one dim, and same length as the labels ({})'.format(data[band].shape,
                                                                                             len(labels)))
            continue
        output_fname = op.join(fol, '{}.npz'.format(labels_data_output_name))
        if norm_data:
            data_minmax = utils.calc_abs_minmax(data[band])
            data[band] /= data_minmax
            labels_data_output_name = '{}_norm'.format(labels_data_output_name)
        data_max = utils.calc_max(data[band], norm_percs=precentiles)
        print('calc_labels_func minmax: {}, {}'.format(0, data_max))
        print('calc_labels_power_bands: Saving results in {}'.format(output_fname))
        np.savez(output_fname, names=np.array(labels), atlas=atlas, data=data[band],
                 title=labels_data_output_name.replace('_', ' '), data_min=0, data_max=data_max, cmap='RdOrYl')
    return all([op.isfile(op.join(fol, '{}_{}_{}.npz'.format(conds, func_name, band))) for band in bands.keys()])


def calc_labels_power_bands_diff(subject, task1, task2, precentiles=(1, 99), func_name='power', norm_data=False,
                                 bands=dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55],
                                            high_gamma=[65, 200])):
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
    for band in bands.keys():
        d1 = utils.Bag(np.load(op.join(fol, '{}_{}_{}.npz'.format(task1, func_name, band))))
        d2 = utils.Bag(np.load(op.join(fol, '{}_{}_{}.npz'.format(task2, func_name, band))))
        if d1.atlas != d2.atlas:
            raise Exception('Both labels data should have the same atlas!')
        if not all(d1.names == d2.names):
            raise Exception('Both labels data should have the same labels!')
        data_diff = d1.data - d2.data
        labels_data_output_name = '{}-{}_{}_{}'.format(task1, task2, func_name, band)
        if norm_data:
            data_minmax = utils.calc_abs_minmax(data_diff)
            data_diff /= data_minmax
            labels_data_output_name = '{}_norm'.format(labels_data_output_name)
        data_diff_minmax = utils.calc_abs_minmax(data_diff, precentiles)
        print('calc_labels_power_bands_diff {} minmax: {}, {}'.format(band, -data_diff_minmax, data_diff_minmax))
        np.savez(op.join(fol, '{}.npz'.format(labels_data_output_name)),
                 names=np.array(d1.names), atlas=d1.atlas, data=data_diff,
                 title=labels_data_output_name.replace('_', ' '),
                 data_min=-data_diff_minmax, data_max=data_diff_minmax, cmap='BuPu-RdOrYl')


@check_globals()
def find_functional_rois_in_stc(
        subject, mri_subject, atlas, stc_name, threshold, threshold_is_precentile=True, time_index=None,
        label_name_template='', peak_mode='abs', extract_time_series_for_clusters=True, extract_mode='mean_flip',
        min_cluster_max=0, min_cluster_size=0, clusters_label='', src=None,
        inv_fname='', fwd_usingMEG=True, fwd_usingEEG=True, stc=None, stc_t_smooth=None, verts=None, connectivity=None,
        labels=None, verts_neighbors_dict=None, find_clusters_overlapped_labeles=True,
        save_func_labels=True, recreate_src_spacing='oct6', calc_cluster_contours=True, save_results=True,
        clusters_output_name='', abs_max=True, modality='meg', crop_times=None, avg_stc=False, uuid='',
        clusters_root_fol='', n_jobs=6):
    import mne.stats.cluster_level as mne_clusters

    # clusters_root_fol = op.join(MMVT_DIR, subject, modality_fol(modality), 'clusters')
    if clusters_root_fol == '':
        clusters_root_fol = op.join(MMVT_DIR, subject, modality_fol(modality), 'clusters')
    if isinstance(extract_mode, list):
        extract_mode = extract_mode[0]
    # todo: Should check for an overwrite flag. Not sure why, if the folder isn't being deleted, the code doesn't work
    # utils.delete_folder_files(clusters_root_fol)
    utils.make_dir(clusters_root_fol)
    labels_fol = op.join(SUBJECTS_MRI_DIR, mri_subject, 'label')
    if find_clusters_overlapped_labeles:
        if not utils.check_if_atlas_exist(labels_fol, atlas):
            from src.preproc import anatomy
            anatomy.create_annotation(mri_subject, atlas, n_jobs=n_jobs)
        if not utils.check_if_atlas_exist(labels_fol, atlas):
            raise Exception("find_functional_rois_in_stc: Can't find the atlas {}!".format(atlas))

    if stc_t_smooth is not None:
        verts = check_stc_with_ply(stc_t_smooth, subject=subject)
    else:
        if '{subject}' in stc_name:
            stc_name = stc_name.replace('{subject}', subject)
        if utils.stc_exist(stc_name):
            stc = mne.read_source_estimate('{}-rh.stc'.format(stc_name))
            stc_name = utils.namebase(stc_name)
        if stc is None:
            stc_fname = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg', '{}-lh.stc'.format(stc_name))
            if not op.isfile(stc_fname):
                stc_fname = op.join(SUBJECT_MEG_FOLDER, '{}-lh.stc'.format(stc_name))
            if not op.isfile(stc_fname):
                raise Exception("Can't find the stc file! ({})".format(stc_name))
            stc = mne.read_source_estimate(stc_fname)
        if crop_times is not None and len(crop_times) == 2:
            stc.crop(crop_times[0], crop_times[1])
        if avg_stc:
            stc = stc.mean()
        if time_index is None:
            if label_name_template == '':
                max_vert, time_index = stc.get_peak(
                    time_as_index=True, vert_as_index=True, mode=peak_mode)
                print('peak time index: {}'.format(time_index))
            else:
                max_vert, time_index = find_pick_activity(
                    subject, stc, atlas, label_name_template, hemi='both', peak_mode=peak_mode)
                print('peak time index: {}'.format(time_index))
        stc_t = create_stc_t(stc, time_index, subject)
        if len(verts['rh']) == len(stc.rh_vertno) and len(verts['lh']) == len(stc.lh_vertno):
            stc_t_smooth = stc_t
        else:
            stc_t_smooth = calc_stc_for_all_vertices(stc_t, subject, subject, n_jobs)
        if verts is None:
            verts = check_stc_with_ply(stc_t_smooth, subject=subject)
    if connectivity is None:
        connectivity = anat.load_connectivity(subject)
    # if verts_dict is None:
    #     verts_dict = utils.get_pial_vertices(subject, MMVT_DIR)
    if threshold_is_precentile:
        # todo: implement the case where the treshold is a dictionary (per hemi)
        threshold = np.percentile(stc_t_smooth.data, threshold)
        if threshold < 1e-4:
            ret = input('threshold < 1e-4, do you want to multiply it by 10^9 to get nAmp (y/n)? ')
            if au.is_true(ret):
                threshold *= np.power(10, 9)  # nAmp
    label_name_template_str = label_name_template.replace('*', '').replace('?', '')
    clusters_name = '{}{}{}'.format(stc_name, '-{}'.format(
        label_name_template_str) if label_name_template_str != '' else '', '-{}'.format(
        uuid) if uuid != '' else '')
    clusters_fol = op.join(clusters_root_fol, clusters_name)
    # data_minmax = utils.get_max_abs(utils.min_stc(stc), utils.max_stc(stc))
    # factor = -int(utils.ceil_floor(np.log10(data_minmax)))

    # threshold_max = max([threshold['rh'], threshold['lh']]) if isinstance(threshold, dict) else threshold
    # min_cluster_max = max([min_cluster_max['rh'], min_cluster_max['lh']]) if isinstance(min_cluster_max, dict) \
    #     else min_cluster_max
    # min_cluster_max_max = max(threshold_max, min_cluster_max)
    # if np.max(stc_t_smooth.data) < threshold_max:
    #     print('{}: stc_t max ({}) < threshold ({})!'.format(stc_name, np.max(stc_t_smooth.data), threshold_max))
    #     return True, {hemi: None for hemi in utils.HEMIS}
    clusters_labels = utils.Bag(
        dict(stc_name=stc_name, threshold=threshold, time=time_index, label_name_template=label_name_template,
             values=[], min_cluster_max=min_cluster_max, min_cluster_size=min_cluster_size,
             clusters_label=clusters_label))
    contours, output_stc_data = {}, {}
    for hemi in utils.HEMIS:
        stc_data = (stc_t_smooth.rh_data if hemi == 'rh' else stc_t_smooth.lh_data).squeeze()
        if np.max(stc_data) < 1e-4:
            stc_data *= np.power(10, 9)
        threshold_hemi = threshold[hemi] if isinstance(threshold, dict) else threshold
        min_cluster_max_hemi = min_cluster_max[hemi] if isinstance(min_cluster_max, dict) else min_cluster_max
        if np.max(stc_data) < min_cluster_max_hemi:
            print('No vertices in {} > {}, continue'.format(hemi, min_cluster_max_hemi))
            output_stc_data[hemi] = np.ones((stc_data.shape[0], 1)) * -1
            continue
        print('Calculating clusters for threshold {}'.format(threshold_hemi))
        clusters, _ = mne_clusters._find_clusters(stc_data, threshold_hemi, adjacency=connectivity[hemi])
        if len(clusters) == 0:
            print('No clusters where found for {}-{}!'.format(stc_name, hemi))
            output_stc_data[hemi] = np.ones((stc_data.shape[0], 1)) * -1
            continue
        print('{} cluster were found for {}'.format(len(clusters), hemi))
        labels_hemi = None if labels is None else labels[hemi]
        if find_clusters_overlapped_labeles:
            clusters_labels_hemi, output_stc_data[hemi] = lu.find_clusters_overlapped_labeles(
                subject, clusters, stc_data, atlas, hemi, verts[hemi], labels_hemi, min_cluster_max_hemi,
                min_cluster_size, clusters_label, abs_max, n_jobs)
        else:
            clusters_labels_hemi = []
            output_stc_data[hemi] = np.ones((stc_data.shape[0], 1)) * -1
            for cluster_ind, cluster in enumerate(clusters):
                x = stc_data[cluster]
                cluster_max = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
                if abs(cluster_max) < min_cluster_max_hemi or len(cluster) < min_cluster_size:
                    continue
                output_stc_data[hemi][cluster, 0] = x
                max_vert_ind = np.argmin(x) if abs(np.min(x)) > abs(np.max(x)) else np.argmax(x)
                max_vert = cluster[max_vert_ind]
                clusters_labels_hemi.append(dict(vertices=cluster, intersects=[], name='{}{}'.format(hemi, cluster_ind),
                                                 coordinates=verts[hemi][cluster], max=cluster_max, hemi=hemi,
                                                 size=len(cluster), max_vert=max_vert))
        if clusters_labels_hemi is None or len(clusters_labels_hemi) == 0:
            print("Can't find overlapped_labeles in {}-{}!".format(stc_name, hemi))
        else:
            clusters_labels_hemi, clusters_cortical_labels = calc_cluster_labels(
                subject, mri_subject, stc, clusters_labels_hemi, clusters_fol, extract_time_series_for_clusters,
                save_func_labels, extract_mode, src, inv_fname, time_index, fwd_usingMEG, fwd_usingEEG,
                recreate_src_spacing=recreate_src_spacing)
            if len(clusters_labels_hemi) > 0 and calc_cluster_contours:
                new_atlas_name = 'clusters-{}-{}'.format(utils.namebase(clusters_fol), hemi)
                contours[hemi] = calc_contours(
                    subject, new_atlas_name, hemi, clusters_cortical_labels, clusters_fol, mri_subject,
                    verts, verts_neighbors_dict)
            clusters_labels.values.extend(clusters_labels_hemi)
    output_stc_data = np.concatenate([output_stc_data['lh'], output_stc_data['rh']])
    output_stc = mne.SourceEstimate(output_stc_data, stc_t_smooth.vertices, 0, 0, subject=subject)
    output_stc.save(op.join(clusters_root_fol, stc_name))
    if save_results:
        if clusters_output_name == '':
            clusters_output_name = 'clusters_labels_{}.pkl'.format(stc_name, atlas)
        clusters_output_fname = op.join(clusters_root_fol, clusters_output_name)
        print('Saving clusters labels: {}'.format(clusters_output_fname))
        # Change Bag to regular dict because we want to load the pickle file in Blender (argggg)
        for ind in range(len(clusters_labels.values)):
            clusters_labels.values[ind] = dict(**clusters_labels.values[ind])
        clusters_labels = dict(**clusters_labels)
        utils.save(clusters_labels, clusters_output_fname)
    return True, contours


def accumulate_stc(subject, stc_org, t_from, t_to, threshold, lookup_atlas, reverse=True, set_t_as_val=False, n_jobs=4):
    data, valid_verts = {}, defaultdict(list)
    verts_labels_lookup = None
    if op.isfile(lookup_atlas):
        verts_labels_lookup = utils.load(lookup_atlas)
    else:
        lookup_fname = op.join(
            MMVT_DIR, subject, '{}_vertices_labels_lookup.pkl'.format(lookup_atlas))
        if op.isfile(lookup_fname):
            verts_labels_lookup = utils.load(lookup_fname)
    time_axis = np.arange(t_from, t_to + 1)
    stc = mne.SourceEstimate(
        stc_org.data[:, t_from:t_to + 1], stc_org.vertices, 0, stc_org.tstep, subject=subject)
    stc = calc_stc_for_all_vertices(stc, subject, subject, n_jobs)
    t0 = time_axis[0]
    time_axis = time_axis - time_axis[0]
    data['rh'] = np.ones((stc.rh_data.shape[0], 1)) * -1
    data['lh'] = np.ones((stc.lh_data.shape[0], 1)) * -1
    for_time = time_axis[::-1] if reverse else time_axis
    now = time.time()
    labels_times = {}
    count_threshold = 50
    for k, t in enumerate(for_time):
        utils.time_to_go(now, k, len(for_time), 100)
        for hemi in utils.HEMIS:
            # todo: check t
            hemi_data = stc.rh_data[:, t] if hemi == 'rh' else stc.lh_data[:, t]
            hemi_threshold = threshold[hemi] if isinstance(threshold, dict) else threshold
            verts = np.where(hemi_data >= hemi_threshold)[0]
            if len(verts) > 0:
                # todo: take the max of data[hemi][verts, 0] and hemi_data[verts]
                data[hemi][verts, 0] = t if set_t_as_val else hemi_data[verts]
                if verts_labels_lookup is not None:
                    labels_count = Counter([verts_labels_lookup[hemi].get(v, None) for v in verts])
                    for label, label_count in labels_count.items():
                        if label is not None and label not in labels_times and label_count > count_threshold:
                            labels_times[label] = stc_org.times[t0 + t]
    data = np.concatenate([data['lh'], data['rh']])
    stc = mne.SourceEstimate(data, stc.vertices, 0, 0, subject=subject)
    return stc, labels_times


def find_pick_activity(subject, stc, atlas, label_name_template='', hemi='both', peak_mode='abs'):
    if isinstance(stc, str):
        stc = mne.read_source_estimate(stc)
    if label_name_template == '':
        max_vert, time_index = stc.get_peak(
            time_as_index=True, vert_as_index=True, mode=peak_mode)
    else:
        hemis = utils.HEMIS if hemi == 'both' else [hemi]
        for hemi in hemis:
            data, vertices = get_stc_data_and_vertices(stc, hemi)
            label_vertices, label_vertices_indices = \
                lu.find_label_vertices(subject, atlas, hemi, vertices, label_name_template)
            label_data = data[label_vertices_indices]
            if peak_mode == 'abs':
                max_vert, time_index = utils.argmax2d(abs(label_data))
            elif peak_mode == 'pos':
                max_vert, time_index = utils.argmax2d(label_data)
            elif peak_mode == 'neg':
                max_vert, time_index = utils.argmax2d(-label_data)
            max_vert = label_vertices_indices[max_vert]
    return max_vert, time_index


def get_stc_data_and_vertices(stc, hemi):
    return (stc.lh_data, stc.lh_vertno) if hemi == 'lh' else (stc.rh_data, stc.rh_vertno)


def calc_cluster_labels(
        subject, mri_subject, stc, clusters, clusters_fol, extract_time_series_for_clusters=True,
        save_labels=True, extract_mode='mean_flip', src=None, inv_fname='', time_index=-1, fwd_usingMEG=True,
        fwd_usingEEG=True, recreate_src_spacing='oct6'):
    utils.make_dir(clusters_fol)
    time_series_fol = op.join(clusters_fol, 'time_series_{}'.format(extract_mode))
    utils.make_dir(time_series_fol)
    if src is None and extract_time_series_for_clusters:
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
        if op.isfile(inv_fname):
            inverse_operator = read_inverse_operator(inv_fname)
            src = inverse_operator['src']
        else:
            # todo: set the recreate_src_spacing according to the stc
            # https://martinos.org/mne/dev/manual/cookbook.html#source-localization
            src = check_src(mri_subject, recreate_src_spacing=recreate_src_spacing)
    labels = []
    for cluster_ind in range(len(clusters)):
        cluster = utils.Bag(clusters[cluster_ind])
        # cluster: vertices, intersects, name, coordinates, max, hemi, size
        cluster_label = mne.Label(
            cluster.vertices, cluster.coordinates, hemi=cluster.hemi, name=cluster.name, subject=subject)
        labels.append(cluster_label)
        cluster_name = 'cluster_size_{}_max_{:.2f}_{}.label'.format(cluster.size, cluster.max, cluster.name)
        if save_labels:
            cluster_label.save(op.join(clusters_fol, cluster_name))

        if extract_time_series_for_clusters:
            label_data = x = stc.extract_label_time_course(
                cluster_label, src, mode=extract_mode, allow_empty=True).squeeze()
            if time_index == -1:
                cluster.ts_max = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
            else:
                cluster.ts_max = x[time_index]
                if cluster.ts_max < 1e-4:
                    cluster.ts_max *= np.power(10, 9)
            if np.all(label_data == 0):
                cluster.label_data = None
            else:
                cluster.label_data = np.squeeze(label_data)
                if np.max(cluster.label_data) < 1e-4:
                    cluster.label_data *= np.power(10, 9)
                np.save(op.join(time_series_fol, '{}.npy'.format(cluster_name)), cluster.label_data)
        else:
            cluster.label_data = None
        clusters[cluster_ind] = cluster
    return clusters, labels


def calc_contours(subject, atlas_name, hemi, clusters_labels, clusters_labels_fol, mri_subject='', verts_dict=None,
                  verts_neighbors_dict=None):
    mri_subject = subject if mri_subject == '' else mri_subject
    # annot_fname = labels_to_annot(atlas_name[:-3], subject, mri_subject, '', hemi, clusters_labels)
    labels_dict = {hemi: [l for l in clusters_labels if l.hemi == hemi] for hemi in utils.HEMIS}
    # if op.isfile(annot_fname):
    #     # for hemi in utils.HEMIS:
    #     # annot_fname = annot_files.format(hemi=hemi)
    #     dest_annot_fname = op.join(clusters_labels_fol, utils.namebase_with_ext(annot_fname))
    #     if op.isfile(dest_annot_fname):
    #         os.remove(dest_annot_fname)
    #     utils.copy_file(annot_fname, dest_annot_fname)
    # else:
    #     print('calc_contours: No annot file!')
    contours, _ = anat.calc_labeles_contours(
        subject, atlas_name[:-3], hemi=hemi, overwrite=True, labels_dict=labels_dict, verts_dict=verts_dict,
        verts_neighbors_dict=verts_neighbors_dict, check_unknown=False, save_lookup=False, return_contours=True)
    return contours[hemi]


def fit_ica(raw=None, n_components=0.95, method='fastica', ica_fname='', raw_fname='', overwrite_ica=False,
            do_plot=False, examine_ica=False, filter_low_freq=1, filter_high_freq=150, n_jobs=6):
    from mne.preprocessing import read_ica
    if raw_fname == '':
        raw_fname = RAW
    if ica_fname == '':
        ica_fname = '{}-{}'.format(op.splitext(raw_fname)[0][:-4], 'ica.fif')
    if op.isfile(ica_fname) and not overwrite_ica:
        ica = read_ica(ica_fname)
    else:
        ica = ICA(n_components=n_components, method=method)
        raw.filter(filter_low_freq, filter_high_freq, n_jobs=n_jobs, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                   filter_length='10s', phase='zero-double')
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ref_meg=False,
                               stim=False, exclude='bads')
        ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
    print(ica)
    if do_plot:
        fig = ica.plot_components(picks=range(max(ica.n_components_, 20)), inst=raw)
        fig_fname = '{}.{}'.format(op.splitext(ica_fname)[0], 'png')
        if not op.isfile(fig_fname):
            fig.savefig(fig_fname)
    if examine_ica:
        output_fname = op.join(utils.get_parent_fol(ica_fname), 'ica_comps.txt')
        comp_num = input('Type the ICA component you want to save: ')
        while not utils.is_int(comp_num):
            print('Please enter a valid integer')
        with open(output_fname, 'a') as f:
            f.write('{},{}\n'.format(utils.namebase_with_ext(ica_fname), comp_num))
    return ica, ica_fname


def find_raw_fname(raw=None, raw_fname='', raw_template='*raw.fif'):
    if op.isfile(raw_fname):
        return raw_fname, True
    if raw is None or (isinstance(raw, str) and not op.isfile(raw)):
        raw_fname, raw_exist = locating_meg_file(raw_fname, glob_pattern=raw_template)
        if not raw_exist:
            return '', False
    elif isinstance(raw, mne.io.fiff.raw.Raw):
        raw_fname = raw.filenames[0]
    if not op.isfile(raw_fname):
        return '', False
    else:
        return raw_fname, True


def remove_artifacts(raw=None, n_max_ecg=3, n_max_eog=1, n_components=0.95, method='fastica',
                     ecg_inds=[], eog_inds=[], eog_channel=None, remove_from_raw=True, save_raw=True, raw_fname='',
                     new_raw_fname='', ica_fname='', overwrite_ica=False, overwrite_raw=False, do_plot=False,
                     raw_template='*raw.fif', n_jobs=6):
    from mne.preprocessing import read_ica
    raw_fname, raw_exist = find_raw_fname(raw, raw_fname, raw_template)
    if not raw_exist:
        print("remove_artifacts: Can't find raw!")
        return False
    if ica_fname == '':
        raw_fname, raw_exist = locating_meg_file(raw_fname, glob_pattern=raw_template)
        ica_fname = '{}-{}'.format(op.splitext(raw_fname)[0][:-4], 'ica.fif')
    if op.isfile(ica_fname) and not overwrite_ica:
        ica = read_ica(ica_fname)
    else:
        if isinstance(raw, str) and op.isfile(raw):
            raw = mne.io.read_raw_fif(raw)
        elif raw is None:
            raw = load_raw(raw_fname, raw_template=raw_template)
        # 1) Fit ICA model using the FastICA algorithm.
        ica, ica_fname = fit_ica(raw, n_components, method, ica_fname, raw_fname, overwrite_ica, do_plot, n_jobs=n_jobs)
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ref_meg=False,
                               stim=False, exclude='bads')
        ###############################################################################
        # 2) identify bad components by analyzing latent sources.
        # generate ECG epochs use detection via phase statistics
        if len(ecg_inds) == 0:
            ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)
            ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
        else:
            scores = [1.0 / len(ecg_inds)] * len(ecg_inds)
        if len(ecg_inds) == 0:
            print('No ECG artifacts!')
        else:
            if do_plot:
                title = 'Sources related to %s artifacts (red)'
                ica.plot_scores(scores, exclude=ecg_inds, title=title % 'ecg', labels='ecg')
                show_picks = np.abs(scores).argsort()[::-1][:5]
                ica.plot_sources(raw, show_picks, exclude=ecg_inds, title=title % 'ecg')
                ica.plot_components(ecg_inds, title=title % 'ecg', colorbar=True)
            ecg_inds = ecg_inds[:n_max_ecg]
            ica.exclude += ecg_inds
        try:
            if len(eog_inds) == 0:
                # detect EOG by correlation # eog_ch = [c for c in raw.ch_names if 'EOG' in c]
                eog_inds, scores = ica.find_bads_eog(raw, ch_name=eog_channel)
            else:
                scores = [1.0 / len(eog_inds)] * len(eog_inds)
            if len(eog_inds) == 0:
                print('No EOG artifacts!')
            else:
                if do_plot:
                    ica.plot_scores(scores, exclude=eog_inds, title=title % 'eog', labels='eog')
                    show_picks = np.abs(scores).argsort()[::-1][:5]
                    ica.plot_sources(raw, show_picks, exclude=eog_inds, title=title % 'eog')
                    ica.plot_components(eog_inds, title=title % 'eog', colorbar=True)
                eog_inds = eog_inds[:n_max_eog]
                ica.exclude += eog_inds
        except:
            print("Can't remove EOG artifacts!")
        ###############################################################################
        # 3) Assess component selection and unmixing quality.
        # estimate average artifact
        if do_plot:
            ecg_evoked = ecg_epochs.average()
            print('We found %i ECG events' % ecg_evoked.nave)
            ica.plot_sources(ecg_evoked, exclude=ecg_inds)  # plot ECG sources + selection
            ica.plot_overlay(ecg_evoked, exclude=ecg_inds)  # plot ECG cleaning
            try:
                eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5, picks=picks, ch_name=eog_channel).average()
                print('We found %i EOG events' % eog_evoked.nave)
                ica.plot_sources(eog_evoked, exclude=eog_inds)  # plot EOG sources + selection
                ica.plot_overlay(eog_evoked, exclude=eog_inds)  # plot EOG cleaning
            except:
                print("Can't create eog epochs, no EEG")
            # check the amplitudes do not change
            ica.plot_overlay(raw)  # EOG artifacts remain
        ica.save(ica_fname)

    if len(ica.exclude) == 0:
        print("ICA didn't find any artifacts!")
    else:
        if remove_from_raw:
            raw = ica.apply(raw)
            if save_raw:
                if new_raw_fname == '':
                    new_raw_fname = raw_fname if overwrite_raw else RAW_ICA
                raw.save(new_raw_fname, overwrite=True)
    return True


def remove_artifacts_with_template_matching(ica_subjects='all', meg_root_fol=''):
    subject_ica_fname = '{}-{}'.format(op.splitext(RAW)[0][:-4], 'ica.fif')
    if not op.isfile(subject_ica_fname):
        print('You should first call remove_artifacts to create an ICA file')
        return False
    if meg_root_fol == '':
        meg_root_fol = MEG_DIR
    eog_teampltes = find_eog_template(ica_subjects, meg_root_fol)
    if eog_teampltes is None:
        print('EOG template is None!')
        return False
    raw_fnames_for_eog_template = list(eog_teampltes.keys())
    reference_ica, reference_eog_inds = eog_teampltes[raw_fnames_for_eog_template[0]]
    subject_ica = mne.preprocessing.read_ica(subject_ica_fname)
    icas = [reference_ica, subject_ica]
    template = (0, reference_eog_inds[0])
    fig_template, fig_detected = mne.preprocessing.corrmap(
        icas, template=template, label="blinks", show=True, threshold=.8, ch_type='mag')
    print('hmmm')


def find_eog_template(subjects='all', n_components=0.95, meg_root_fol='', method='fastica'):
    eog_teampltes_fname = op.join(MEG_DIR, 'eog_templates.pkl')
    if op.isfile(eog_teampltes_fname):
        eog_teampltes = utils.load(eog_teampltes_fname)
        return eog_teampltes
    # subject_ica_fname = '{}-{}'.format(op.splitext(RAW)[0][:-4], 'artifacts_removal-ica.fif')
    # subject_ica = mne.preprocessing.read_ica(subject_ica_fname)
    # n_components = subject_ica.n_components
    raw_files = collect_raw_files(subjects, meg_root_fol, excludes=(RAW))
    reject = dict(mag=5e-12, grad=4000e-13)
    valid_raw_files, eog_templates = {}, {}
    template_found = False
    for raw_fname in raw_files:
        raw_ica_fname = '{}-{}'.format(op.splitext(raw_fname)[0][:-4], 'ica.fif')
        if not op.isfile(raw_ica_fname) or raw_fname == RAW:
            continue
        try:
            raw = mne.io.read_raw_fif(raw_fname)
            eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
            if len(eog_epochs) > 0:
                valid_raw_files[raw_fname] = eog_epochs
                if op.isfile(raw_ica_fname):
                    ica = mne.preprocessing.read_ica(raw_ica_fname)
                else:
                    ica = ICA(n_components=n_components, method=method)
                    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ref_meg=False,
                                           stim=False, exclude='bads')
                    ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
                eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
                if len(eog_inds) > 0:
                    eog_templates[raw_fname] = (ica, eog_inds)
                    template_found = True
                    break
                    # ica.plot_components(eog_inds, colorbar=True)
                    # eog_average = eog_epochs.average()
                    # ica.plot_sources(eog_average, exclude=eog_inds)
        except:
            print("Can't examine EOG artifacts on {}".format(raw_fname))
            print(traceback.format_exc())
    if template_found:
        utils.save(eog_templates, eog_teampltes_fname)
        return eog_templates
    else:
        print("Couldn't find any teamplte for EOG!")
        return None


def fit_ica_on_subjects(subjects='all', n_components=0.95, method='fastica', ica_fname='', meg_root_fol='',
                        overwrite_ica=False):
    from mne.preprocessing import ICA

    if ica_fname == '':
        ica_fname = 'ica.fif'
    ica_files_num, raw_files_num = 0, 0
    raw_files = collect_raw_files(subjects, meg_root_fol)
    for raw_fname in raw_files:
        subject_ica_fname = op.join(utils.get_parent_fol(raw_fname), ica_fname)
        if (not op.isfile(subject_ica_fname) or overwrite_ica) and 'empty' not in utils.namebase(raw_fname):
            raw_files_num += 1
    print('{} raw data were found!'.format(raw_files_num))
    for raw_fname in raw_files:
        try:
            subject_ica_fname = '{}-{}'.format(op.splitext(raw_fname)[0][:-4], 'artifacts_removal-ica.fif')
            if op.isfile(subject_ica_fname) and not overwrite_ica:
                ica_files_num += 1
                continue
            # Don't analyze empty room...
            if 'empty' in utils.namebase(raw_fname):
                continue
            print('Fitting ICA on {}'.format(raw_fname))
            raw = mne.io.read_raw_fif(raw_fname, preload=True)
            raw.filter(1, 45, n_jobs=1, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                       filter_length='10s', phase='zero-double')
            picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ref_meg=False,
                                   stim=False, exclude='bads')
            ica = ICA(n_components=n_components, method=method)
            ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
            if not op.isfile(subject_ica_fname) or overwrite_ica:
                ica.save(subject_ica_fname)
            ica_files_num += 1
            del raw, ica
        except:
            print(traceback.format_exc())
    print('{} ICA files were created'.format(ica_files_num))


def collect_raw_files(subjects='all', meg_root_fol='', excludes=()):
    if meg_root_fol == '':
        meg_root_fol = MEG_DIR
    if subjects == 'all' or 'all' in subjects:
        raw_files = glob.glob(op.join(meg_root_fol, '**', '*raw.fif'), recursive=True)
    else:
        raw_files = []
        for subject in subjects:
            subject_raw_file = utils.select_one_file(glob.glob(op.join(meg_root_fol, subject, '*raw.fif')))
            if subject_raw_file is not None:
                raw_files.append(subject_raw_file)
    raw_files = [raw_fname for raw_fname in raw_files if
                 'empty' not in utils.namebase(raw_fname) and utils.namebase(raw_fname) not in excludes]
    return raw_files


def stc_time_average(subject, dt, stc_template='*rh.stc', overwrite=False):
    if stc_template == '':
        stc_template = '*rh.stc'
    stc_files = glob.glob(op.join(MMVT_DIR, subject, 'meg', stc_template))
    stc_fname = utils.select_one_file(stc_files)
    if not op.isfile(stc_fname):
        return False
    new_stc_fname = op.join(utils.get_parent_fol(stc_fname), '{}_{}'.format(
        utils.namebase(stc_fname)[:-len('-rh')], dt))
    new_stc_fname_template = '{}-{}.stc'.format(new_stc_fname, '{hemi}')
    if utils.both_hemi_files_exist(new_stc_fname_template) and not overwrite:
        return True
    stc = mne.read_source_estimate(stc_fname, subject)
    V, T = stc.data.shape
    trim_data = stc.data[:, :-(T % dt)]
    avg_data = trim_data.reshape(V, -1, dt).mean(axis=2)
    residue = stc.data[:, -(T % dt):].mean(axis=1)
    residue = residue[:, np.newaxis]
    avg_data = np.hstack((avg_data, residue))
    avg_data_stc = mne.SourceEstimate(avg_data, stc.vertices, stc.tmin, stc.tstep * dt, subject)
    avg_data_stc.save(new_stc_fname)
    return utils.both_hemi_files_exist(new_stc_fname_template)


def sensors_time_average(subject, dt=10, overwrite=False):
    fol = op.join(MMVT_DIR, subject, 'meg')
    sensors_evoked_files = glob.glob(op.join(fol, '*sensors_evoked_data.npy'))
    if len(sensors_evoked_files) == 0:
        return
    sensors_evoked_fname = utils.select_one_file(sensors_evoked_files)
    if not op.isfile(sensors_evoked_fname):
        return False
    prefix = op.basename(sensors_evoked_fname)[:-len('sensors_evoked_data.npy')]
    sensors_evoked_new_fname = op.join(fol, '{}_{}.npy'.format(utils.namebase(sensors_evoked_fname), dt))
    meta_fname = op.join(fol, '{}sensors_evoked_data_meta.npz'.format(prefix))
    meta_new_fname = op.join(fol, '{}_meta.npz'.format(utils.namebase(sensors_evoked_new_fname), dt))
    evoked_minmax_new_fname = op.join(fol, '{}_minmax.npy'.format(utils.namebase(sensors_evoked_new_fname), dt))
    if op.isfile(sensors_evoked_new_fname) and op.isfile(meta_new_fname) and op.isfile(evoked_minmax_new_fname) \
            and not overwrite:
        return True
    data = np.load(op.join(sensors_evoked_fname))

    C, T, K = data.shape
    trim_data = data[:, :-(T % dt), :]
    avg_data = trim_data.reshape(C, -1, dt, K).mean(axis=2)
    residue = data[:, -(T % dt):, :].mean(axis=1)
    residue = residue[:, np.newaxis, :]
    avg_data = np.hstack((avg_data, residue))
    data_max, data_min = utils.get_data_max_min(avg_data, True, (1, 99))

    np.save(sensors_evoked_new_fname, avg_data)
    meta = utils.Bag(np.load(meta_fname))
    np.savez(meta_new_fname, names=meta.names, conditions=meta.conditions, dt=meta.dt * dt)
    np.save(evoked_minmax_new_fname, [data_min, data_max])
    return op.isfile(sensors_evoked_new_fname) and op.isfile(meta_new_fname) and op.isfile(evoked_minmax_new_fname)


def load_fieldtrip_volumetric_data(subject, data_name, data_field_name,
                                   overwrite_nii_file=False, overwrite_surface=False, overwrite_stc=False):
    import scipy.io as sio
    volumetric_meg_fname = op.join(MEG_DIR, subject, '{}.nii'.format(data_name))
    if not op.isfile(volumetric_meg_fname) or overwrite_nii_file:
        fname = op.join(MEG_DIR, subject, '{}.mat'.format(data_name))
        # load Matlab/Fieldtrip data
        mat = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
        ft_data = mat[data_name]
        data = getattr(ft_data, data_field_name)
        data[np.isnan(data)] = 0
        affine = ft_data.transform
        nib.save(nib.Nifti1Image(data, affine), volumetric_meg_fname)
    surface_output_template = op.join(MEG_DIR, subject, '{}_{}.mgz'.format(data_name, '{hemi}'))
    if not utils.both_hemi_files_exist(surface_output_template) or overwrite_surface:
        fu.project_on_surface(subject, volumetric_meg_fname, surface_output_template, overwrite_surf_data=True,
                              modality='meg')
    stcs_exist = utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'meg', '{}-{}.stc'.format(data_name, '{hemi}')))
    if not stcs_exist or overwrite_stc:
        data = {}
        for hemi in utils.HEMIS:
            data[hemi] = np.load(op.join(MMVT_DIR, subject, 'meg', 'meg_{}_{}.npy'.format(data_name, hemi)))
            data[hemi][np.where(np.isnan(data[hemi]))] = 0
        stc = create_stc_t_from_data(subject, data['rh'], data['lh'])
        stc.save(op.join(MMVT_DIR, subject, 'meg', data_name))
    return stcs_exist


def get_digitization_points(subject, raw_fname):
    raw = mne.io.read_raw_fif(raw_fname)
    info = raw.info
    pos = np.array([p['r'] for p in info['dig']])
    kind = np.array([p['kind'] for p in info['dig']])
    ident = np.array([p['ident'] for p in info['dig']])
    coord_frame = np.array([p['coord_frame'] for p in info['dig']])
    utils.make_dir(op.join(MMVT_DIR, subject, 'meg'))
    output_fname = op.join(MMVT_DIR, subject, 'meg', 'digitization_points.npz')
    np.savez(output_fname, pos=pos, kind=kind, ident=ident, coord_frame=coord_frame)
    return op.isfile(output_fname)


def stc_to_contours(subject, stc_name, pick_t=0, thresholds_min=None, thresholds_max=None, thresholds_dx=1,
                    min_cluster_size=10, atlas='', clusters_label='', find_clusters_overlapped_labeles=False,
                    mri_subject='', stc_t_smooth=None, modality='meg', n_jobs=4):
    '''

    :param subject:
    :param stc_name:
    :param pick_t:
    :param thresholds_min: The minimum activation level to plot
    :param thresholds_max: The maximum activation level to plot
    :param thresholds_dx: The difference between the minimum and maximum activation levels
    :param min_cluster_size: The minimum number of voxels to define a cluster
    :param atlas: The atlas used to map the brain
    :param clusters_label:
    :param find_clusters_overlapped_labeles:
    :param mri_subject:
    :param stc_t_smooth:
    :param modality:
    :param n_jobs:
    :return:
    '''
    if mri_subject == '':
        mri_subject = subject
    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, subject, modality_fol(modality), 'clusters'))
    output_fname = op.join(clusters_root_fol, '{}_contoures_{}.pkl'.format(stc_name, pick_t))
    connectivity = anat.load_connectivity(subject)
    if stc_t_smooth is None:
        stc_t_smooth_fname = op.join(clusters_root_fol, '{}_{}_smooth'.format(stc_name, pick_t))
        stc_fname = op.join(MMVT_DIR, subject, 'meg', '{}-lh.stc'.format(stc_name))
        if not op.isfile(stc_fname):
            raise Exception("Can't find the stc file! ({})".format(stc_name))
        if utils.both_hemi_files_exist('{}-{}.stc'.format(stc_t_smooth_fname, '{hemi}')):
            stc_t_smooth = mne.read_source_estimate(stc_t_smooth_fname)
            verts = utils.get_pial_vertices(subject, MMVT_DIR)
        else:
            stc_fname = op.join(MMVT_DIR, subject, 'meg', '{}-lh.stc'.format(stc_name))
            if not op.isfile(stc_fname):
                raise Exception("Can't find the stc file! ({})".format(stc_name))
            stc = mne.read_source_estimate(stc_fname)
            stc_t = create_stc_t(stc, pick_t, subject)
            stc_t_smooth = calc_stc_for_all_vertices(stc_t, subject, subject, n_jobs)
            stc_t_smooth.save(stc_t_smooth_fname)
            verts = check_stc_with_ply(stc_t_smooth, subject=subject)
    else:
        verts = utils.get_pial_vertices(subject, MMVT_DIR)

    if thresholds_min is None:
        thresholds_min = utils.min_stc(stc_t_smooth)
    if thresholds_max is None:
        thresholds_max = utils.max_stc(stc_t_smooth)
    # thresholds_dx = (thresholds_max - thresholds_min) / contours_num
    thresholds = np.arange(thresholds_min, thresholds_max + thresholds_dx, thresholds_dx)
    print('threshold: {}'.format(thresholds))

    verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors_{}.pkl')
    verts_neighbors_dict = {hemi: utils.load(verts_neighbors_fname.format(hemi)) for hemi in utils.HEMIS}

    all_contours = {}
    now = time.time()
    for run, threshold in enumerate(thresholds):
        key = '{:.2f}'.format(threshold)
        all_contours[key] = {}
        utils.time_to_go(now, run, len(thresholds), 1)
        flag, contours = find_functional_rois_in_stc(
            subject, mri_subject, atlas, stc_name, threshold, threshold_is_precentile=False,
            min_cluster_size=min_cluster_size, time_index=pick_t, extract_time_series_for_clusters=False,
            stc=stc_t_smooth, stc_t_smooth=stc_t_smooth, verts=verts, connectivity=connectivity,  # verts_dict=verts,
            find_clusters_overlapped_labeles=find_clusters_overlapped_labeles, modality=modality,
            verts_neighbors_dict=verts_neighbors_dict, save_results=False, clusters_label=clusters_label,
            n_jobs=n_jobs)
        for hemi in utils.HEMIS:
            if hemi in contours:
                all_contours[key][hemi] = np.where(contours[hemi]['contours'])
    print('Results are saved in {}'.format(output_fname))
    utils.save(all_contours, output_fname)
    return op.isfile(output_fname), all_contours


def average_power_spectrum_per_label(subject, task, inverse_method='dSPM', extract_modes='mean_flip',
                                     conditions=['power_ttest'], labels_filter='', overwrite=False):
    import re
    filter_re = re.compile(labels_filter)
    all_output_files_exist = True
    for extract_mode in extract_modes:
        labels_output_fname = op.join(
            MMVT_DIR, subject, 'meg', 'labels_data_{}_{}_{}_power_spectrum_stat_{}.npz'.format(
                task, inverse_method, extract_mode, '{hemi}'))
        if utils.both_hemi_files_exist(labels_output_fname) and not overwrite:
            continue
        labels_data, labels_names = defaultdict(list), defaultdict(list)
        fol = op.join(MMVT_DIR, subject, 'meg')
        labels_fnames = glob.glob(op.join(
            fol, 'clusters', '{}_{}_vertices_power_spectrum_stat'.format(inverse_method, extract_mode), '*.label'))
        stc = mne.read_source_estimate(
            op.join(fol, '{}_{}_vertices_power_spectrum_stat'.format(inverse_method, extract_mode)))
        for label_fname in labels_fnames:
            label = mne.read_label(label_fname, subject)
            if not filter_re.search(label.name):
                continue
            vertno = stc.lh_vertno if label.hemi == 'lh' else stc.rh_vertno
            this_vertno = np.intersect1d(vertno, label.vertices)
            vertidx = np.searchsorted(vertno, this_vertno)
            if len(vertidx) == 0:
                print('No stc vertices in {}!'.format(label.name))
                continue
            data = stc.lh_data if label.hemi == 'lh' else stc.rh_data
            label_data = data[vertidx].mean(axis=0)
            labels_data[label.hemi].append(label_data)
            labels_names[label.hemi].append(label_data)
        for hemi in utils.HEMIS:
            if hemi in labels_data:
                np.savez(labels_output_fname.format(hemi=hemi), data=np.array(labels_data[hemi]),
                         names=labels_names[hemi], conditions=conditions)
                all_output_files_exist = all_output_files_exist and op.isfile(labels_output_fname.format(hemi=hemi))
    return all_output_files_exist


def find_clusters_over_time(
        subject, stc_name, threshold, times=None, min_cluster_size=10, atlas='', clusters_label='',
        find_clusters_overlapped_labeles=False, mri_subject='', modality='meg', n_jobs=4):
    if mri_subject == '':
        mri_subject = subject
    clusters_root_fol = utils.make_dir(op.join(MMVT_DIR, subject, modality_fol(modality), 'clusters'))
    output_fname = op.join(clusters_root_fol, '{}_clusters_times.pkl'.format(stc_name))
    connectivity = anat.load_connectivity(subject)

    stc_fname = op.join(MMVT_DIR, subject, modality_fol(modality), '{}-lh.stc'.format(stc_name))
    if not op.isfile(stc_fname):
        raise Exception("Can't find the stc file! ({})".format(stc_name))
    stc = mne.read_source_estimate(stc_fname)
    verts = utils.get_pial_vertices(subject, MMVT_DIR)
    if times is None:
        times = range(stc.shape[1])

    verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors_{}.pkl')
    verts_neighbors_dict = {hemi: utils.load(verts_neighbors_fname.format(hemi)) for hemi in utils.HEMIS}

    all_contours = {}
    indices = np.array_split(np.arange(len(times)), n_jobs)
    chunks = [([times[ind] for ind in chunk_indices], subject, mri_subject, atlas, stc_name, threshold,
               min_cluster_size, stc, verts, connectivity, modality, verts_neighbors_dict, clusters_label, stc_name)
              for chunk_indices in indices]
    results = utils.run_parallel(_find_clusters_over_time_parallel, chunks, n_jobs)
    for chunk_contours in results:
        for t, contours in chunk_contours.items():
            all_contours[t] = contours
    print('Results are saved in {}'.format(output_fname))
    utils.save(all_contours, output_fname)
    return op.isfile(output_fname), all_contours


def _find_clusters_over_time_parallel(p):
    (times, subject, mri_subject, atlas, stc_name, threshold, min_cluster_size, stc, verts, connectivity, modality,
     verts_neighbors_dict, clusters_label, stc_name) = p
    all_contours = defaultdict(dict)
    for t in times:
        print('find_functional_rois_in_stc for time {}'.format(t))
        flag, contours = find_functional_rois_in_stc(
            subject, mri_subject, atlas, stc_name, threshold, threshold_is_precentile=False,
            min_cluster_size=min_cluster_size, time_index=t, extract_time_series_for_clusters=False,
            stc=stc, verts=verts, connectivity=connectivity, verts_dict=verts,
            find_clusters_overlapped_labeles=False, modality=modality,
            verts_neighbors_dict=verts_neighbors_dict, save_results=False, clusters_label=clusters_label,
            clusters_output_name='clusters_t{}.pkl'.format(stc_name, t), n_jobs=1)
        for hemi in utils.HEMIS:
            if hemi in contours:
                all_contours[t][hemi] = np.where(contours[hemi]['contours']) if contours[hemi] is not None else None
    return all_contours


def parse_dip_file(dip_fname, read_clusters=False, run_num=1):
    '''
    Parse dip file and return dict of events
    :param dip_fname:
    :return: dict of events
    '''
    # Can take code from mne.dipole.read_dipole(dip_fname)
    # read_dipole does not take the dipoles' names

    results, clusters = defaultdict(list), defaultdict(list)
    if utils.file_type(dip_fname) == 'bdip':
        dipoles = mne.read_dipole(dip_fname)
        dipole_name = utils.namebase(dip_fname) if dipoles.name is None else dipoles.name
        for k in range(len(dipoles)):
            begin_t = end_t = dipoles.times[k]
            x, y, z = dipoles.pos[k] * 1e3
            q = dipoles.amplitude[k]
            qx, qy, qz = dipoles.ori[k]
            gf = dipoles.gof[k]
            results[dipole_name].append([begin_t, end_t, x, y, z, q, qx, qy, qz, gf])
        return results, clusters if read_clusters else results

    def parse(clusters, results, dipole_num):
        line_parts = line.split("\"")
        dipole_name = line_parts[1] if len(line_parts) >= 2 else 'run_{}_{}'.format(
            run_num, same_name_lines[0].split()[0])
        cluster_name = line_parts[3] if len(line_parts) >= 4 else 'dipoles1'
        # tmp_line = (line[line_start:])
        # name = tmp_line.split("\"", maxsplit=1)[0]
        if dipole_name == '':
            dipole_name = "dipole_" + str(dipole_num)
            dipole_num += 1
        clusters[cluster_name].append(dipole_name)
        for item in same_name_lines:
            results[dipole_name].append([float(x) for x in item.split()])

    dipole_num = 1
    same_name_lines = []
    # line_start = len("## Name \"")
    with open(dip_fname, 'r') as target:
        for line in target.readlines():
            if line.startswith('#'):
                if same_name_lines:
                    parse(clusters, results, dipole_num)
                    same_name_lines = []
            else:
                same_name_lines.append(line)

    if same_name_lines:
        parse(clusters, results, dipole_num)
    return results, clusters if read_clusters else results


def convert_dipoles_to_mri_space(
        subject, dip_fname, modality='meg', dipole_keyword='', read_clusters=False, return_dipoles=False,
        run_num=1, overwrite=False, subject_dir='', trans_file='', save_dipoles=True):
    '''
    :param dipole:
    :return:
    '''
    # if False: # op.isfile(output_fname) and not overwrite:
    #     if return_dipoles:
    #         return utils.load(output_fname)
    #     else:
    #         return True
    if subject_dir == '':
        subject_dir = op.join(SUBJECTS_MRI_DIR, subject_dir)
    if not op.isfile(dip_fname):
        dip_fname = op.join(MMVT_DIR, subject, modality_fol(modality), dip_fname)
    if not op.isfile(dip_fname):
        dip_fname = utils.select_one_file(glob.glob(op.join(MEG_DIR, subject, '*.dip')))
    if not op.isfile(dip_fname):
        dip_fname = utils.select_one_file(glob.glob(op.join(MMVT_DIR, subject, modality_fol(modality), '*.dip')))
    if not op.isfile(dip_fname):
        print('Can\'t find a dipoles file!')
        return False
    output_fname = op.join(MMVT_DIR, subject, modality_fol(modality), '{}_dipoles.pkl'.format(
        utils.namebase(dip_fname)))
    dipoles, clusters = parse_dip_file(dip_fname, read_clusters=True, run_num=run_num)
    dipoles_cluster_dic = calc_dipoles_cluster_dic(clusters)
    # If the trans file doesn't exist, you should calculate it using mne-python / MNE-analyzer
    if not op.isfile(trans_file):
        trans_file = find_trans_file(subject=subject, subjects_dir=subject_dir)
    head_mri_trans = mne.transforms.read_trans(trans_file)
    head_mri_trans = mne.transforms._ensure_trans(head_mri_trans, 'head', 'mri')

    mri_dipoles = defaultdict(list)
    for dipole_name, dipoles in dipoles.items():
        if dipole_keyword not in dipole_name.lower():
            continue
        for dipole in dipoles:
            if len(dipole) == 10:
                # begin end(ms)  X (mm)  Y (mm)  Z (mm)  Q(nAm) Qx(nAm) Qy(nAm) Qz(nAm)  g(%)
                begin_t, end_t, x, y, z, q, qx, qy, qz, gf = dipole
            elif len(dipole) == 24:
                # begin end(ms)  X (mm)  Y (mm)  Z (mm) dist(mm) r(mm) th(deg)  phi(deg) Q(nAm) Qx(nAm) Qy(nAm) Qz(nAm) g(%)
                # khi(^2) free prob vol(mm^3) depth(mm) long(mm)   trans(mm)  Qlong(nAm)  Qtrans(nAm) noise(fT/cm)
                begin_t, end_t, x, y, z, dist, r, th, phi, q, qx, qy, qz, gf, khi, free, prob, vol, depth, long, \
                trans, Qlong, Qtrans, noise = dipole
            else:
                raise Exception('This format of dipole is not supported yet!')
            mri_pos = mne.transforms.apply_trans(head_mri_trans, [np.array([x, y, z]) * 1e-3])[0]
            dir_xyz = mne.transforms.apply_trans(head_mri_trans, [np.array([qx, qy, qz]) / q])[0]
            # print('{}: loc:{} dir:{}'.format(dipole_name, mri_pos, dir_xyz))
            if read_clusters:
                mri_dipoles[dipole_name].append(
                    [begin_t, end_t, *mri_pos, q, *dir_xyz, gf, dipoles_cluster_dic[dipole_name]])
            else:
                mri_dipoles[dipole_name].append(
                    [begin_t, end_t, *mri_pos, q, *dir_xyz, gf])
    if save_dipoles:
        print("Saving dipoles in {}".format(output_fname))
        utils.save(mri_dipoles, output_fname)
    if return_dipoles:
        if read_clusters:
            return mri_dipoles, clusters, dipoles_cluster_dic
        else:
            return mri_dipoles
    else:
        return op.isfile(output_fname)


def merge_dipoles(subject, modality):
    output_fname = op.join(MMVT_DIR, subject, modality_fol(modality), 'dipoles-all.pkl')
    dipoles_files = glob.glob(op.join(MMVT_DIR, subject, modality_fol(modality), '*_dipoles.pkl'))
    print('Merging the following dipoles files:')
    all_dipoles = {}
    for dipoles_fname in dipoles_files:
        mri_dipoles = utils.load(dipoles_fname)
        print('{} dipoles from {}'.format(len(mri_dipoles), utils.namebase_with_ext(dipoles_fname)))
        same_keys = all_dipoles.keys() & mri_dipoles.keys()
        if len(same_keys) > 0:
            print('Duplicate keys!')
        all_dipoles.update(mri_dipoles)
    print('Merged dipoles are saved to {}'.format(output_fname))
    utils.save(all_dipoles, output_fname)
    return op.isfile(output_fname)


def calc_dipoles_cluster_dic(clusters):
    dipoles_cluster_dic = {}
    for cluster_name, dipoles_names in clusters.items():
        for dipole_name in dipoles_names:
            dipoles_cluster_dic[dipole_name] = cluster_name
    return dipoles_cluster_dic


def calc_dipoles_rois(
        subject, dip_fname, dipoles_dict=None, atlas='laus125', calc_per_group=True, overwrite=False, n_jobs=4):
    links_dir = utils.get_links_dir()
    subjects_dir = utils.get_link_dir(links_dir, 'subjects')
    mmvt_dir = utils.get_link_dir(links_dir, 'mmvt')
    dip_name = utils.namebase(dip_fname)
    diploes_rois_output_fname = op.join(mmvt_dir, subject, 'meg', '{}_dipoles_rois.pkl'.format(dip_name))
    if op.isfile(diploes_rois_output_fname) and not overwrite:
        diploes_rois = utils.load(diploes_rois_output_fname)
        return diploes_rois

    if dipoles_dict is None:
        diploes_input_fname = op.join(mmvt_dir, subject, 'meg', '{}_dipoles.pkl'.format(dip_name))
        if not op.isfile(diploes_input_fname):
            print('No dipoles file!')
            return None
        dipoles_dict = utils.load(diploes_input_fname)

    labels = lu.read_labels(subject, subjects_dir, atlas, n_jobs=n_jobs)
    labels = list([{'name': label.name, 'hemi': label.hemi, 'vertices': label.vertices}
                   for label in labels])
    if len(labels) == 0:
        print('Can\'t find the labels for atlas {}!'.format(atlas))
        return None

    # find the find_rois package
    mmvt_code_fol = utils.get_mmvt_code_root()
    ela_code_fol = op.join(utils.get_parent_fol(mmvt_code_fol), 'electrodes_rois')
    if not op.isdir(ela_code_fol) or not op.isfile(op.join(ela_code_fol, 'find_rois', 'main.py')):
        print("Can't find ELA folder!")
        print('git pull https://github.com/pelednoam/electrodes_rois.git')
        return None

    # load the find_rois package
    try:
        import sys
        if ela_code_fol not in sys.path:
            sys.path.append(ela_code_fol)
        from find_rois import main as ela
    except:
        print('Can\'t load find_rois package!')
        utils.print_last_error_line()
        return None

    diploles_names, dipoles_pos = [], []
    for cluster_name, dipoles in dipoles_dict.items():
        if calc_per_group:
            _, _, x, y, z = dipoles[0][:5]
            dipoles_pos.append([k * 1e3 for k in [x, y, z]])
            diploles_names.append(cluster_name.replace(' ', ''))
        else:
            for dipole in dipoles:
                begin_t, _, x, y, z = dipole[:5]
                dipole_name = '{}_{}'.format(cluster_name, begin_t) if len(dipoles) > 1 else cluster_name
                diploles_names.append(dipole_name.replace(' ', ''))
                dipoles_pos.append([k * 1e3 for k in [x, y, z]])
    dipoles_rois = ela.identify_roi_from_atlas(
        atlas, labels, diploles_names, dipoles_pos, approx=3, elc_length=0, hit_only_cortex=True,
        subjects_dir=subjects_dir, subject=subject, n_jobs=n_jobs)
    # Convert the list to a dict
    dipoles_rois_dict = {dipoles_rois['name']: dipoles_rois for dipoles_rois in dipoles_rois}
    utils.save(dipoles_rois_dict, diploes_rois_output_fname)
    return dipoles_rois_dict


def init_main(subject, mri_subject, remote_subject_dir, args):
    if args.events_fname != '':
        args.events_fname = op.join(MEG_DIR, args.task, subject, args.events_fname)
        if '{subject}' in args.events_fname:
            args.events_fname = args.events_fname.format(subject=subject)
    args.remote_subject_dir = remote_subject_dir
    args.remote_subject_meg_dir = utils.build_remote_subject_dir(args.remote_subject_meg_dir, subject)
    prepare_subject_folder(mri_subject, remote_subject_dir, SUBJECTS_MRI_DIR,
                           args.mri_necessary_files, args)
    fname_format, fname_format_cond, conditions = get_fname_format_args(args)
    return fname_format, fname_format_cond, conditions


def init(subject, args, mri_subject='', remote_subject_dir=''):
    if mri_subject == '':
        mri_subject = subject
    fname_format, fname_format_cond, conditions = init_main(subject, mri_subject, remote_subject_dir, args)
    init_globals_args(subject, mri_subject, fname_format, fname_format_cond, args=args)
    fname_format = fname_format.replace('{subject}', SUBJECT)
    fname_format = fname_format.replace('{ana_type}', 'raw')
    if args.raw_fname == '':
        if '{file_type}' in fname_format:
            fname_format = fname_format.replace('{file_type}', '*')
            raw_files = glob.glob(op.join(SUBJECT_MEG_FOLDER, fname_format))
        else:
            raw_files = glob.glob(op.join(SUBJECT_MEG_FOLDER, '{}.fif'.format(fname_format)))
        if len(raw_files) == 1:
            args.raw_fname = raw_files[0]
    return fname_format, fname_format_cond, conditions


def main(tup, remote_subject_dir, org_args, flags=None):
    args = utils.Bag({k: copy.deepcopy(org_args[k]) for k in org_args.keys()})
    (subject, mri_subject), inverse_method = tup
    args.raw_fname = args.raw_fname.format(subject=subject)
    args.epo_fname = args.epo_fname.format(subject=subject)
    evoked, epochs, raw = None, None, None
    stcs_conds, stcs_conds_smooth = None, None
    if flags is None:
        flags = {}
    fname_format, fname_format_cond, conditions = init(subject, args, mri_subject, remote_subject_dir)
    if len(conditions) == 1:
        args.cond_name = list(conditions)[0]
    if args.raw_fname != '':
        args.raw_template = args.raw_fname
    # fname_format, fname_format_cond, conditions = init_main(subject, mri_subject, remote_subject_dir, args)
    # init_globals_args(
    #     subject, mri_subject, fname_format, fname_format_cond, MEG_DIR, SUBJECTS_MRI_DIR, MMVT_DIR, args)
    stat = STAT_AVG if len(conditions) == 1 else STAT_DIFF
    args.modality = get_modality(args.fwd_usingMEG, args.fwd_usingEEG)

    if args.bad_channels_fname != '':
        if '{subject}' in args.bad_channels_fname:
            args.bad_channels_fname = args.bad_channels_fname.format(subject=subject)
        args.bad_channels = []
        for line in utils.read_list_from_file(args.bad_channels_fname):
            ch_type = 'MEG' if line.startswith('MEG') else 'EEG'
            args.bad_channels.extend(['{}{}'.format(ch_type, x) for x in line[4:].strip().split(',')])

    if utils.should_run(args, 'read_sensors_layout'):
        flags['read_sensors_layout'] = read_sensors_layout(
            mri_subject, args, overwrite_sensors=args.overwrite_sensors, raw_template=args.raw_template,
            trans_file=args.trans_fname, info_fname=args.info_fname, read_info_file=args.read_info_file,
            raw_fname=args.raw_fname)

    # flags: calc_evoked
    flags, evoked, epochs = calc_evokes_wrapper(subject, conditions, args, flags, mri_subject=mri_subject)
    # flags: make_forward_solution, calc_inverse_operator
    flags = calc_fwd_inv_wrapper(subject, args, conditions, flags, mri_subject)
    # flags: calc_stc_per_condition
    flags, stcs_conds, stcs_num = calc_stc_per_condition_wrapper(
        subject, conditions, inverse_method, args, flags)
    # flags: calc_labels_avg_per_condition
    flags = calc_labels_avg_per_condition_wrapper(
        subject, conditions, args.atlas, inverse_method, stcs_conds, args, flags, stcs_num, raw, epochs,
        modality=args.modality, mri_subject=mri_subject)

    if 'calc_stc_zvals' in args.function:
        flags['calc_stc_zvals'] = calc_stc_zvals(
            subject, args.stc_name, args.baseline_stc_name, args.modality, args.use_abs,
            args.from_index, args.to_index, args.stc_zvals_name, False, args.overwrite_stc)

    if 'calc_power_spectrum' in args.function:
        flags['calc_power_spectrum'] = calc_power_spectrum(subject, conditions, args)

    if 'save_vertex_activity_map' in args.function:
        stc_fnames = [STC_HEMI_SMOOTH.format(cond='{cond}', method=inverse_method, hemi=hemi)
                      for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['save_vertex_activity_map'] = save_vertex_activity_map(conditions, stat, stcs_conds_smooth,
                                                                     inverse_method)

    if 'calc_labels_avg_for_rest' in args.function:
        flags['calc_labels_avg_for_rest'] = calc_labels_avg_for_rest(
            subject, args.atlas, inverse_method, None, args.pick_ori, args.extract_mode, args.snr, args.raw_fname,
            args.inv_fname, args.labels_data_template, args.overwrite_stc, args.overwrite_labels_data,
            args.fwd_usingMEG, args.fwd_usingEEG, cond_name='all', positive=False, moving_average_win_size=0,
            save_data_files=True, n_jobs=args.n_jobs)

    # functions that aren't in the main pipeline
    if 'smooth_stc' in args.function:
        stc_fnames = [STC_HEMI.format(cond='{cond}', method=inverse_method, hemi=hemi) for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['smooth_stc'], stcs_conds_smooth = smooth_stc(
            conditions, stcs_conds, inverse_method, args.stc_t, args.morph_to_subject, args.n_jobs)

    if 'save_activity_map' in args.function:
        stc_fnames = [STC_HEMI_SMOOTH.format(cond='{cond}', method=inverse_method, hemi=hemi)
                      for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['save_activity_map'] = save_activity_map(
            conditions, stat, stcs_conds_smooth, inverse_method, args.save_smoothed_activity, args.morph_to_subject,
            args.stc_t, args.norm_by_percentile, args.norm_percs)

    if 'find_functional_rois_in_stc' in args.function:
        flags['find_functional_rois_in_stc'], _ = find_functional_rois_in_stc(
            subject, mri_subject, args.atlas, args.stc_name, args.threshold, args.threshold_is_precentile,
            args.peak_stc_time_index, args.label_name_template, args.peak_mode, args.extract_time_series_for_clusters,
            args.extract_mode, args.min_cluster_max, args.min_cluster_size, args.clusters_label,
            inv_fname=args.inv_fname, fwd_usingMEG=args.fwd_usingMEG, fwd_usingEEG=args.fwd_usingEEG,
            recreate_src_spacing=args.recreate_src_spacing, save_func_labels=args.save_func_labels,
            calc_cluster_contours=args.calc_cluster_contours, n_jobs=args.n_jobs)

    if 'print_files_names' in args.function:
        print_files_names()

    if 'calc_single_trial_labels_per_condition' in args.function:
        calc_single_trial_labels_per_condition(args.atlas, conditions, stcs_conds, extract_mode=args.extract_mode)

    sub_corticals_codes_file = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
    if 'calc_sub_cortical_activity' in args.function:
        # todo: call get_meg_files
        calc_sub_cortical_activity(conditions, sub_corticals_codes_file, inverse_method, args.pick_ori, evoked, epochs)

    if 'save_subcortical_activity_to_blender' in args.function:
        save_subcortical_activity_to_blender(sub_corticals_codes_file, conditions, stat, inverse_method,
                                             args.norm_by_percentile, args.norm_percs)
    if 'plot_sub_cortical_activity' in args.function:
        plot_sub_cortical_activity(conditions, sub_corticals_codes_file, inverse_method=inverse_method)

    # if 'calc_activity_significance' in args.function:
    #     calc_activity_significance(conditions, inverse_method, stcs_conds)

    if 'save_activity_map_minmax' in args.function:
        flags['save_activity_map_minmax'] = save_activity_map_minmax(
            None, conditions, stat, stcs_conds_smooth, inverse_method, args.morph_to_subject,
            args.norm_by_percentile, args.norm_percs, False)

    if 'remove_artifacts' in args.function:
        flags['remove_artifacts'] = remove_artifacts(
            n_max_ecg=args.ica_n_max_ecg, n_max_eog=args.ica_n_max_eog, n_components=args.ica_n_components,
            method=args.ica_method, remove_from_raw=args.remove_artifacts_from_raw, overwrite_ica=args.overwrite_ica,
            overwrite_raw=args.ica_overwrite_raw, raw_fname=args.raw_fname)

    if 'morph_stc' in args.function:
        flags['morph_stc'] = morph_stc(
            MRI_SUBJECT, conditions, args.morph_to_subject, args.inverse_method[0], args.grade,
            args.smoothing_iterations, args.modality, args.overwrite_stc, args.n_jobs)

    if 'morph_stc_file' in args.function:
        flags['morph_stc_file'] = morph_stc_file(
           subject, args.stc_fname, args.morph_to_subject, args.grade, args.modality, args.overwrite_stc)

    if 'calc_stc_diff' in args.function:
        flags['calc_stc_diff'] = calc_stc_diff_both_hemis(
            subject, conditions, args.modality, STC_HEMI, args.inverse_method[0], args.overwrite_stc)

    if 'calc_labels_connectivity' in args.function:
        # Warning!!! Need to add more variables to the function!!!
        flags['calc_labels_connectivity'] = calc_labels_connectivity(
            SUBJECT, args.atlas, conditions, MRI_SUBJECT, SUBJECTS_MRI_DIR, MMVT_DIR, inverse_method,
            args.epo_fname, args.inv_fname, args.raw_fname, args.snr, args.pick_ori, args.apply_SSP_projection_vectors,
            args.add_eeg_ref, args.fwd_usingMEG, args.fwd_usingEEG, args.extract_mode, args.surf_name,
            args.con_method, args.con_mode, args.cwt_n_cycles, args.max_epochs_num,
            overwrite_connectivity=args.overwrite_connectivity,n_jobs=args.n_jobs)
        '''
        def calc_labels_connectivity(
            subject, atlas, events, mri_subject='', subjects_dir='', mmvt_dir='', inverse_method='dSPM',
            epo_fname='', inv_fname='', raw_fname='', snr=3.0, pick_ori=None, apply_SSP_projection_vectors=True,
            add_eeg_ref=True, fwd_usingMEG=True, fwd_usingEEG=True, extract_modes=['mean_flip'], surf_name='pial',
            con_method='coh', con_mode='cwt_morlet', cwt_n_cycles=7, max_epochs_num=0, min_order=1, max_order=100,
            estimate_order=False, windows_length=0, windows_shift=0, calc_only_granger_causality_likelihood=False,
            overwrite_connectivity=False, raw=None, epochs=None, src=None, inverse_operator=None, bands=None, labels=None,
            cwt_frequencies=None, con_indentifer='', symetric_con=None, downsample=1, crops_times=None, output_fname='',
            n_jobs=6):
        '''

    if 'calc_labels_connectivity_from_stc' in args.function:
        flags['calc_labels_connectivity_from_stc'] = calc_labels_connectivity_from_stc(
            SUBJECT, args.atlas, conditions, args.stc_name, args.file_name_with_info, MRI_SUBJECT, SUBJECTS_MRI_DIR,
            MMVT_DIR, args.inv_fname, args.fwd_usingMEG, args.fwd_usingEEG, args.extract_mode, args.surf_name,
            args.con_method, args.con_mode, args.cwt_n_cycles, args.overwrite_connectivity,
            n_jobs=args.n_jobs)

    if 'calc_baseline_sensors_bands_psd' in args.function:
        flags['calc_baseline_sensors_bands_psd'] = calc_baseline_sensors_bands_psd(
            mri_subject, args.epo_fname, args.raw_template, args.modality, args.bad_channels,
            args.baseline_len, args.fmin, args.fmax, args.bandwidth, args.cond_name,
            args.precentiles, overwrite=args.overwrite_baseline_sensors_bands_psd, n_jobs=args.n_jobs)

    if 'calc_epochs_psd' in args.function:
        flags['calc_epochs_psd'] = calc_epochs_psd(
            subject, conditions, max_epochs_num=args.max_epochs_num, raw_template=args.raw_template,
            n_jobs=args.n_jobs)

    if 'calc_epochs_bands_psd' in args.function:
        flags['calc_epochs_bands_psd'] = calc_epochs_bands_psd(
            subject, conditions, args.precentiles, overwrite=args.overwrite_sensors_psd)

    if 'calc_source_baseline_psd' in args.function:
        flags['calc_source_baseline_psd'] = calc_source_baseline_psd(
            subject, args.task, mri_subject, args.raw_fname, args.epo_fname, args.inv_fname, inverse_method,
            args.snr, args.baseline_len, args.fmin, args.fmax, fwd_usingMEG=args.fwd_usingMEG,
            fwd_usingEEG=args.fwd_usingEEG, overwrite=args.overwrite_source_baseline_psd, n_jobs=args.n_jobs)

    if 'calc_source_power_spectrum' in args.function:
        flags['calc_source_power_spectrum'] = calc_source_power_spectrum(
            subject, conditions, args.atlas, inverse_method, args.extract_mode, args.fmin, args.fmax, args.bandwidth,
            args.bands, args.max_epochs_num, MRI_SUBJECT, args.epo_fname, args.inv_fname, args.snr, args.pick_ori,
            args.apply_SSP_projection_vectors, args.add_eeg_ref, args.fwd_usingMEG, args.fwd_usingEEG, args.surf_name,
            args.precentiles, (args.baseline_min, args.baseline_max), overwrite=args.overwrite_labels_power_spectrum,
            save_tmp_files=args.save_tmp_files, label_stat=args.label_stat, n_jobs=args.n_jobs)

    if 'average_power_spectrum_per_label' in args.function:
        flags['average_power_spectrum_per_label'] = average_power_spectrum_per_label(
            subject, args.task, inverse_method, extract_modes=args.extract_mode,
            conditions=args.power_spectrum_conditions, labels_filter=args.labels_filter)

    if 'calc_vertices_data_power_bands' in args.function:
        flags['calc_vertices_data_power_bands'] = calc_vertices_data_power_bands(
            subject, conditions, MRI_SUBJECT, inverse_method, args.extract_mode)

    if 'calc_labels_power_bands' in args.function:
        flags['calc_labels_power_bands'] = calc_labels_power_bands(
            subject, args.atlas, conditions, inverse_method, args.extract_mode, args.precentiles,
            overwrite=args.overwrite_labels_power_spectrum, n_jobs=args.n_jobs)

    if 'calc_labels_induced_power' in args.function:
        flags['calc_labels_induced_power'] = calc_labels_induced_power(
            subject, args.atlas, conditions, inverse_method, args.extract_mode, args.bands,
            args.max_epochs_num, args.average_over_label_indices, args.cwt_n_cycles, MRI_SUBJECT, args.epo_fname,
            args.inv_fname, args.snr, args.pick_ori, args.apply_SSP_projection_vectors, args.add_eeg_ref,
            args.fwd_usingMEG, args.fwd_usingEEG, overwrite=args.overwrite_labels_induced_power, n_jobs=args.n_jobs)

    if 'calc_source_morph_mat' in args.function:
        flags['calc_source_morph_mat'], _ = calc_source_morph_mat(subject, subject)

    if 'load_fieldtrip_volumetric_data' in args.function:
        flags['load_fieldtrip_volumetric_data'] = load_fieldtrip_volumetric_data(
            subject, args.fieldtrip_data_name, args.fieldtrip_data_field_name, args.overwrite_nii_file,
            args.overwrite_surface, args.overwrite_stc)

    if 'fit_ica_on_subjects' in args.function:
        fit_ica_on_subjects(args.subject, meg_root_fol=args.meg_root_fol)

    if 'create_helmet_mesh' in args.function:
        flags['create_helmet_mesh'] = create_helmet_mesh(subject, excludes=[], overwrite_faces_verts=True)

    if 'find_trans_file' in args.function:
        flags['find_trans_file'] = op.isfile(find_trans_file(
            '', args.remote_subject_dir, MRI_SUBJECT, SUBJECTS_MRI_DIR))

    if 'stc_time_average' in args.function:
        flags['stc_time_average'] = stc_time_average(subject, args.stc_time_average_dt, args.stc_template)

    if 'sensors_time_average' in args.function:
        flags['sensors_time_average'] = sensors_time_average(subject, args.stc_time_average_dt, args.overwrite)

    if 'get_digitization_points' in args.function:
        flags['get_digitization_points'] = get_digitization_points(subject, args.raw_fname)

    if 'stc_to_contours' in args.function:
        flags['stc_to_contours'], _ = stc_to_contours(
            subject, args.stc_name, args.peak_stc_time_index, args.thresholds_min, args.thresholds_max,
            args.thresholds_dx, mri_subject, 'meg', args.n_jobs)

    if 'find_clusters_over_time' in args.function:
        flags['find_clusters_over_time'], _ = find_clusters_over_time(
            subject, args.stc_name, threshold=2, times=None, min_cluster_size=10,
            mri_subject=mri_subject, n_jobs=args.n_jobs)

    if 'plot_evoked' in args.function:
        flags['plot_evoked'], _ = plot_evoked(
            subject, args.evo_fname, args.evoked_key, args.pick_meg, args.pick_eeg, args.pick_eog, args.ssp_proj,
            args.spatial_colors, args.window_title, args.hline, args.channels_to_exclude)

    if 'plot_topomap' in args.function:
        flags['plot_topomap'], _ = plot_topomap(
            subject, args.evo_fname, args.evoked_key, args.times, args.find_peaks, args.ch_type, args.ssp_proj,
            args.average,
            args.n_peaks, args.window_title)

    if 'plot_max_stc' in args.function:
        flags['plot_max_stc'] = plot_max_stc(subject, args.stc_name, args.modality, args.use_abs)

    if 'plot_max_labels_data' in args.function:
        flags['plot_max_labels_data'] = plot_max_labels_data(subject, args.atlas, inverse_method=args.inverse_method)

    if 'convert_dipoles_to_mri_space' in args.function:
        flags['convert_dipoles_to_mri_space'] = convert_dipoles_to_mri_space(
            subject, args.dipoles_fname, args.modality, overwrite=args.overwrite_dipoles)

    if 'merge_dipoles' in args.function:
        flags['merge_dipoles'] = merge_dipoles(subject, args.modality)

    return flags


def read_cmd_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='MMVT anatomy preprocessing')
    parser.add_argument('-m', '--mri_subject', help='mri subject name', required=False, default=None,
                        type=au.str_arr_type)
    parser.add_argument('-t', '--task', help='task name', required=False, default='')
    parser.add_argument('-c', '--conditions', help='conditions', required=False, default='', type=au.str_arr_type)
    parser.add_argument('-i', '--inverse_method', help='inverse_method', required=False, default='dSPM',
                        type=au.str_arr_type)
    parser.add_argument('--modality', help='', required=False, default='meg')
    parser.add_argument('--sub_dirs_for_tasks', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--get_task_defaults', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fname_format', help='', required=False, default='{subject}-{ana_type}.{file_type}')
    parser.add_argument('--fname_format_cond', help='', required=False,
                        default='{subject}_{cond}-{ana_type}.{file_type}')
    parser.add_argument('--data_per_task', help='task-subject-data', required=False, default=0, type=au.is_true)
    parser.add_argument('--raw_fname_format', help='', required=False, default='')
    parser.add_argument('--raw_fname', help='', required=False, default='')
    parser.add_argument('--fwd_fname_format', help='', required=False, default='')
    parser.add_argument('--inv_fname_format', help='', required=False, default='')
    parser.add_argument('--inv_fname', help='', required=False, default='')
    parser.add_argument('--fwd_fname', help='', required=False, default='')
    parser.add_argument('--epo_fname', help='', required=False, default='')
    parser.add_argument('--evo_fname', help='', required=False, default='')
    parser.add_argument('--cor_fname', help='', required=False, default='')
    parser.add_argument('--info_fname', help='', required=False, default='')
    parser.add_argument('--file_name_with_info', help='', required=False, default='')
    parser.add_argument('--read_info_file', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--noise_cov_fname', help='', required=False, default='')
    parser.add_argument('--empty_fname', help='', required=False, default='')
    parser.add_argument('--calc_evoked_for_all_epoches', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--average_per_event', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--set_eeg_reference', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--max_epochs_num', help='', required=False, default=0, type=int)
    parser.add_argument('--overwrite', help='general overwrite', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_epochs', help='overwrite_epochs', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_evoked', help='overwrite_evoked', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_sensors', help='overwrite_sensors', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_fwd', help='overwrite_fwd', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_inv', help='overwrite_inv', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_stc', help='overwrite_stc', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_labels_data', help='overwrite_labels_data', required=False, default=0,
                        type=au.is_true)
    parser.add_argument('--overwrite_labels_power_spectrum', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--labels_filter', help='', required=False, default='')
    parser.add_argument('--power_spectrum_conditions', required=False, default='power_ttest', type=au.str_arr_type)
    parser.add_argument('--overwrite_sensors_psd', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_labels_induced_power', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_baseline_sensors_bands_psd', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_source_baseline_psd', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--read_events_from_file', help='read_events_from_file', required=False, default=0,
                        type=au.is_true)
    parser.add_argument('--read_events_as_annotation', help='read_events_as_annotation', required=False, default=0,
                        type=au.is_true)
    parser.add_argument('--events_fname', help='events_fname', required=False, default='')
    parser.add_argument('--use_demi_events', help='use_demi_events', required=False, default=0, type=au.is_true)
    parser.add_argument('--windows_length', help='', required=False, default=1000, type=int)
    parser.add_argument('--windows_shift', help='', required=False, default=500, type=int)
    parser.add_argument('--windows_num', help='', required=False, default=0, type=int)
    parser.add_argument('--bad_channels', help='bad_channels', required=False, default=[], type=au.str_arr_type)
    parser.add_argument('--bad_channels_fname', help='bad_channels_files', required=False, default='')
    parser.add_argument('--calc_epochs_from_raw', help='calc_epochs_from_raw', required=False, default=0,
                        type=au.is_true)
    parser.add_argument('--l_freq', help='low freq filter', required=False, default=None, type=float)
    parser.add_argument('--h_freq', help='high freq filter', required=False, default=None, type=float)
    parser.add_argument('--pick_meg', help='pick meg events', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_eeg', help='pick eeg events', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_eog', help='pick eog events', required=False, default=0, type=au.is_true)
    parser.add_argument('--remove_power_line_noise', help='remove power line noise', required=False, default=1,
                        type=au.is_true)
    parser.add_argument('--power_line_freq', help='power line freq', required=False, default=60, type=int)
    parser.add_argument('--power_line_notch_widths', help='notch_widths', required=False, default=None,
                        type=au.float_or_none)
    parser.add_argument('--stim_channels', help='stim_channels', required=False, default='STI001', type=au.str_arr_type)
    parser.add_argument('--reject', help='reject trials', required=False, default=1, type=au.is_true)
    parser.add_argument('--reject_grad', help='', required=False, default=4000e-13, type=float)
    parser.add_argument('--reject_mag', help='', required=False, default=4e-12, type=float)
    parser.add_argument('--reject_eog', help='', required=False, default=150e-6, type=float)
    parser.add_argument('--apply_SSP_projection_vectors', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--add_eeg_ref', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_ori', help='', required=False, default='normal')
    parser.add_argument('--t_min', help='', required=False, default=-0.2, type=float)  # MNE python defaults
    parser.add_argument('--t_max', help='', required=False, default=0.5, type=float)  # MNE python defaults
    parser.add_argument('--zero_time', help='', required=False, default=None, type=au.float_or_none)  # MNE python defaults
    parser.add_argument('--noise_t_min', help='', required=False, default=None, type=au.float_or_none)
    parser.add_argument('--noise_t_max', help='', required=False, default=0, type=float)
    parser.add_argument('--snr', help='', required=False, default=3.0, type=float)
    parser.add_argument('--calc_stc_for_all', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--stc_t_min', help='', required=False, default=None, type=float)
    parser.add_argument('--stc_t_max', help='', required=False, default=None, type=float)
    parser.add_argument('--stc_time_average_dt', help='', required=False, default=10, type=int)
    parser.add_argument('--baseline_min', help='', required=False, default=None, type=float)
    parser.add_argument('--baseline_max', help='', required=False, default=0, type=au.float_or_none)
    parser.add_argument('--baseline_len', help='', required=False, default=10000, type=au.float_or_none)
    parser.add_argument('--files_includes_cond', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--inv_no_cond', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fwd_no_cond', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--contrast', help='', required=False, default='')
    parser.add_argument('--cleaning_method', help='', required=False, default='')  # nTSSS
    parser.add_argument('--fwd_usingMEG', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fwd_usingEEG', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fwd_calc_corticals', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fwd_calc_subcorticals', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--fwd_recreate_source_space', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--recreate_bem_solution', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--bem_ico', help='', required=False, default=4, type=int)
    parser.add_argument('--recreate_src_spacing', help='', required=False, default='oct6')
    parser.add_argument('--recreate_src_surface', help='', required=False, default='white')
    parser.add_argument('--surf_name', help='', required=False, default='pial')
    parser.add_argument('--inv_loose', help='', required=False, default=0.2, type=float)
    parser.add_argument('--inv_depth', help='', required=False, default=0.8, type=float)
    parser.add_argument('--trans_fname', help='', required=False, default='')
    parser.add_argument('--use_raw_for_noise_cov', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--use_empty_room_for_noise_cov', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_noise_cov', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--inv_calc_cortical', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--inv_calc_subcorticals', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--evoked_flip_positive', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--evoked_moving_average_win_size', help='', required=False, default=0, type=int)
    parser.add_argument('--normalabel_statlize_evoked', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--average_over_label_indices', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--calc_max_min_diff', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--raw_template', help='', required=False, default='*raw.fif')
    parser.add_argument('--eve_template', help='', required=False, default='*eve.fif')
    parser.add_argument('--stc_template', help='', required=False, default='')
    parser.add_argument('--labels_data_template', help='', required=False, default='')
    parser.add_argument('--save_stc', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--stc_t', help='', required=False, default=-1, type=int)
    parser.add_argument('--calc_stc_diff', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--downsample_r', help='', required=False, default=1, type=int)
    parser.add_argument('--morph_to_subject', help='', required=False, default='colin27')
    parser.add_argument('--stc_fname', help='', required=False, default='')
    parser.add_argument('--single_trial_stc', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--calc_source_band_induced_power', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--apply_on_raw', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--extract_mode', help='', required=False, default='mean_flip', type=au.str_arr_type)
    parser.add_argument('--read_only_from_annot', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--colors_map', help='', required=False, default='OrRd')
    parser.add_argument('--save_smoothed_activity', help='', required=False, default=True, type=au.is_true)
    parser.add_argument('--normalize_data', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--norm_by_percentile', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--norm_percs', help='', required=False, default='1,99', type=au.int_arr_type)
    parser.add_argument('--remote_subject_meg_dir', help='remote_subject_dir', required=False, default='')
    parser.add_argument('--meg_root_fol', required=False, default='')
    parser.add_argument('--bands', required=False, default=None)
    parser.add_argument('--calc_inducde_power_per_label', required=False, default=1, type=au.is_true)
    parser.add_argument('--induced_power_normalize_proj', required=False, default=1, type=au.is_true)
    parser.add_argument('--fmin', required=False, default=1, type=int)
    parser.add_argument('--fmax', required=False, default=120, type=int)
    parser.add_argument('--bandwidth', required=False, default=2., type=float)
    parser.add_argument('--precentiles', required=False, default='1,99', type=au.str_arr_type)
    parser.add_argument('--check_for_channels_inconsistency', required=False, default=1, type=au.is_true)
    parser.add_argument('--save_tmp_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--from_index', required=False, default=None, type=au.int_or_none)
    parser.add_argument('--to_index', required=False, default=None, type=au.int_or_none)
    parser.add_argument('--stc_zvals_name', required=False, default='')
    parser.add_argument('--label_stat', required=False, default='mean')

    # parser.add_argument('--sftp_sso', help='ask for sftp pass only once', required=False, default=0, type=au.is_true)
    parser.add_argument('--eeg_electrodes_excluded_from_mesh', help='', required=False, default='',
                        type=au.str_arr_type)
    # **** ICA *****
    parser.add_argument('--ica_n_components', required=False, default=0.95, type=float)
    parser.add_argument('--ica_n_max_ecg', required=False, default=3, type=int)
    parser.add_argument('--ica_n_max_eog', required=False, default=1, type=int)
    parser.add_argument('--ica_method', required=False, default='fastica')
    parser.add_argument('--overwrite_ica', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--remove_artifacts_from_raw', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--ica_overwrite_raw', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--do_plot_ica', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--calc_spectrum_with_no_windows', help='', required=False, default=0, type=au.is_true)
    # Clusters
    parser.add_argument('--stc_name', required=False, default='')
    parser.add_argument('--baseline_stc_name', required=False, default='')
    parser.add_argument('--use_abs', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--threshold', required=False, default=75, type=float)
    parser.add_argument('--threshold_is_precentile', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--peak_stc_time_index', required=False, default=None, type=au.int_or_none)
    parser.add_argument('--label_name_template', required=False, default='')
    parser.add_argument('--peak_mode', required=False, default='abs')
    parser.add_argument('--extract_time_series_for_clusters', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--min_cluster_max', required=False, default=0, type=float)
    parser.add_argument('--min_cluster_size', required=False, default=0, type=int)
    parser.add_argument('--clusters_label', required=False, default='')
    parser.add_argument('--save_func_labels', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--calc_cluster_contours', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--thresholds_min', required=False, default=None, type=au.float_or_none)
    parser.add_argument('--thresholds_max', required=False, default=None, type=au.float_or_none)
    parser.add_argument('--thresholds_dx', required=False, default=1, type=int)

    # FieldTrip
    parser.add_argument('--fieldtrip_data_name', required=False, default='')
    parser.add_argument('--fieldtrip_data_field_name', required=False, default='')
    parser.add_argument('--overwrite_nii_file', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_surface', help='', required=False, default=0, type=au.is_true)
    # Smoothing / Morphing
    parser.add_argument('--grade', help='', required=False, default=5, type=au.int_or_none)
    parser.add_argument('--smoothing_iterations', help='', required=False, default=None, type=au.int_or_none)
    # Connectivty
    parser.add_argument('--con_method', required=False, default='pli')
    parser.add_argument('--con_mode', required=False, default='cwt_morlet')
    parser.add_argument('--cwt_n_cycles', required=False, default=7, type=int)
    parser.add_argument('--overwrite_connectivity', required=False, default=0, type=au.is_true)
    # AutoReject
    parser.add_argument('--using_auto_reject', required=False, default=0, type=au.is_true)
    parser.add_argument('--ar_compute_thresholds_method', required=False, default='random_search',
                        choices=['random_search', 'bayesian_optimization'])
    parser.add_argument('--bad_ar_threshold', required=False, default=0.5, type=float)
    parser.add_argument('--ar_consensus_percs', required=False, default=None)
    parser.add_argument('--ar_n_interpolates', required=False, default=None)
    # evoked plotting
    parser.add_argument('--evoked_key', required=False, default=None, type=au.str_or_none)
    parser.add_argument('--ssp_proj', required=False, default=0, type=au.is_true)
    parser.add_argument('--spatial_colors', required=False, default=1, type=au.is_true)
    parser.add_argument('--window_title', required=False, default='')
    parser.add_argument('--hline', required=False, default=None, type=au.float_arr_type)
    parser.add_argument('--channels_to_exclude', required=False, default='bads', type=au.str_arr_type)
    # topoplot plotting
    parser.add_argument('--times', required=False, default='peaks')
    parser.add_argument('--find_peaks', required=False, default=False, type=bool)
    parser.add_argument('--n_peaks', required=False, default=5, type=int)
    parser.add_argument('--average', required=False, default=None, type=au.float_or_none)
    parser.add_argument('--ch_type', required=False, default=None, type=au.str_or_none)
    # dipoles
    parser.add_argument('--dipoles_fname', required=False, default='')
    parser.add_argument('--overwrite_dipoles', required=False, default=0, type=au.is_true)

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    pu.set_default_folders(args)
    if args.meg_dir == '':
        args.meg_dir = MEG_DIR
    args.mri_necessary_files = {'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg'],
                                'label': ['{}.{}.annot'.format(hemi, args.atlas) for hemi in utils.HEMIS]}
    if not args.mri_subject:
        args.mri_subject = args.subject
    if args.baseline_min is None and args.baseline_max is None:
        args.baseline = None
    else:
        args.baseline = (args.baseline_min, args.baseline_max)
    if args.task == 'rest':
        if not args.apply_on_raw:
            # todo: check why it was set to True
            # args.single_trial_stc = True
            args.calc_epochs_from_raw = True
        args.use_empty_room_for_noise_cov = True
        args.baseline_min = 0
        args.baseline_max = 0
    try:
        args.precentiles = [float(p) for p in args.precentiles]
    except:
        args.precentiles = [0, 100]
    # todo: Was set as a remark, why?
    if args.pick_ori == 'None':
        args.pick_ori = None
    if args.n_jobs == -1:
        args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.function == ['rest_functions']:
        args.function = 'calc_epochs,make_forward_solution,calc_inverse_operator,calc_stc,calc_labels_avg_per_condition'
    # print(args)
    return args


def get_subjects_itr_func(args):
    subjects_itr = product(zip(args.subject, args.mri_subject), args.inverse_method)
    subject_func = lambda x: x[0][1]
    return subjects_itr, subject_func


def call_main(args):
    subjects_itr, subject_func = get_subjects_itr_func(args)
    return pu.run_on_subjects(args, main, subjects_itr, subject_func)


if __name__ == '__main__':
    args = read_cmd_args()
    call_main(args)
    print('finish!')
