import os.path as op
import numpy as np
try:
    import mne.io
except:
    print('No mne!')
import traceback
from functools import partial

from src.utils import utils
from src.utils import preproc_utils as pu
from src.preproc import meg as meg

SUBJECTS_MRI_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
SUBJECTS_EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
# if SUBJECTS_EEG_DIR == '':
#     print('No EEG folder, using MEG folder')
#     SUBJECTS_EEG_DIR = MEG_DIR
# if SUBJECTS_EEG_DIR == '':
#     raise Exception('No EEG folder (not MEG)!')
SUBJECT_EEG_DIR = ''

calc_evokes = meg.calc_evokes_wrapper
calc_fwd_inv = meg.calc_fwd_inv_wrapper
calc_stc_per_condition = meg.calc_stc_per_condition_wrapper
plot_evoked = meg.plot_evoked
plot_topomap = meg.plot_topomap


def read_sensors_layout_args(mri_subject, args):
    return meg.read_sensors_layout(
        mri_subject, args, pick_meg=False, pick_eeg=True, overwrite_sensors=args.overwrite_sensors,
        raw_template=args.raw_template, trans_file=args.trans_fname, info_fname=args.info_fname,
        read_info_file=args.read_info_file, raw_fname=args.raw_fname)


def read_sensors_layout(mri_subject, args=None, overwrite_sensors=False, raw_template='', trans_file='', info_fname='',
                        info=None, read_info_file=True, raw_fname=''):
    return meg.read_sensors_layout(
        mri_subject, args, overwrite_sensors=overwrite_sensors, pick_meg=False, pick_eeg=True,
        trans_file=trans_file, info=info, info_fname=info_fname, read_info_file=read_info_file,
        raw_template=raw_template, raw_fname=raw_fname)


def save_evoked_to_blender(mri_subject, events, args, evoked=None):
    fol = op.join(MMVT_DIR, mri_subject, 'eeg')
    utils.make_dir(fol)
    if '{cond}' in meg.EVO:
        for event_ind, event_id in enumerate(events.keys()):
            if evoked is None:
                evo = mne.read_evokeds(meg.get_cond_fname(meg.EVO, event_id))
            else:
                evo = evoked[event_id]
            if event_ind == 0:
                ch_names = np.array(evo[0].ch_names)
                dt = np.diff(evo[0].times[:2])[0]
                data = np.zeros((evo[0].data.shape[0], evo[0].data.shape[1], 2))
            data[:, :, event_ind] = evo[0].data
    else:
        if evoked is None:
            evoked = mne.read_evokeds(meg.EVO)
        data = evoked[0].data
        data = data[..., np.newaxis]
        ch_names = np.array(evoked[0].ch_names)
        dt = np.diff(evoked[0].times[:2])[0]
    if 'Event' in ch_names:
        event_ind = np.where(ch_names == 'Event')[0]
        ch_names = np.delete(ch_names, event_ind)
        data = np.delete(data, event_ind, 0)
    data_max, data_min = utils.get_data_max_min(data, args.norm_by_percentile, args.norm_percs)
    max_abs = utils.get_max_abs(data_max, data_min)
    if args.normalize_evoked:
        data = data / max_abs
    np.save(op.join(fol, 'eeg_data.npy'), data)
    np.savez(op.join(fol, 'eeg_data_meta.npz'), names=ch_names, conditions=list(events.keys()), dt=dt,
             minmax=(-max_abs, max_abs))
    return True


def calc_minmax(mri_subject, args):
    #todo: merge into save_evoked_to_blender
    fol = op.join(MMVT_DIR, mri_subject, 'eeg')
    data = np.load(op.join(fol, 'eeg_data.npy'))
    data_max, data_min = utils.get_data_max_min(np.diff(data), args.norm_by_percentile, args.norm_percs)
    max_abs = utils.get_max_abs(data_max, data_min)
    np.save(op.join(fol, 'eeg_data_minmax.npy'), [-max_abs, max_abs])
    return op.isfile(op.join(fol, 'eeg_data_minmax.npy'))


def create_helmet_mesh(subject, excludes=[], overwrite_faces_verts=True):
    return meg.create_helmet_mesh(subject, excludes, overwrite_faces_verts, modality='eeg')


def calc_eeg_mesh_verts_sensors(subject, sensors_verts, helmet_verts, modality='eeg'):
    from scipy.spatial.distance import cdist
    max_dists = np.max(np.min(cdist(sensors_verts, helmet_verts), axis=1))
    if max_dists > 0.01:
        raise Exception('Wrong distances!')
    eeg_helmet_indices = np.argmin(cdist(sensors_verts, helmet_verts), axis=1)
    utils.save(eeg_helmet_indices, op.join(MMVT_DIR, subject, modality, '{}_vertices_sensors.pkl'.format(modality)))


def snap_sensors_to_outer_skin(subject, overwrite=True):
    from src.preproc.electrodes import snap_electrodes_to_surface

    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'eeg'))
    output_fname = op.join(fol, 'eeg_snap_sensors.npz')
    if op.isfile(output_fname) and not overwrite:
        print('{} already exist'.format(output_fname))
        return True

    eeg_pos_fname = op.join(fol, 'eeg_sensors_positions.npz')
    if not op.isfile(eeg_pos_fname):
        print('EEG sensors pos file does not exist! {}'.format(eeg_pos_fname))
        return False
    eeg_info = utils.Bag(np.load(eeg_pos_fname))
    pos = eeg_info.pos * 10
    return snap_electrodes_to_surface(
        subject, pos, grid_name='sensors', subjects_dir=SUBJECTS_MRI_DIR, surface='seghead',
        overwrite=False, snap_to_pial=True, points_type='eeg', surface_per_hemi=False)


def init(subject, args, mri_subject='', remote_subject_dir=''):
    if mri_subject == '':
        mri_subject = subject
    fname_format, fname_format_cond, conditions = meg.init_main(subject, mri_subject, remote_subject_dir, args)
    meg.init_globals_args(subject, mri_subject, fname_format, fname_format_cond, args=args)
    # meg.MEG_DIR = SUBJECTS_EEG_DIR
    meg.FWD = meg.FWD_EEG
    meg.INV = meg.INV_EEG
    stat = meg.STAT_AVG if len(conditions) == 1 else meg.STAT_DIFF
    SUBJECT_EEG_DIR = op.join(SUBJECTS_EEG_DIR, subject)
    meg.locating_file = partial(utils.locating_file, parent_fol=SUBJECT_EEG_DIR)
    return conditions, stat


def main(tup, remote_subject_dir, args, flags):
    (subject, mri_subject), inverse_method = tup
    evoked, epochs = None, None
    conditions, stat = init(subject, args, mri_subject, remote_subject_dir)

    if utils.should_run(args, 'read_sensors_layout'):
        flags['read_sensors_layout'] = read_sensors_layout_args(mri_subject, args)

    flags, evoked, epochs = meg.calc_evokes_wrapper(subject, conditions, args, flags, mri_subject=mri_subject)

    flags = meg.calc_fwd_inv_wrapper(subject, args, conditions, flags, mri_subject)

    flags, stcs_conds, stcs_num = meg.calc_stc_per_condition_wrapper(
        subject, conditions, inverse_method, args, flags)

    flags = meg.calc_labels_avg_per_condition_wrapper(
        subject, conditions, args.atlas, inverse_method, stcs_conds, args, flags, stcs_num, None, epochs)

    if 'calc_stc_zvals' in args.function:
        flags['calc_stc_zvals'] = meg.calc_stc_zvals(
            subject, args.stc_name, args.baseline_stc_name, 'eeg', args.use_abs, args.from_index,
            args.to_index, args.stc_zvals_name, False, args.overwrite_stc)

    if utils.should_run(args, 'create_helmet_mesh'):
        flags['create_helmet_mesh'] = create_helmet_mesh(mri_subject, args.eeg_electrodes_excluded_from_mesh)

    if utils.should_run(args, 'save_evoked_to_blender'):
        flags['save_evoked_to_blender'] = save_evoked_to_blender(mri_subject, conditions, args, evoked)

    if utils.should_run(args, 'calc_minmax'):
        flags['calc_minmax'] = calc_minmax(mri_subject, args)

    if 'plot_evoked' in args.function:
        flags['plot_evoked'], _ = plot_evoked(
            args.evo_fname, args.evoked_key, args.pick_meg, args.pick_eeg, args.pick_eog, args.ssp_proj,
            args.spatial_colors, args.window_title, args.hline, args.channels_to_exclude)

    if 'snap_sensors_to_outer_skin' in args.function:
        flags['snap_sensors_to_outer_skin'] = snap_sensors_to_outer_skin(mri_subject)


    return flags


def read_cmd_args(argv=None, subject='', mri_subject='', atlas=''):
    if argv is None and subject != '':
        mri_subject = subject if mri_subject == '' else mri_subject
        argv = ['-s', subject, '-m', mri_subject]
    args = meg.read_cmd_args(argv)
    if atlas != '' and args.atlas != atlas:
        args.atlas = atlas
    args.pick_meg = False
    args.pick_eeg = True
    args.reject = False
    args.fwd_usingMEG = False
    args.fwd_usingEEG = True
    args.modality = 'eeg'
    args.meg_dir = utils.get_link_dir(LINKS_DIR, 'eeg')
    if not op.isdir(args.meg_dir):
        raise Exception('EEG dir can\'t be found! Please rerun src.setup with -f create_links')
    return args


def call_main(args):
    subjects_itr, subject_func = meg.get_subjects_itr_func(args)
    pu.run_on_subjects(args, main, subjects_itr, subject_func)


if __name__ == '__main__':
    from src.utils import preproc_utils as pu
    from itertools import product

    args = read_cmd_args()
    subjects_itr = product(zip(args.subject, args.mri_subject), args.inverse_method)
    subject_func = lambda x:x[0][1]
    pu.run_on_subjects(args, main, subjects_itr, subject_func)
    print('finish!')
