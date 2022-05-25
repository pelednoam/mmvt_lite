import os.path as op
import numpy as np
import shutil
import csv
import nibabel as nib
# from mne.label import _read_annot
from collections import Iterable
import glob

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import args_utils as au
from src.utils import freesurfer_utils as fu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def create_freeview_cmd(subject, args):#, atlas, bipolar, create_points_files=True, way_points=False):
    blender_freeview_fol = op.join(MMVT_DIR, subject, 'freeview')
    freeview_command = 'freeview -v T1.mgz:opacity=0.3 ' + \
        '{0}+aseg.mgz:opacity=0.05:colormap=lut:lut={0}ColorLUT.txt '.format(args.atlas)
    if args.elecs_names:
        groups = set([utils.elec_group(name, args.bipolar) for name in args.elecs_names])
        freeview_command += '-w ' if args.way_points else '-c '
        postfix = '.label' if args.way_points else '.dat'
        for group in groups:
            freeview_command += group + postfix + ' '
    utils.make_dir(blender_freeview_fol)
    freeview_cmd_fname = op.join(blender_freeview_fol, 'run_freeview.sh')
    with open(freeview_cmd_fname, 'w') as sh_file:
        sh_file.write(freeview_command)
    print(freeview_command)
    return op.isfile(freeview_cmd_fname)


# todo: fix duplications!
@utils.tryit()
def create_lut_file_for_atlas(subject, atlas):
    from mne.label import _read_annot
    if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))):
        print('No annot file was found for {}!'.format(atlas))
        print('Run python -m src.preproc.anatomy -s {} -a {} -f create_surfaces,create_annotation'.format(subject, atlas))
        return False

    # Read the subcortical segmentation from the freesurfer lut
    new_lut_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}ColorLUT.txt'.format(atlas))
    mmvt_lut_fname = op.join(MMVT_DIR, subject, 'freeview', '{}ColorLUT.txt'.format(atlas))
    # if op.isfile(mmvt_lut_fname) and not args.overwrite_aseg_file:
    #     return
    lut = utils.read_freesurfer_lookup_table(get_colors=True)
    lut_new = [[l[0], l[1].astype(str), l[2], l[3], l[4], l[5]] for l in lut if l[0] < 1000]
    for hemi, offset in zip(['lh', 'rh'], [1000, 2000]):
        if hemi == 'lh':
            lut_new.append([offset, 'ctx-lh-unknown', 25, 5,  25, 0])
        else:
            lut_new.append([offset, 'ctx-rh-unknown', 25,  5, 25,  0])
        _, ctab, names = _read_annot(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas)))
        names = [name.astype(str) for name in names]
        for index, (label, cval) in enumerate(zip(names, ctab)):
            r,g,b,a, _ = cval
            lut_new.append([index + offset + 1, label, r, g, b, a])
    lut_new.sort(key=lambda x:x[0])
    # Add the values above 3000
    for l in [l for l in lut if l[0] >= 3000]:
        lut_new.append([l[0], l[1].astype(str), l[2], l[3], l[4], l[5]])
    with open(new_lut_fname, 'w') as fp:
        csv_writer = csv.writer(fp, delimiter='\t')
        csv_writer.writerows(lut_new)
    # np.savetxt(new_lut_fname, lut_new, delimiter='\t', fmt="%s")
    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    utils.copy_file(new_lut_fname, mmvt_lut_fname)
    lut_npz_fname = utils.change_fname_extension(mmvt_lut_fname, 'npz')
    x = np.genfromtxt(mmvt_lut_fname, dtype=np.str)
    np.savez(lut_npz_fname, names=x[:, 1], ids=x[:, 0].astype(int))
    return op.isfile(mmvt_lut_fname) and op.isfile(lut_npz_fname)


@utils.check_for_freesurfer
def create_aparc_aseg_file(subject, atlas, overwrite_aseg_file=False, print_only=False, args={}):
    if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))):
        print('No annot file was found for {}!'.format(atlas))
        print('Run python -m src.preproc.anatomy -s {} -a {} -f create_surfaces,create_annotation'.format(subject, atlas))
        return False

    # aparc_aseg_fname
    ret = fu.create_aparc_aseg_file(
        subject, atlas, SUBJECTS_DIR, overwrite_aseg_file, print_only, mmvt_args=args)
    if isinstance(ret, Iterable):
        ret, aparc_aseg_fname = ret
    if not ret:
        return False

    aparc_aseg_file = utils.namebase_with_ext(aparc_aseg_fname)
    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    blender_file = op.join(MMVT_DIR, subject, 'freeview', aparc_aseg_file)
    utils.remove_file(blender_file)
    utils.copy_file(aparc_aseg_fname, blender_file)
    atlas_mat_fname = utils.change_fname_extension(blender_file, 'npy')
    if not op.isfile(atlas_mat_fname) or overwrite_aseg_file:
        d = nib.load(blender_file)
        x = d.get_data()
        np.save(atlas_mat_fname, x)
    return op.isfile(blender_file) and op.isfile(atlas_mat_fname)


# def check_mgz_values(atlas):
#     import nibabel as nib
#     vol = nib.load(op.join(MMVT_DIR, subject, 'freeview', '{}+aseg.mgz'.format(atlas)))
#     vol_data = vol.get_data()
#     vol_data = vol_data[np.where(vol_data)]
#     data = vol_data.ravel()
#     import matplotlib.pyplot as plt
#     plt.hist(data, bins=100)
#     plt.show()


def create_electrodes_points(subject, args): # bipolar=False, create_points_files=True, create_volume_file=False,
                             # way_points=False, electrodes_pos_fname=''):
    if len(args.elecs_names) == 0:
        return True
    groups = set([utils.elec_group(name, args.bipolar) for name in args.elecs_names])
    freeview_command = 'freeview -v T1.mgz:opacity=0.3 aparc+aseg.mgz:opacity=0.05:colormap=lut ' + \
        ('-w ' if args.way_points else '-c ')
    for group in groups:
        postfix = 'label' if args.way_points else 'dat'
        freeview_command = freeview_command + group + postfix + ' '
        group_pos = np.array([pos for name, pos in zip(args.elecs_names, args.elecs_pos) if
                              utils.elec_group(name, args.bipolar) == group])
        file_name = '{}.{}'.format(group, postfix)
        with open(op.join(MMVT_DIR, subject, 'freeview', file_name), 'w') as fp:
            writer = csv.writer(fp, delimiter=' ')
            if args.way_points:
                writer.writerow(['#!ascii label  , from subject  vox2ras=Scanner'])
                writer.writerow([len(group_pos)])
                points = np.hstack((np.ones((len(group_pos), 1)) * -1, group_pos, np.ones((len(group_pos), 1))))
                writer.writerows(points)
            else:
                writer.writerows(group_pos)
                writer.writerow(['info'])
                writer.writerow(['numpoints', len(group_pos)])
                writer.writerow(['useRealRAS', '0'])

    if args.create_volume_file:
        import nibabel as nib
        from itertools import product
        sig = nib.load(op.join(MMVT_DIR, subject, 'freeview', 'T1.mgz'))
        sig_header = sig.get_header()
        data = np.zeros((256, 256, 256), dtype=np.int16)
        # positions_ras = np.array(utils.to_ras(electrodes_positions, round_coo=True))
        elecs_pos = np.array(args.elecs_pos, dtype=np.int16)
        for pos_ras in elecs_pos:
            for x, y, z in product(*([[d+i for i in range(-5,6)] for d in pos_ras])):
                data[z,y,z] = 1
        img = nib.Nifti1Image(data, sig_header.get_affine(), sig_header)
        nib.save(img, op.join(MMVT_DIR, subject, 'freeview', 'electrodes.nii.gz'))


def copy_T1(subject):
    files_exist = True
    for brain_file in ['T1.mgz', 'orig.mgz']:
        blender_brain_file = op.join(MMVT_DIR, subject, 'freeview', brain_file)
        subject_brain_file = op.join(SUBJECTS_DIR, subject, 'mri', brain_file)
        if not op.isfile(blender_brain_file):
            utils.copy_file(subject_brain_file, blender_brain_file)
        files_exist = files_exist and op.isfile(blender_brain_file)
    return files_exist


def read_electrodes_pos(subject, args):
    # electrodes_file = args.electrodes_pos_fname if args.electrodes_pos_fname != '' else op.join(
    #     SUBJECTS_DIR, subject, 'electrodes', 'electrodes{}_positions.npz'.format('_bipolar' if args.bipolar else ''))
    if args.electrodes_pos_fname != '':
        electrodes_file = args.electrodes_pos_fname
    else:
        # electrodes_file = op.join(SUBJECTS_DIR, subject, 'electrodes', 'electrodes{}_{}positions.npz'.format(
        #     '_bipolar' if args.bipolar else '', 'snap_' if args.snap else ''))
        electrodes_files = glob.glob(op.join(MMVT_DIR, subject, 'electrodes', 'electrodes*_*positions.npz'))
        if len(electrodes_files) == 0:
            electrodes_files = glob.glob(op.join(SUBJECTS_DIR, subject, 'electrodes', 'electrodes*_*positions.npz'))
        if len(electrodes_files) == 0:
            print('No electrodes pos files were found!')
            return [], []
        electrodes_file = utils.select_one_file(electrodes_files, 'electrodes*_*positions.npz', 'electrodes')

    if op.isfile(electrodes_file):
        elecs = np.load(electrodes_file)
        elecs_pos, elecs_names = elecs['pos'], [name.astype(str) for name in elecs['names']]
        return elecs_pos, elecs_names
    else:
        # raise Exception("Can't find the electrode coordinates file! ({})".format(electrodes_file))
        print("Can't find the electrode coordinates file! ({})".format(electrodes_file))
        return [], []

# def read_vox2ras0():
#     import nibabel as nib
#     from nibabel.affines import apply_affine
#     mri = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz'))
#     mri_header = mri.get_header()
#     ras_tkr2vox = np.linalg.inv(mri_header.get_vox2ras_tkr())
#     vox2ras = mri_header.get_vox2ras()
#     ras_rkr2ras = np.dot(ras_tkr2vox, vox2ras)
#     print(np.dot([-22.37, 22.12, -11.70], ras_rkr2ras))
#     print('sdf')


def main(subject, remote_subject_dir, args, flags):
    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    args.elecs_pos, args.elecs_names = read_electrodes_pos(subject, args)

    if utils.should_run(args, 'copy_T1'):
        flags['copy_T1'] = copy_T1(subject)

    if utils.should_run(args, 'create_freeview_cmd'):
        flags['create_freeview_cmd'] = create_freeview_cmd(subject, args)

    if utils.should_run(args, 'create_electrodes_points'):
        flags['create_electrodes_points'] = create_electrodes_points(subject, args)

    if utils.should_run(args, 'create_aparc_aseg_file'):
        flags['create_aparc_aseg_file'] = create_aparc_aseg_file(subject, args.atlas, args.overwrite_aseg_file)

    if utils.should_run(args, 'create_lut_file_for_atlas'):
        flags['create_lut_file_for_atlas'] = create_lut_file_for_atlas(subject, args.atlas)

    return flags


def read_cmd_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='MMVT freeview preprocessing')
    parser.add_argument('-b', '--bipolar', help='bipolar', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_aseg_file', help='overwrite_aseg_file', required=False, default=0, type=au.is_true)
    parser.add_argument('--create_volume_file', help='create_volume_file', required=False, default=1, type=au.is_true)
    parser.add_argument('--electrodes_pos_fname', help='electrodes_pos_fname', required=False, default='')
    parser.add_argument('--way_points', help='way_points', required=False, default=0, type=au.is_true)
    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.necessary_files = {'mri': ['T1.mgz', 'orig.mgz']}
    # print(args)
    return args


if __name__ == '__main__':
    args = read_cmd_args(None)
    pu.run_on_subjects(args, main)
    print('finish!')