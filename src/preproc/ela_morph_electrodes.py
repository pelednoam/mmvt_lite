import glob
import os.path as op
import importlib
import sys
import nibabel as nib
import numpy as np
from scipy.spatial.distance import pdist, cdist

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import labels_utils as lu
from src.preproc import anatomy as anat

mmvt_code_fol = utils.get_mmvt_code_root()
ela_code_fol = op.join(utils.get_parent_fol(mmvt_code_fol), 'electrodes_rois')
if ela_code_fol not in sys.path:
    sys.path.append(ela_code_fol)

from find_rois import main as find_rois_main
from find_rois import freesurfer_utils as fu
importlib.reload(find_rois_main)

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
WHITES = set(['Left-Cerebral-White-Matter', 'Left-Cerebellum-White-Matter', 'Right-Cerebral-White-Matter', 'Right-Cerebellum-White-Matter'])

mri_robust_register = 'mri_robust_register --mov {subjects_dir}/{subject_from}/mri/T1.mgz --dst {subjects_dir}/{subject_to}/mri/T1.mgz --lta {subjects_dir}/{subject_from}/mri/{lta_name}.lta --satit --mapmov {subjects_dir}/{subject_from}/mri/T1_to_{subject_to}.mgz --cost nmi'


def init(subject, atlas, n_jobs):
    from src.utils import geometry_utils as gu
    if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))):
        anat.create_annotation(subject, atlas)
        if not utils.both_hemi_files_exist(
                op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))):
            raise Exception('Can\'t find the cortical atlas {} for subject {}'.format(atlas, subject))
    labels_vertices = find_rois_main.read_labels_vertices(SUBJECTS_DIR, subject, atlas, n_jobs)
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    labels_names = [l.name for l in labels]
    aseg_atlas_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'aseg.mgz')
    aseg_data = nib.load(aseg_atlas_fname).get_data()
    lut = fu.import_freesurfer_lut()
    pia_verts = {}
    for hemi in ['rh', 'lh']:
        pia_verts[hemi], _ = gu.read_surface(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)))
        # pia_verts[hemi], _ = nib.freesurfer.read_geometry(
        #     op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)))
    subs_center_of_mass, subs_names = calc_subcorticals_pos(subject, aseg_data, lut)
    labels_center_of_mass = lu.calc_center_of_mass(labels, ret_mat=True) * 1000
    regions_center_of_mass = np.concatenate((labels_center_of_mass, subs_center_of_mass))
    regions_names = labels_names + subs_names
    # save_com_as_elecs(subject, regions_center_of_mass, regions_names, atlas)
    # save_com_as_elecs(subject, subs_center_of_mass, subs_names, atlas)
    return labels_vertices, regions_center_of_mass, regions_names, aseg_data, lut, pia_verts,


def save_com_as_elecs(subject, regions_center_of_mass, regions_names, atlas):
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'electrodes'))
    output_fname = op.join(fol, 'electrodes_positions_com_{}.npz'.format(atlas))
    regions_names = ['{}_com'.format(c) for c in regions_names]
    np.savez(output_fname, pos=regions_center_of_mass, names=regions_names, pos_org=[])


def calc_subcorticals_pos(subject, aseg_data, lut):
    output_fname = op.join(MMVT_DIR, subject, 'subcorticals_pos.npz')
    if op.isfile(output_fname):
        d = np.load(output_fname)
        return d['pos'], list(d['names'])
    subject_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).header
    # codes_file = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
    # subcortical_lookup = np.genfromtxt(codes_file, dtype=str, delimiter=',')
    aseg_unique_vals = np.unique(aseg_data)
    subcortical_lookup = [(label, index) for label, index in zip(lut['label'], lut['index'])
                          if index < 1000 and label not in WHITES and label != 'Unknown' and index in aseg_unique_vals]
    subs_pos = np.zeros((len(subcortical_lookup), 3))
    for sub_ind, (sub_name, sub_code) in enumerate(subcortical_lookup):
        t1_inds = np.where(aseg_data == int(sub_code))
        center_vox = np.array(t1_inds).mean(axis=1)
        subs_pos[sub_ind] = utils.apply_trans(subject_header.get_vox2ras_tkr(), center_vox)
    names = [name for name, index in subcortical_lookup]
    np.savez(output_fname, pos=subs_pos, names=names)
    return subs_pos, names


def get_electrodes_info(subject, atlas, bipolar, n_jobs=1):
    cmd_args = ['-s', subject, '-a', atlas, '-b', str(bipolar), '--n_jobs', str(n_jobs)]
    args = find_rois_main.get_args(cmd_args)
    elecs_names, elecs_pos, elecs_dists, elecs_types, _ = find_rois_main.get_electrodes(subject, bipolar, args)
    elecs_oris = find_rois_main.get_electrodes_orientation(elecs_names, elecs_pos, bipolar, elecs_types)
    return elecs_names, elecs_pos, elecs_dists, elecs_types, elecs_oris, args.excludes


@utils.timeit
def calc_elas(subject, template, specific_elecs_names=[], bipolar=False, atlas='aparc.DKTatlas',
              error_radius=3, elc_length=4, print_warnings=False, overwrite=False, n_jobs=1):
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'electrodes', 'ela_morphed'))
    elecs_names, elecs_pos, elecs_dists, elecs_types, elecs_oris, excludes = get_electrodes_info(
        subject, atlas, bipolar, n_jobs)
    specific_elecs_names = specific_elecs_names if len(specific_elecs_names) > 0 else elecs_names
    elecs_info = [(elec_name, elec_pos, elec_dist, elec_type, elec_ori) for
                  elec_name, elec_pos, elec_dist, elec_type, elec_ori in \
                  zip(elecs_names, elecs_pos, elecs_dists, elecs_types, elecs_oris)
                  if elec_name in specific_elecs_names]
    (labels_vertices, regions_center_of_mass, regions_names, aseg_data, lut, pia_verts) = init(
        subject, atlas, n_jobs)
    atlas = utils.fix_atlas_name(subject, atlas, SUBJECTS_DIR)
    annot_fname = op.join(SUBJECTS_DIR, template, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    if not utils.both_hemi_files_exist(annot_fname):
        raise Exception('No {} atlas for {}!'.format(annot_fname, template))
    else:
        template_atlas = atlas

    (template_labels_vertices, template_regions_center_of_mass, template_regions_names, template_aseg_data, lut,
     template_pia_verts) = init(template, template_atlas, n_jobs)
    len_lh_pia = len(pia_verts['lh'])
    template_len_lh_pia = len(template_pia_verts['lh'])
    template_header = nib.load(op.join(SUBJECTS_DIR, template, 'mri', 'T1.mgz')).header
    epsilon = 0
    max_run_num = 1000
    parallel = True

    for elec_name, elec_pos, elec_dist, elec_type, elec_ori in elecs_info:
        elec_output_fname = op.join(fol, '{}_ela_morphed.npz'.format(elec_name))
        if op.isfile(elec_output_fname) and not overwrite:
            d = np.load(elec_output_fname)
            print('{}: err: {}, new_pos={}'.format(elec_name, d['err'], d['pos']))
            continue
        elec_labeling = calc_ela(
            subject, bipolar, elec_name, elec_pos, elec_type, elec_ori, elec_dist, labels_vertices, aseg_data, lut,
            pia_verts, len_lh_pia, excludes, error_radius, elc_length, print_warnings, overwrite, n_jobs)
        print('subject_ela:')
        print_ela(elec_labeling)

        elec_labeling_no_whites = calc_elec_labeling_no_white(elec_labeling)
        template_elec_pos = calc_prob_pos(
            elec_labeling_no_whites, template_regions_center_of_mass, template_regions_names)
        if template_elec_pos is None:
            print('!!!Error in {}!!!\nelec_labeling_no_whites={}'.format(elec_name, elec_labeling_no_whites))
            continue
        subject_prob_pos_in_template_space = template_elec_pos.copy()
        template_elec_vox = np.rint(
            utils.apply_trans(np.linalg.inv(template_header.get_vox2ras_tkr()), template_elec_pos).astype(int))

        elec_labeling_template = calc_ela(
            template, bipolar, elec_name, template_elec_pos, elec_type, elec_ori, elec_dist, template_labels_vertices, template_aseg_data, lut,
            template_pia_verts, template_len_lh_pia, excludes, error_radius, elc_length, print_warnings, overwrite, n_jobs)
        err = comp_elecs_labeling(
            elec_labeling_template, template_regions_center_of_mass, template_regions_names,
            subject_prob_pos_in_template_space)
        run_num = 0
        stop_gradient = False
        print(err)
        while not stop_gradient and err > epsilon and run_num < max_run_num:
            dxyzs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            if parallel:
                new_template_elec_pos_arr = [
                    utils.apply_trans(template_header.get_vox2ras_tkr(), template_elec_vox + dxyz) for
                    dxyz in dxyzs]
                params = [(template, bipolar, elec_name, new_template_elec_pos, elec_type, elec_ori, elec_dist,
                           template_labels_vertices, template_aseg_data, lut, template_pia_verts, template_len_lh_pia,
                           elec_labeling_no_whites, regions_center_of_mass, regions_names,
                           template_regions_center_of_mass, template_regions_names, subject_prob_pos_in_template_space,
                           excludes, error_radius, elc_length, overwrite)
                          for new_template_elec_pos in new_template_elec_pos_arr]
                results = utils.run_parallel(_parallel_calc_ela_err, params, len(dxyzs) if n_jobs > 1 else 1)
                errs = [res[1] for res in results]
                ind = np.argmin(errs)
                new_template_pos = new_template_elec_pos_arr[ind]
                best_ela = results[ind][0]
                regions, new_probs = norm_probs(calc_elec_labeling_no_white(best_ela))
                print(['{} ({}) '.format(r, p) for r, p in zip(regions, new_probs)])
                min_err = errs[ind]
                if min_err >= err:
                    stop_gradient = True
                else:
                    err = min_err
            else:
                for dxyz in dxyzs:
                    new_template_elec_pos = utils.apply_trans(
                        template_header.get_vox2ras_tkr(), template_elec_vox + dxyz)
                    elec_labeling_template = calc_ela(
                        template, bipolar, elec_name, new_template_elec_pos, elec_type, elec_ori, elec_dist,
                        template_labels_vertices, template_aseg_data, lut,
                        template_pia_verts, template_len_lh_pia, excludes, error_radius, elc_length,
                        print_warnings, overwrite, n_jobs)
                    min_err = comp_elecs_labeling(
                        elec_labeling_template, template_regions_center_of_mass, template_regions_names,
                        subject_prob_pos_in_template_space)
                    if min_err < err:
                        new_template_pos = new_template_elec_pos
                        err = min_err
                        break
                else:
                    stop_gradient = True

            print('*** {}){} ***'.format(run_num + 1, min_err))
            print_ela(elec_labeling_template)
            run_num += 1
            template_elec_vox = np.rint(
                utils.apply_trans(np.linalg.inv(template_header.get_vox2ras_tkr()), new_template_pos).astype(int))
            if stop_gradient:
                print('Stop gradient!!!')
                print('subject_ela:')
                print_ela(elec_labeling)
                print('template ela:')
                print_ela(elec_labeling_template)
        print('Save output to {}'.format(elec_output_fname))
        np.savez(elec_output_fname, pos=new_template_pos, name=elec_name, err=min_err)


def print_ela(ela):
    print(','.join(sorted(['{}:{:.2f}'.format(region, prob) for region, prob in zip(
        ela['regions'], ela['regions_probs'])])))


def _parallel_calc_ela_err(p):
    (template, bipolar, elec_name, new_template_elec_pos, elec_type, elec_ori, elec_dist, template_labels_vertices,
     template_aseg_data, lut, template_pia_verts, template_len_lh_pia, elec_labeling_no_whites, regions_center_of_mass,
     regions_names, template_regions_center_of_mass, template_regions_names, subject_prob_pos, excludes,
     error_radius, elc_length, overwrite) = p
    elec_labeling_template = calc_ela(
        template, bipolar, elec_name, new_template_elec_pos, elec_type, elec_ori, elec_dist,
        template_labels_vertices, template_aseg_data, lut,
        template_pia_verts, template_len_lh_pia, excludes, error_radius, elc_length, overwrite, 1)
    err = comp_elecs_labeling(
        elec_labeling_template, template_regions_center_of_mass, template_regions_names,
        subject_prob_pos)
    return elec_labeling_template, err


def calc_elec_labeling_no_white(elec_labeling):
    return [(region, prob) for region, prob in zip(elec_labeling['regions'], elec_labeling['regions_probs']) if
        region not in WHITES]


def calc_prob_pos(elec_labeling_no_whites, regions_center_of_mass, regions_names):
    # norm = 1 / sum([p for _, p in elec_labeling_no_whites])
    # new_probs = [p * norm for _, p in elec_labeling_no_whites]
    # regions = [r for r, _ in elec_labeling_no_whites]
    try:
        regions, new_probs = norm_probs(elec_labeling_no_whites)
        elec_prob_dist = [p * regions_center_of_mass[regions_names.index(r)] for p, r in zip(new_probs, regions)]
        return np.sum(elec_prob_dist, axis=0)
    except:
        return None


def norm_probs(elec_labeling):
    norm = 1 / sum([p for _, p in elec_labeling])
    new_probs = [p * norm for _, p in elec_labeling]
    regions = [r for r, _ in elec_labeling]
    return regions, new_probs


def comp_elecs_labeling(elec_labeling_template, template_regions_center_of_mass, template_regions_names,
                        subject_prob_pos):
    template_elec_lebeling_no_whites = calc_elec_labeling_no_white(elec_labeling_template)
    template_prob_pos = calc_prob_pos(
        template_elec_lebeling_no_whites, template_regions_center_of_mass, template_regions_names)
    err = pdist([subject_prob_pos, template_prob_pos])[0]
    return err


def calc_ela(subject, bipolar, elec_name, elec_pos, elec_type, elec_ori, elec_dist, labels, aseg_data, lut, pia_verts,
             len_lh_pia, excludes, error_radius=3, elc_length=4, print_warnings=False, overwrite=False, n_jobs=1):
    enlarge_if_no_hit, hit_min_three, strech_to_dist, hit_only_cortex = True, True, False, False
    nei_dimensions = None
    ret_val = find_rois_main.identify_roi_from_atlas_per_electrode(
        labels, elec_pos, pia_verts, len_lh_pia, lut,
        aseg_data, elec_name, error_radius, elc_length, nei_dimensions, elec_ori, elec_dist, elec_type, strech_to_dist,
        enlarge_if_no_hit, hit_min_three, hit_only_cortex, bipolar, SUBJECTS_DIR, subject, excludes, print_warnings,
        n_jobs=1)

    (regions, regions_hits, subcortical_regions, subcortical_hits, approx_after_strech, elc_length, elec_hemi_vertices,
     elec_hemi_vertices_dists, hemi) = ret_val

    regions_probs = np.hstack((regions_hits, subcortical_hits)) / float(
        np.sum(regions_hits) + np.sum(subcortical_hits))
    ret = {'name': elec_name, 'regions': regions + subcortical_regions,
           'regions_probs': regions_probs, 'hemi': hemi}
    return ret


def write_electrodes_pos(subject, subject_to, specific_elecs=[]):
    names, pos = [], []
    morphed_electrodes_files = glob.glob(op.join(
        MMVT_DIR, subject, 'electrodes', 'ela_morphed', '*_ela_morphed.npz'))
    for elec_fname in morphed_electrodes_files:
        d = np.load(elec_fname)
        elc_name = d['name']
        if len(specific_elecs) > 0 and elc_name not in specific_elecs:
            continue
        names.append('{}_{}'.format(subject, elc_name))
        pos.append(d['pos'])
    fol = utils.make_dir(op.join(MMVT_DIR, subject_to, 'electrodes'))
    output_fname = op.join(fol, 'electrodes_positions_from_{}.npz'.format(subject))
    np.savez(output_fname, pos=pos, names=names, pos_org=[])
    return op.isfile(output_fname)

#
# if __name__ == '__main__':
#     subject = 'mg114'
#     elec_name = ['RPT10']
#     template = 'colin27'
#     # atlas = 'aparc.DKTatlas40'
#     atlas = 'laus125'
#     overwrite = True
#
#     calc_elas(subject, template, elec_name, bipolar=False, atlas=atlas, print_warnings=False, overwrite=overwrite,
#               n_jobs=1)
#     write_electrodes_pos(subject, template, elec_name)