try:
    import mne.surface
except:
    print('No mne!')
import time
import os.path as op
import numpy as np
import os
import shutil
import glob
import traceback
import functools
from collections import defaultdict, Counter
# from tqdm import tqdm
try:
    import nibabel as nib
except:
    print('No nibabel!')
from functools import partial
import re

from src.mmvt_addon import mmvt_utils as mu
from src.utils import freesurfer_utils as fu
from src.utils import args_utils as au

read_labels_from_annots = mu.read_labels_from_annots
read_labels_from_annot = mu.read_labels_from_annot
Label = mu.Label

try:
    from src.utils import utils
    from src.utils import preproc_utils as pu
    SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
    from src.mmvt_addon import colors_utils as cu
except:
    print("Sorry, no src libs...")
    utils.print_last_error_line()

HEMIS = ['rh', 'lh']


def find_template_brain_with_annot_file(aparc_name, fsaverage, subjects_dir, find_in_all=True):
    optional_templates = []
    original_template_brain = fsaverage[0] if len(fsaverage) == 1 else ''
    if isinstance(fsaverage, str):
        # fsaverage = [fsaverage]
        fsaverage = fsaverage.split(',')
    fsaverage = list(set(fsaverage))
    for fsav in fsaverage:
        fsaverage_annot_files_exist = utils.both_hemi_files_exist(op.join(
            subjects_dir, fsav, 'label', '{}.{}.annot'.format('{hemi}', aparc_name)))
        fsaverage_labels_exist = len(glob.glob(op.join(subjects_dir, fsav, 'label', aparc_name, '*.label'))) > 0
        if fsaverage_annot_files_exist or fsaverage_labels_exist:
            optional_templates.append(fsav)
    if len(optional_templates) == 0:
        if find_in_all:
            fsaverage.extend([utils.namebase(d) for d in glob.glob(op.join(subjects_dir, 'fs*'))])
        for fsav in fsaverage:
            fsaverage_annot_files_exist = utils.both_hemi_files_exist(op.join(
                subjects_dir, fsav, 'label', '{}.{}.annot'.format('{hemi}', aparc_name)))
            fsaverage_labels_exist = len(glob.glob(op.join(subjects_dir, fsav, 'label', aparc_name, '*.label'))) > 0
            if fsaverage_annot_files_exist or fsaverage_labels_exist:
                optional_templates.append(fsav)
        if len(optional_templates) == 0:
            print("Can't find the annot file for any of the templates brains! ({})".format(fsaverage))
            return ''
    else:
        if 'fsaverage' in optional_templates and all([t.startswith('fsaverage') for t in optional_templates]):
            return 'fsaverage'
        else:
            if original_template_brain in optional_templates:
                return original_template_brain
            else:
                return utils.select_one_file(optional_templates)


def morph_labels_from_fsaverage(subject, subjects_dir, mmvt_dir, aparc_name='aparc250', fs_labels_fol='',
            sub_labels_fol='', n_jobs=6, fsaverage='fsaverage', overwrite=False):
    fsaverage = find_template_brain_with_annot_file(aparc_name, fsaverage, subjects_dir)
    if fsaverage == '':
        return False
    if subject == fsaverage:
        return True
    subject_dir = op.join(subjects_dir, subject)
    labels_fol = op.join(subjects_dir, fsaverage, 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    sub_labels_fol = op.join(subject_dir, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    if not op.isdir(sub_labels_fol):
        os.makedirs(sub_labels_fol)
    labels = read_labels(fsaverage, subjects_dir, aparc_name, n_jobs=n_jobs)
    verts = utils.load_surf(subject, mmvt_dir, subjects_dir)

    # Make sure we have a morph map, and if not, create it here, and not in the parallel function
    mne.surface.read_morph_map(subject, fsaverage, subjects_dir=subjects_dir)
    indices = np.array_split(np.arange(len(labels)), n_jobs)
    chunks = [([labels[ind] for ind in chunk_indices], subject, fsaverage, labels_fol, sub_labels_fol, verts,
               subjects_dir, overwrite) for chunk_indices in indices]
    results = utils.run_parallel(_morph_labels_parallel, chunks, n_jobs)
    return all(results)


def _morph_labels_parallel(p):
    labels, subject, fsaverage, labels_fol, sub_labels_fol, verts, subjects_dir, overwrite = p
    ok = True
    for fs_label in labels:
        label_file = op.join(labels_fol, '{}.label'.format(fs_label.name))
        local_label_name = op.join(sub_labels_fol, '{}.label'.format(op.splitext(op.split(label_file)[1])[0]))
        if not op.isfile(local_label_name) or overwrite:
            # fs_label = mne.read_label(label_file)
            fs_label.values.fill(1.0)
            sub_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=1, subjects_dir=subjects_dir)
            if np.all(sub_label.pos == 0):
                sub_label.pos = verts[sub_label.hemi][sub_label.vertices]
            sub_label.save(local_label_name)
            ok = ok and op.isfile(local_label_name)
    return ok


def labels_to_annot(subject, subjects_dir='', aparc_name='aparc250', labels_fol='', overwrite=True, labels=[],
                    fix_unknown=True, hemi='both', print_error=False, n_jobs=6):

    if subjects_dir == '':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = op.join(subjects_dir, subject)
    annot_files_exist = utils.both_hemi_files_exist(
        op.join(subject_dir, 'label', '{}.{}.annot'.format('{hemi}', aparc_name)))
    if annot_files_exist and not overwrite:
        return True
    if len(labels) > 1:
        if isinstance(labels[0], str):
            labels_fol = utils.get_parent_fol(labels[0])
            labels = read_labels_parallel(subject, SUBJECTS_DIR, aparc_name, labels_fol=labels_fol, n_jobs=n_jobs)
    if len(labels) == 0:
        labels = []
        labels_fol = op.join(subject_dir, 'label', aparc_name) if labels_fol=='' else labels_fol
        labels_files = glob.glob(op.join(labels_fol, '*.label'))
        if len(labels_files) == 0:
            if not annot_files_exist:
                raise Exception('labels_to_annot: No labels files!')
            else:
                print("Can't find label files, using the annot files instead")
                return True
        for label_file in labels_files:
            if fix_unknown and 'unknown' in utils.namebase(label_file):
                continue
            label = mne.read_label(label_file)
            # print(label.name)
            label.name = get_label_hemi_invariant_name(label.name)
            labels.append(label)
        labels.sort(key=lambda l: l.name)
    if hemi != 'both':
        # todo: read only from this hemi
        labels = [l for l in labels if l.hemi == hemi]
    if overwrite:
        hemis = HEMIS if hemi == 'both' else [hemi]
        for remove_hemi in hemis:
            utils.remove_file(op.join(subject_dir, 'label', '{}.{}.annot'.format(remove_hemi, aparc_name)))
    try:
        mne.write_labels_to_annot(subject=subject, labels=labels, parc=aparc_name, overwrite=overwrite,
                                  subjects_dir=subjects_dir, hemi=hemi)
    except:
        print('Error in writing annot file!')
        utils.print_last_error_line()
        if print_error:
            print(traceback.format_exc())
        return False
    return utils.both_hemi_files_exist(op.join(subject_dir, 'label', '{}.{}.annot'.format('{hemi}', aparc_name))) \
        if hemi == 'both' else op.isfile(op.join(subject_dir, 'label', '{}.{}.annot'.format(hemi, aparc_name)))


def check_labels(subject, atlas, subjects_dir, mmvt_dir, labels=None):
    if labels is None:
        labels = read_labels(subject, subjects_dir, atlas)
    verts = utils.load_surf(subject, mmvt_dir, subjects_dir)
    verts = {hemi:range(len(verts[hemi])) for hemi in utils.HEMIS}
    ok = True
    for hemi in utils.HEMIS:
        labels_indices = []
        for l in labels:
            if l.hemi != hemi:
                continue
            labels_indices.extend(l.vertices.tolist())
        labels_indices = set(labels_indices)
        print('{}: labels vertices len: {}, verts len: {}'.format(hemi, len(labels_indices), len(verts[hemi])))
    for label in labels:
        if not all(np.in1d(label.vertices, verts[label.hemi])):
            print('Not all {} vertices are in {} verts!'.format(label.name, label.hemi))
            ok = False
    return ok


def solve_labels_collision(subject, atlas, subjects_dir, mmvt_dir, backup_atlas, overwrite_vertices_labels_lookup=False,
                           surf_type='inflated', n_jobs=1):
    backup_labels_fol = op.join(subjects_dir, subject, 'label', backup_atlas)
    backup_files = glob.glob(op.join(backup_labels_fol, '*.label'))
    if True: #not op.isdir(backup_labels_fol) or len(backup_files) < 4:
        labels_fol = op.join(subjects_dir, subject, 'label', atlas)
        if op.isdir(backup_labels_fol):
            shutil.rmtree(backup_labels_fol)
        # utils.copy_filetree(labels_fol, backup_labels_fol)
        shutil.copytree(labels_fol, backup_labels_fol)
    return save_labels_from_vertices_lookup(
        subject, atlas, subjects_dir, mmvt_dir, surf_type='pial', read_labels_from_fol=backup_labels_fol,
        overwrite_vertices_labels_lookup=overwrite_vertices_labels_lookup, n_jobs=n_jobs)


def create_unknown_labels(subject, atlas):
    labels_fol = op.join(SUBJECTS_DIR, subject, 'label', atlas)
    utils.make_dir(labels_fol)
    unknown_labels_fname_template = op.join(labels_fol,  'unknown-{}.label'.format('{hemi}'))
    if utils.both_hemi_files_exist(unknown_labels_fname_template):
        unknown_labels = {hemi:mne.read_label(unknown_labels_fname_template.format(hemi=hemi), subject)
                          for hemi in utils.HEMIS}
        return unknown_labels

    unknown_labels = {}
    for hemi in utils.HEMIS:
        labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        unknown_label_name = 'unknown-{}'.format(hemi)
        labels_names = [l.name for l in labels]
        if unknown_label_name not in labels_names:
            verts, _ = utils.read_pial(subject, MMVT_DIR, hemi)
            unknown_verts = set(range(verts.shape[0]))
            for label in labels:
                unknown_verts -= set(label.vertices)
            unknown_verts = np.array(sorted(list(unknown_verts)))
            unknown_label = mne.Label(unknown_verts, hemi=hemi, name=unknown_label_name, subject=subject)
        else:
            unknown_label = labels[labels_names.index(unknown_label_name)]
        unknown_labels[hemi] = unknown_label
        if not op.isfile(unknown_labels_fname_template.format(hemi=hemi)):
            unknown_label.save(unknown_labels_fname_template.format(hemi=hemi))
    return unknown_labels


def fix_unknown_labels(subject, atlas):
    for hemi in utils.HEMIS:
        labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        labels_names = [l.name for l in labels]
        while 'unknown-{}'.format(hemi) in labels_names:
            del labels[labels_names.index('unknown-{}'.format(hemi))]
            labels_names = [l.name for l in labels]
        mne.write_labels_to_annot(
                labels, subject=subject, parc=atlas, overwrite=True, subjects_dir=SUBJECTS_DIR, hemi=hemi)
    return utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}-{}.annot'.format(atlas, '{hemi}')))


def create_vertices_labels_lookup(subject, atlas, save_labels_ids=False, overwrite=False, read_labels_from_fol='',
                                  hemi='both', labels_dict=None, verts_dict=None, check_unknown=True, save_lookup=True):
    from src.utils import geometry_utils as gu

    def check_loopup_is_ok(lookup):
        unique_values_num = sum([len(set(lookup[hemi].values())) for hemi in hemis])
        # check it's not only the unknowns
        lookup_ok = not all([len(set(lookup[hemi].values())) == 1 and 'unknown' in lookup[hemi].values() for hemi in hemis])
        err = ''
        if not lookup_ok:
            err = 'unique_values_num = {}\n'.format(unique_values_num)
        for hemi in hemis:
            if verts_dict is None:
                if utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'surf', '{hemi}.pial')):
                    # verts, _ = nib.freesurfer.read_geometry(
                    #     op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)))
                    verts, _ = gu.read_surface(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)))
                elif utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.pial.ply')):
                    verts, _ = utils.read_pial(subject, MMVT_DIR, hemi)
                elif utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'surf', '{hemi}.pial.ply')):
                    verts, _ = utils.read_pial(subject, SUBJECTS_DIR, hemi)
                else:
                    raise Exception('Can\'t find {} pial surfaces!'.format(subject))
            else:
                verts = verts_dict[hemi]
            lookup_ok = lookup_ok and len(lookup[hemi].keys()) == len(verts)
            if not lookup_ok:
                err += 'len(lookup[{}].keys()) != len(verts) ({}!={})\n'.format(hemi, len(lookup[hemi].keys()), len(verts))
        return lookup_ok, err

    hemis = utils.HEMIS if hemi == 'both' else [hemi]
    output_fname = op.join(MMVT_DIR, subject, '{}_vertices_labels_{}lookup.pkl'.format(
        atlas, 'ids_' if save_labels_ids else ''))
    if op.isfile(output_fname) and not overwrite:
        lookup = utils.load(output_fname)
        loopup_is_ok, _ = check_loopup_is_ok(lookup)
        if loopup_is_ok:
            return lookup
    lookup = {}

    for hemi in hemis:
        lookup[hemi] = {}
        if labels_dict is None:
            if read_labels_from_fol != '':
                labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi, try_first_from_annotation=False,
                                     labels_fol=read_labels_from_fol)
            else:
                labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        else:
            labels = labels_dict[hemi]
        if len(labels) == 0:
            raise Exception("Can't read labels from {} {}".format(subject, atlas))
        labels_names = [l.name for l in labels]
        if check_unknown and len([l for l in labels_names if 'unknown' in l.lower()]) == 0:
            # add the unknown label
            # todo: this code is needed to be debugged!
            annot_fname = get_annot_fnames(subject, SUBJECTS_DIR, atlas, hemi=hemi)[0]
            if op.isfile(annot_fname):
                backup_fname = utils.add_str_to_file_name(annot_fname, '_backup')
                utils.copy_file(annot_fname, backup_fname)
            try:
                mne.write_labels_to_annot(subject=subject, hemi=hemi, labels=labels, parc=atlas, overwrite=True,
                                          subjects_dir=SUBJECTS_DIR)
            except:
                print('create_vertices_labels_lookup: Error writing labels to annot file!')
                print('Creating unknown label manually')
                create_unknown_labels(subject, atlas)
            labels = mne.read_labels_from_annot(
                subject, atlas, subjects_dir=SUBJECTS_DIR, surf_name='pial', hemi=hemi)
            labels_names = [l.name for l in labels]
        if check_unknown and len([l for l in labels_names if 'unknown' in l.lower()]) == 0:
            raise Exception('No unknown label in {}'.format(annot_fname))
        if verts_dict is None:
            if utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'surf', '{hemi}.pial')):
                # verts, _ = nib.freesurfer.read_geometry(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)))
                verts, _ = gu.read_surface(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)))
            elif utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.pial.ply')):
                verts, _ = utils.read_pial(subject, MMVT_DIR, hemi)
            elif utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'surf', '{hemi}.pial.ply')):
                verts, _ = utils.read_pial(subject, SUBJECTS_DIR, hemi)
            else:
                raise Exception('Can\'t find {} pial surfaces!'.format(subject))
        else:
            verts = verts_dict[hemi]
        if len([l for l in labels_names if 'unknown' in l.lower()]) > 0 and \
                sum([len(l.vertices) for l in labels]) != len(verts):
            ret = input('{}: labels vertices num: {} != surface verts num {}. Do you want to continue?'.format(
                hemi, sum([len(l.vertices) for l in labels]), len(verts)))
            if not au.is_true(ret):
                raise Exception('Wrong number of vertices!')
        verts_indices = set(range(len(verts)))
        assign_vertices = set()
        for label in labels:
            for vertice in label.vertices:
                if vertice in verts_indices:
                    lookup[hemi][vertice] = labels_names.index(label.name) if save_labels_ids else label.name
                    assign_vertices.add(vertice)
                else:
                    print('vertice {} of label {} not in verts! ({}, {})'.format(vertice, label.name, subject, hemi))
        not_assign_vertices = verts_indices - assign_vertices
        for vertice in not_assign_vertices:
            lookup[hemi][vertice] = len(labels_names) if save_labels_ids else 'unknown_{}'.format(hemi)
    loopup_is_ok, err = check_loopup_is_ok(lookup)
    if loopup_is_ok:
        if save_lookup:
            utils.save(lookup, output_fname)
        return lookup
    else:
        print('unknown labels: ', [l for l in labels_names if 'unknown' in l])
        raise Exception('Error in vertices_labels_lookup!\n{}'.format(err))


def find_label_vertices(subject, atlas, hemi, vertices, label_template='*'):
    import re
    vertices_labels_lookup = create_vertices_labels_lookup(subject, atlas)
    label_re_template = re.compile(label_template) if label_template != '*' else None
    label_vertices, label_vertices_indices = [], []
    for vert_ind, vert in enumerate(vertices):
        vert_label = vertices_labels_lookup[hemi].get(vert, '')
        if vert_label == '':
            print('find_pick_activity: No label for vert {}'.format(vert))
            continue
        if label_re_template is None or label_re_template.search(vert_label) is not None:
            label_vertices.append(vert)
            label_vertices_indices.append(vert_ind)
    return label_vertices, label_vertices_indices


def save_labels_from_vertices_lookup(
        subject, atlas, subjects_dir, mmvt_dir, surf_type='pial', read_labels_from_fol='',
        overwrite_vertices_labels_lookup=False, overwrite_labels=False, lookup=None, n_jobs=6):
    if lookup is None or len(lookup.get('rh', {})) == 0 or len(lookup.get('lh', {})) == 0:
        lookup = create_vertices_labels_lookup(
            subject, atlas, read_labels_from_fol=read_labels_from_fol, overwrite=overwrite_vertices_labels_lookup)
    labels_fol = utils.make_dir(op.join(subjects_dir, subject, 'label', atlas))
    surf = utils.load_surf(subject, mmvt_dir, subjects_dir)
    if overwrite_labels:
        utils.delete_folder_files(labels_fol)
    ok = True
    for hemi in utils.HEMIS:
        labels_vertices = defaultdict(list)
        # surf_fname = op.join(subjects_dir, subject, 'surf', '{}.{}'.format(hemi, surf_type))
        # surf, _ = mne.surface.read_surface(surf_fname)
        for vertice, label in lookup[hemi].items():
            labels_vertices[label].append(vertice)
        chunks_indices = np.array_split(np.arange(len(labels_vertices)), n_jobs)
        labels_vertices_items = list(labels_vertices.items())
        chunks = [([labels_vertices_items[ind] for ind in chunk_indices], subject, labels_vertices, surf, hemi,
                   labels_fol, overwrite_labels) for chunk_indices in chunks_indices]
        results = utils.run_parallel(_save_labels_from_vertices_lookup_hemi, chunks, n_jobs)
        ok = ok and all(results)
    return ok


def _save_labels_from_vertices_lookup_hemi(p):
    labels_vertices_items, subject, labels_vertices, surf, hemi, labels_fol, overwrite = p
    ok = True
    for label, vertices in labels_vertices_items:
        output_fname = op.join(labels_fol, '{}.label'.format(label))
        if op.isfile(output_fname) and not overwrite:
            continue
        label = get_label_hemi_invariant_name(label)
        if 'unknown' in label.lower():
            # Don't save the unknown label, the labels_to_annot will do that, otherwise there will be 2 unknown labels
            continue
        new_label = mne.Label(sorted(vertices), surf[hemi][vertices], hemi=hemi, name=label, subject=subject)
        new_label.save(op.join(labels_fol, label))
        ok = ok and op.isfile(output_fname)
    return ok


def morph_annot(subject, template_brain, atlas, overwrite_subject_vertices_labels_lookup=False,
                overwrite_labels=False, overwrite_annot=False, n_jobs=4, subjects_dir='',
                do_create_atlas_coloring=True):
    vertices_labels_lookup = calc_subject_vertices_labels_lookup_from_template(
        subject, template_brain, atlas, subjects_dir, overwrite_subject_vertices_labels_lookup)
    if do_create_atlas_coloring:
        create_atlas_coloring(subject, atlas, n_jobs)
    save_labels_from_vertices_lookup(
        subject, atlas, subjects_dir, MMVT_DIR, overwrite_labels=overwrite_labels,
        lookup=vertices_labels_lookup, n_jobs=n_jobs)
    labels_to_annot(subject, subjects_dir, atlas, overwrite=overwrite_annot, fix_unknown=False, n_jobs=n_jobs)
    return utils.atlas_exist(subject, atlas, subjects_dir)


def calc_subject_vertices_labels_lookup_from_template(
        subject, template_brain, atlas, subjects_dir='', overwrite=False):
    # from scipy.spatial.distance import cdist
    # max_upper_limit = 4
    # check_distances = False
    filter_unknown = True

    output_fname = op.join(MMVT_DIR, subject, '{}_vertices_labels_lookup.pkl'.format(atlas))
    if op.isfile(output_fname) and not overwrite:
        subject_vertices_labels_lookup = utils.load(output_fname)
        if len(subject_vertices_labels_lookup.get('rh', {})) > 0 and \
                len(subject_vertices_labels_lookup.get('lh', {})) > 0:
            return subject_vertices_labels_lookup
    template_vertices_labels_lookup = create_vertices_labels_lookup(template_brain, atlas)
    if not op.isdir(subjects_dir):
        for subjects_dir in [MMVT_DIR, SUBJECTS_DIR]:
            morph_maps_fol = op.join(subjects_dir, 'morph_maps')
            if op.isfile(op.join(morph_maps_fol, '{}-{}-morph.fif'.format(subject, template_brain))) and \
                    op.isfile(op.join(morph_maps_fol, '{}-{}-morph.fif'.format(template_brain, subject))):
                break
    # left_map, right_map : sparse matrix, subject verts x template verts
    if overwrite:
        morph_maps = glob.glob(op.join(SUBJECTS_DIR, 'morph-maps', '*{}*'.format(subject)))
        for morph_map_fname in morph_maps:
            utils.delete_file(morph_map_fname)
    morph_maps = mne.read_morph_map(template_brain, subject, subjects_dir=subjects_dir)

    subject_vertices_labels_lookup = defaultdict(dict)
    vertices_num = []
    for hemi_ind, hemi in enumerate(['lh', 'rh']):
        subject_vertices, _ = read_pial(subject, hemi, subjects_dir)
        template_vertices, _ = read_pial(template_brain, hemi, subjects_dir)

        if len(subject_vertices) != morph_maps[hemi_ind].shape[0]:
            raise Exception('Wrong number of vertices!')
        if not (len(template_vertices) == morph_maps[hemi_ind].shape[1] ==
                len(template_vertices_labels_lookup[hemi].keys())):
            raise Exception('Wrong number of vertices in the morphing map!')

        for subject_vert in range(len(subject_vertices)):
            template_verts_inds = morph_maps[hemi_ind][subject_vert].nonzero()[1]
            verts_labels = [
                template_vertices_labels_lookup[hemi][template_vert] for template_vert in template_verts_inds]
            # if len(template_verts_inds) > 1 and check_distances:
            #     dists = np.max(cdist(template_vertices[template_verts_inds], template_vertices[template_verts_inds]))
            #     if np.max(dists) > max_upper_limit:
            #         raise Exception('dist to high! {} max_dists {}'.format(verts_labels, dists))
            # filter unknown
            if filter_unknown:
                verts_labels = [l for l in verts_labels if 'unknown' not in l]
            vertices_num.append(len(verts_labels))
            if len(verts_labels) == 0:
                verts_label = 'unknown-{}'.format(hemi)
            elif len(verts_labels) == 1:
                verts_label = verts_labels[0]
            else:
                verts_label = Counter(verts_labels).most_common()[0][0]
            if verts_label.endswith('_{}'.format(hemi)):
                verts_label = '{}-{}'.format(verts_label[:-3], hemi)
            subject_vertices_labels_lookup[hemi][subject_vert] = verts_label
    utils.save(subject_vertices_labels_lookup, output_fname)
    return subject_vertices_labels_lookup


def calc_subject_to_subject_vertices_lookup(subject_from, subject_to, overwrite=False):
    output_fname = op.join(MMVT_DIR, subject_to, 'vertices_lookup_from_{}.pkl'.format(subject_from))
    if op.isfile(output_fname) and not overwrite:
        return utils.load(output_fname)
    for morph_maps_root in [MMVT_DIR, SUBJECTS_DIR]:
        morph_maps_fol = op.join(morph_maps_root, 'morph_maps')
        if op.isfile(op.join(morph_maps_fol, '{}-{}-morph.fif'.format(subject_from, subject_to))) and \
                op.isfile(op.join(morph_maps_fol, '{}-{}-morph.fif'.format(subject_to, subject_from))):
            break
    # left_map, right_map : sparse matrix, subject verts x template verts
    morph_maps = mne.read_morph_map(subject_from, subject_to, subjects_dir=morph_maps_root)

    subject_vertices_lookup = defaultdict(dict)
    for hemi_ind, hemi in enumerate(['lh', 'rh']):
        subject_from_vertices, _ = read_pial(subject_from, hemi)
        subject_to_vertices, _ = read_pial(subject_to, hemi)

        if len(subject_from_vertices) != morph_maps[hemi_ind].shape[1]:
            raise Exception('Wrong number of vertices!')
        if len(subject_to_vertices) != morph_maps[hemi_ind].shape[0]:
            raise Exception('Wrong number of vertices in the morphing map!')

        for subject_vert in range(len(subject_from_vertices)):
            subject_vertices_lookup[hemi][subject_vert] = morph_maps[hemi_ind][subject_vert].nonzero()[1]
    utils.save(subject_vertices_lookup, output_fname)
    return subject_vertices_lookup


def read_pial(subject, hemi, subjects_dir='', mmvt_dir=''):
    if not op.isdir(subjects_dir):
        subjects_dir = SUBJECTS_DIR
    if not op.isdir(mmvt_dir):
        mmvt_dir = MMVT_DIR
    from src.utils import geometry_utils as gu
    if utils.both_hemi_files_exist(op.join(subjects_dir, subject, 'surf', '{hemi}.pial')):
        # return nib.freesurfer.read_geometry(op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi)))
        return gu.read_surface(op.join(subjects_dir, subject, 'surf', '{}.pial'.format(hemi)))
    elif utils.both_hemi_files_exist(op.join(mmvt_dir, subject, 'surf', '{hemi}.pial.ply')):
        return utils.read_pial(subject, mmvt_dir, hemi)
    elif utils.both_hemi_files_exist(op.join(subjects_dir, subject, 'surf', '{hemi}.pial.ply')):
        return utils.read_pial(subject, subjects_dir, hemi)
    else:
        raise Exception('Can\'t find {} pial surface!'.format(subject))


def calc_labels_centroids(labels_hemi, hemis_verts):
    centroids = {}
    for hemi in HEMIS:
        centroids[hemi] = np.zeros((len(labels_hemi[hemi]), 3))
        for ind, label in enumerate(labels_hemi[hemi]):
            coo = hemis_verts[label.hemi][label.vertices]
            centroids[label.hemi][ind, :] = np.mean(coo, axis=0)
    return centroids


def backup_annotation_files(subject, subjects_dic, aparc_name, backup_str='backup'):
    # Backup annotation files
    for hemi in HEMIS:
        annot_fname = op.join(subjects_dic, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name))
        if op.isfile(annot_fname):
            utils.copy_file(op.join(subjects_dic, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name)),
                            op.join(subjects_dic, subject, 'label', '{}.{}.{}.annot'.format(hemi, aparc_name, backup_str)),)


def get_atlas_labels_names(subject, atlas, subjects_dir, delim='-', pos='end', return_flat_labels_list=False, include_unknown=False,
                           include_corpuscallosum=False, n_jobs=1):
    annot_fname_hemi = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    labels_names_hemis = dict(lh=[], rh=[])
    all_labels = []
    if utils.both_hemi_files_exist(annot_fname_hemi):
        for hemi in ['rh', 'lh']:
            annot_fname = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
            _, _, labels_names = mne.label._read_annot(annot_fname)
            labels_names = fix_labels_names(labels_names, hemi, delim, pos)
            all_labels.extend(labels_names)
            labels_names_hemis[hemi] = labels_names
    else:
        all_labels = read_labels_parallel(subject, subjects_dir, atlas, labels_fol='' , n_jobs=n_jobs)
        for label in all_labels:
            labels_names_hemis[label.hemi].append(label.name)
    if len(labels_names_hemis['rh']) == 0 or len(labels_names_hemis['lh']) == 0:
        raise Exception("Can't read {} labels for atlas {}".format(subject, atlas))
    if return_flat_labels_list:
        if not include_unknown:
            all_labels = [l for l in all_labels if 'unknown' not in l]
        if not include_corpuscallosum:
            all_labels = [l for l in all_labels if 'corpuscallosum' not in l]
        return all_labels
    else:
        if not include_unknown:
            for hemi in HEMIS:
                labels_names_hemis[hemi] = [l for l in labels_names_hemis[hemi] if 'unknown' not in l]
        if not include_corpuscallosum:
            for hemi in HEMIS:
                labels_names_hemis[hemi] = [l for l in labels_names_hemis[hemi] if 'corpuscallosum' not in l]
        return labels_names_hemis


def fix_labels_names(labels_names, hemi, delim='-', pos='end'):
    fixed_labels_names = []
    for label_name in labels_names:
        if isinstance(label_name, bytes):
            label_name = label_name.decode('utf-8')
        if not '{}-'.format(hemi) in label_name or \
            not '{}.'.format(hemi) in label_name or \
            not '-{}'.format(hemi) in label_name or \
            not '.{}'.format(hemi) in label_name:
                if pos == 'end':
                    label_name = '{}{}{}'.format(label_name, delim, hemi)
                elif pos == 'start':
                    label_name = '{}{}{}'.format(hemi, delim, label_name)
                else:
                    raise Exception("pos can be 'end' or 'start'")
        fixed_labels_names.append(label_name)
    return fixed_labels_names


def get_hemi_delim_and_pos(label_name):
    delim, pos, label, label_hemi = '', '', label_name, ''
    for hemi in ['rh', 'lh']:
        if label_name == hemi:
            delim, pos, label = '', '', ''
            label_hemi = hemi
            break
        if label_name.startswith('{}-'.format(hemi)):
            delim, pos, label = '-', 'start', label_name[3:]
            label_hemi = hemi
            break
        if label_name.startswith('{}_'.format(hemi)):
            delim, pos, label = '_', 'start', label_name[3:]
            label_hemi = hemi
            break
        if label_name.startswith('{}.'.format(hemi)):
            delim, pos, label = '.', 'start', label_name[3:]
            label_hemi = hemi
            break
        if label_name.endswith('-{}'.format(hemi)):
            delim, pos, label = '-', 'end', label_name[:-3]
            label_hemi = hemi
            break
        if label_name.endswith('_{}'.format(hemi)):
            delim, pos, label = '_', 'end', label_name[:-3]
            label_hemi = hemi
            break
        if label_name.endswith('.{}'.format(hemi)):
            label_hemi = hemi
            delim, pos, label = '.', 'end', label_name[:-3]
            break
        if '_{}'.format(hemi) in label_name:
            label_hemi = hemi
            delim, pos, label = '_', 'middle_start', label_name.replace('_{}'.format(hemi), '')
            break
        if '{}_'.format(hemi) in label_name:
            label_hemi = hemi
            delim, pos, label = '_', 'middle_end', label_name.replace('{}_'.format(hemi), '')
            break
        if '.{}'.format(hemi) in label_name:
            label_hemi = hemi
            delim, pos, label = '.', 'middle_start', label_name.replace('.{}'.format(hemi), '')
            break
        if '{}.'.format(hemi) in label_name:
            label_hemi = hemi
            delim, pos, label = '.', 'middle_end', label_name.replace('{}.'.format(hemi), '')
            break
    return delim, pos, label, label_hemi


def get_label_hemi(label_name):
    _, _, _, hemi = get_hemi_delim_and_pos(label_name)
    return hemi


def get_label_hemi_invariant_name(label_name):
    if isinstance(label_name, mne.Label):
        label_name = label_name.name
    _, _, label_inv_name, _ = get_hemi_delim_and_pos(label_name)
    while label_inv_name != label_name:
        label_name = label_inv_name
        _, _, label_inv_name, _ = get_hemi_delim_and_pos(label_name)
    return label_inv_name


def remove_duplicate_hemis(label_name):
    delim, pos, label, hemi = get_hemi_delim_and_pos(label_name)
    while label != label_name:
        label_name = label
        _, _, label, _ = get_hemi_delim_and_pos(label_name)
    res = build_label_name(delim, pos, label, hemi)
    return res


def get_other_hemi(hemi):
    return 'rh' if hemi == 'lh' else 'lh'


def get_other_hemi_label_name(label_name):
    delim, pos, label, hemi = get_hemi_delim_and_pos(label_name)
    other_hemi = get_other_hemi(hemi)
    if pos == 'middle_start':
        res_label_name = label_name.replace('{}{}'.format(delim, hemi), '{}{}'.format(delim, other_hemi))
    elif pos == 'middle_end':
        res_label_name = label_name.replace('{}{}'.format(hemi, delim), '{}{}'.format(other_hemi, delim))
    else:
        res_label_name = build_label_name(delim, pos, label, other_hemi)
    return res_label_name


def get_template_hemi_label_name(label_name, wild_char=False):
    delim, pos, label, hemi = get_hemi_delim_and_pos(label_name)
    hemi_temp = '?h' if wild_char else '{hemi}'
    res_label_name = build_label_name(delim, pos, label, hemi_temp)
    return res_label_name


def build_label_name(delim, pos, label, hemi):
    if pos == 'end':
        return '{}{}{}'.format(label, delim, hemi)
    elif pos == 'start':
        return '{}{}{}'.format(hemi, delim, label)


def get_hemi_from_name(label_name):
    _, _, _, hemi = get_hemi_delim_and_pos(str(label_name))
    return hemi


def find_hemi_from_full_fname(fname):
    folder = utils.namebase(fname)
    hemi = get_hemi_from_name(folder)
    while hemi == '' and folder != '':
        fname = utils.get_parent_fol(fname)
        folder = fname.split(op.sep)[-1]
        hemi = get_hemi_from_name(folder)
    return hemi


def get_labels_num(subject, subjects_dir, atlas, hemi='both'):
    from mne.label import _read_annot
    annot_fnames = get_annot_fnames(subject, subjects_dir, atlas, hemi)
    return np.concatenate([_read_annot(annot_fname)[2] for annot_fname in annot_fnames]).shape[0]


def get_labels_names(subject, subjects_dir, atlas, hemi='both'):
    from mne.label import _read_annot
    annot_fnames = get_annot_fnames(subject, subjects_dir, atlas, hemi)
    hemis = get_hemis(hemi)
    labels = []
    for annot_fname, hemi in zip(annot_fnames, hemis):
        labels_names = _read_annot(annot_fname)[2]
        labels_names = fix_labels_names(labels_names, hemi)
        labels.extend(labels_names)
    return labels


def get_hemis(hemi):
    return HEMIS if hemi == 'both' else [hemi]


def get_annot_fnames(subject, subjects_dir, atlas, hemi='both'):
    from mne.label import _get_annot_fname
    annot_fnames, hemis = _get_annot_fname(None, subject, hemi, atlas, subjects_dir)
    return annot_fnames


@functools.lru_cache(maxsize=None)
def read_labels(subject, subjects_dir, atlas, try_first_from_annotation=True, only_names=False,
                output_fname='', exclude=None, rh_then_lh=False, lh_then_rh=False, sorted_according_to_annot_file=False,
                hemi='both', surf_name='pial', labels_fol='', read_only_from_annot=False, n_jobs=1):
    try:
        labels = []
        # Fix for supporting both FS5.3 and FS6
        if atlas == 'aparc.DKTatlas' and not op.isfile(op.join(
                SUBJECTS_DIR, subject, 'label', 'rh.aparc.DKTatlas.annot')) and op.isfile(
                op.join(SUBJECTS_DIR, subject, 'label', 'rh.aparc.DKTatlas40.annot')):
            atlas = 'aparc.DKTatlas40'
        if try_first_from_annotation:
            try:
                labels = mne.read_labels_from_annot(
                    subject, atlas, subjects_dir=subjects_dir, surf_name=surf_name, hemi=hemi)
            except:
                # print(traceback.format_exc())
                print("read_labels_from_annot failed! subject {} atlas {} surf name {} hemi {}.".format(
                    subject, atlas, surf_name, hemi))
                utils.print_last_error_line()
                print('Trying to read labels files')
                if not read_only_from_annot:
                    labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol == '' else labels_fol
                    labels = read_labels_parallel(subject, subjects_dir, atlas, hemi, labels_fol=labels_fol, n_jobs=n_jobs)
        else:
            if not read_only_from_annot:
                labels = read_labels_parallel(
                    subject, subjects_dir, atlas, hemi=hemi, labels_fol=labels_fol, n_jobs=n_jobs)
        if len(labels) == 0:
            if fu.is_fs_atlas(atlas):
                annotations_exist = fu.create_annotation_file(
                    subject, atlas, subjects_dir=SUBJECTS_DIR, freesurfer_home=FREESURFER_HOME,
                    overwrite_annot_file=True)
                if not annotations_exist:
                    print("Can't recreate {}!".format(atlas))
            else:
                print('Can\'t find any labels for {} {}!'.format(hemi, atlas))
                return []
            labels = mne.read_labels_from_annot(
                subject, atlas, subjects_dir=subjects_dir, surf_name=surf_name, hemi=hemi)
            if len(labels) == 0:
                if not annotations_exist:
                    raise Exception("Can't read the {} labels!".format(atlas))
        if exclude is not None:
            labels = [l for l in labels if not np.any([e in l.name for e in exclude])]
        if rh_then_lh or lh_then_rh:
            rh_labels = [l for l in labels if l.hemi == 'rh']
            lh_labels = [l for l in labels if l.hemi == 'lh']
            labels = rh_labels + lh_labels if rh_then_lh else lh_labels + rh_labels
        if sorted_according_to_annot_file:
            annot_labels = get_atlas_labels_names(
                subject, atlas, subjects_dir, return_flat_labels_list=True,
                include_corpuscallosum=True, include_unknown=True)
            try:
                labels.sort(key=lambda x: annot_labels.index(x.name))
            except ValueError:
                print("Can't sort labels according to the annot file")
                print(traceback.format_exc())
        if output_fname != '':
            with open(output_fname, 'w') as output_file:
                for label in labels:
                    output_file.write('{}\n'.format(label.name))
        if only_names:
            labels = [l.name for l in labels]
        return labels
    except:
        print(traceback.format_exc())
        return []


def read_labels_files(subject, labels_fol, subjects_dir='', n_jobs=4):
    return read_labels_parallel(subject, labels_fol=labels_fol, n_jobs=n_jobs)


def read_labels_parallel(subject, subjects_dir='', atlas='', hemi='', labels_fol='', n_jobs=1):
    try:
        if subjects_dir == '':
            subjects_dir = SUBJECTS_DIR
        labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol == '' else labels_fol
        if not op.isdir(labels_fol):
            labels_fol = op.join(MMVT_DIR, subject, 'labels', atlas)
        if not op.isdir(labels_fol):
            print('Can\'t find the atlas ({}) folder!'.format(atlas))
            return []
        if hemi in ['rh', 'lh']:
            labels_files = glob.glob(op.join(labels_fol, '*{}.label'.format(hemi)))
            labels_files.extend(glob.glob(op.join(labels_fol, '{}.*label'.format(hemi))))
        else:
            labels_files = glob.glob(op.join(labels_fol, '*.label'))
        files_chunks = utils.chunks(labels_files, len(labels_files) / n_jobs)
        results = utils.run_parallel(_read_labels_parallel, files_chunks, njobs=n_jobs)
        labels = []
        for labels_chunk in results:
            labels.extend(labels_chunk)
        for l in labels:
            l.subject = subject
        return labels
    except:
        print(traceback.format_exc())
        return []


def _read_labels_parallel(files_chunk):
    labels = []
    for label_fname in files_chunk:
        delim, pos, label_name, label_hemi = mu.get_hemi_delim_and_pos(utils.namebase(label_fname))
        label_name = get_label_hemi_invariant_name(label_name)
        label = mne.read_label(label_fname)
        label.name = mu.build_label_name(delim, pos, label_name, label_hemi)
        labels.append(label)
    return labels


# def read_hemi_labels(subject, subjects_dir, atlas, hemi, surf_name='pial', labels_fol=''):
#     # todo: replace with labels utils read labels function
#     labels_fol = op.join(subjects_dir, subject, 'label', atlas) if labels_fol=='' else labels_fol
#     annot_fname_template = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
#     if utils.both_hemi_files_exist(annot_fname_template):
#         try:
#             labels = mne.read_labels_from_annot(subject, atlas, hemi, surf_name)
#         except:
#             labels = read_labels_from_annots(atlas, hemi)
#         if len(labels) == 0:
#             raise Exception('No labels were found in the {} annot file!'.format(annot_fname_template))
#     else:
#         labels = []
#         for label_file in glob.glob(op.join(labels_fol, '*{}.label'.format(hemi))):
#             label = mne.read_label(label_file)
#             labels.append(label)
#         if len(labels) == 0:
#             print('No labels were found in {}!'.format(labels_fol))
#             return []
#     return labels


def calc_center_of_mass(labels, ret_mat=False, find_vertice=False, verts_pos=None):
    if find_vertice:
        if labels[0].subject == '':
            raise Exception('find_vertice=True and no subject in labels!')
        if verts_pos is None:
            verts_pos = utils.load_surf(labels[0].subject, MMVT_DIR, SUBJECTS_DIR)
    center_of_mass = np.zeros((len(labels), 3)) if ret_mat else {}
    verts = np.zeros((len(labels)), dtype=int) if ret_mat else {}
    for ind, label in enumerate(labels):
        if find_vertice:
            vert = label.center_of_mass(restrict_vertices=True, subjects_dir=SUBJECTS_DIR)
            pos = verts_pos[label.hemi][vert] / 1000
        else:
            pos = np.mean(label.pos, 0)
            vert = -1
        if ret_mat:
            center_of_mass[ind] = pos
            verts[ind] = vert
        else:
            center_of_mass[label.name] = pos
            verts[label.name] = vert
    if find_vertice:
        return center_of_mass, verts
    else:
        return center_of_mass


def label_is_excluded(label_name, compiled_excludes):
    return not compiled_excludes.search(label_name) is None


def label_name(l):
    return l if isinstance(l, str) else l.name


def remove_exclude_labels(labels, excludes=()):
    _label_is_excluded = partial(label_is_excluded, compiled_excludes=re.compile('|'.join(excludes)))
    labels_tup = [(l, ind) for ind, l in enumerate(labels) if not _label_is_excluded(label_name(l))]
    labels = [t[0] for t in labels_tup]
    indices = [t[1] for t in labels_tup]
    return labels, indices


def remove_exclude_labels_names(labels_names, excludes=()):
    _label_is_excluded = partial(label_is_excluded, compiled_excludes=re.compile('|'.join(excludes)))
    return [l for l in labels_names if not _label_is_excluded(label_name(l))]


def remove_exclude_labels_and_data(labels_names, labels_data, excludes=()):
    if len(excludes) > 0:
        org_labels_names = labels_names
        labels_names, indices = remove_exclude_labels(labels_names, excludes)
        remove_indices = list(set(range(len(org_labels_names))) - set(indices))
        if len(remove_indices) > len(excludes):
            raise Exception('Error in removing excludes')
        if len(remove_indices) > 0:
            labels_data = np.delete(labels_data, remove_indices, 0)
    if len(labels_names) != labels_data.shape[0]:
        raise Exception(
            'Error in remove_exclude_labels_and_data! len(labels_names) {} != labels_data.shape {}'.format(
                labels_names, labels_data.shape[0]))
    return labels_names, labels_data


def calc_time_series_per_label(x, labels, measure, excludes=(),
                               figures_dir='', do_plot=False, do_plot_all_vertices=False):
    import sklearn.decomposition as deco
    if do_plot:
        import matplotlib.pyplot as plt

    labels, _ = remove_exclude_labels(labels, excludes)
    if measure.startswith('pca'):
        comps_num = 1 if '_' not in measure else int(measure.split('_')[1])
        labels_data = np.zeros((len(labels), x.shape[-1], comps_num))
    else:
        labels_data = np.zeros((len(labels), x.shape[-1]))
    labels_names = []
    if do_plot_all_vertices:
        all_vertices_plots_dir = op.join(figures_dir, 'all_vertices')
        utils.make_dir(all_vertices_plots_dir)
    if do_plot:
        measure_plots_dir = op.join(figures_dir, measure)
        utils.make_dir(measure_plots_dir)
    for ind, label in enumerate(labels):
        if measure == 'mean':
            labels_data[ind, :] = np.mean(x[label.vertices, 0, 0, :], 0)
        elif measure.startswith('pca'):
            print(label)
            _x = x[label.vertices, 0, 0, :].T
            remove_cols = np.where(np.all(_x == np.mean(_x, 0), 0))[0]
            _x = np.delete(_x, remove_cols, 1)
            _x = (_x - np.mean(_x, 0)) / np.std(_x, 0)
            comps = 1 if '_' not in measure else int(measure.split('_')[1])
            pca = deco.PCA(comps)
            x_r = pca.fit(_x).transform(_x)
            # if x_r.shape[1] == 1:
            labels_data[ind, :] = x_r
            # else:
            #     labels_data[ind, :] = x_r
        elif measure == 'cv': #''coef_of_variation':
            label_mean = np.mean(x[label.vertices, 0, 0, :], 0)
            label_std = np.std(x[label.vertices, 0, 0, :], 0)
            labels_data[ind, :] = label_std / label_mean
        labels_names.append(label.name)
        if do_plot_all_vertices:
            plt.figure()
            plt.plot(x[label.vertices, 0, 0, :].T)
            plt.savefig(op.join(all_vertices_plots_dir, '{}.jpg'.format(label.name)))
            plt.close()
        if do_plot:
            plt.figure()
            plt.plot(labels_data[ind, :])
            plt.savefig(op.join(measure_plots_dir, '{}_{}.jpg'.format(measure, label.name)))
            plt.close()

    return labels_data, labels_names


def morph_labels(morph_from_subject, morph_to_subject, atlas, hemi, n_jobs=1):
    labels_fol = op.join(SUBJECTS_DIR, morph_to_subject, 'label')
    labels_fname = op.join(labels_fol, '{}.{}.pkl'.format(hemi, atlas,morph_from_subject))
    annot_file = op.join(SUBJECTS_DIR, morph_from_subject, 'label', '{}.{}.annot'.format(hemi, atlas))
    if not op.isfile(annot_file):
        print("Can't find the annot file in {}!".format(annot_file))
        return []
    if not op.isfile(labels_fname):
        labels = mne.read_labels_from_annot(morph_from_subject, atlas, subjects_dir=SUBJECTS_DIR, hemi=hemi)
        if morph_from_subject != morph_to_subject:
            morphed_labels = []
            for label in labels:
                label.values.fill(1.0)
                morphed_label = label.morph(morph_from_subject, morph_to_subject, 5, None, SUBJECTS_DIR, n_jobs)
                morphed_labels.append(morphed_label)
            labels = morphed_labels
        utils.save(labels, labels_fname)
    else:
        labels = utils.load(labels_fname)
    return labels


def create_atlas_coloring(subject, atlas, n_jobs=-1):
    ret = False
    coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
    utils.make_dir(coloring_dir)
    coloring_fname = op.join(coloring_dir, 'labels_{}_coloring.csv'.format(atlas))
    coloring_names_fname = op.join(coloring_dir, 'labels_{}_colors_names.txt'.format(atlas))
    try:
        labels = read_labels(subject, SUBJECTS_DIR, atlas, n_jobs=n_jobs)
        if len(labels) == 0:
            print('create_atlas_coloring: No labels for {}!'.format(atlas))
        colors_rgb_and_names = cu.get_distinct_colors_and_names()
        labels_colors_rgb, labels_colors_names = {}, {}
        for label in labels:
            label_inv_name = get_label_hemi_invariant_name(label.name)
            if label_inv_name not in labels_colors_rgb:
                labels_colors_rgb[label_inv_name], labels_colors_names[label_inv_name] = next(colors_rgb_and_names)
        print('Writing to {} and {}'.format(coloring_fname, coloring_names_fname))
        with open(coloring_fname, 'w') as colors_file, open(coloring_names_fname, 'w') as col_names_file:
            if atlas != '':
                colors_file.write('atlas={}\n'.format(atlas))
            for label in labels:
                label_inv_name = get_label_hemi_invariant_name(label.name)
                color_rgb = labels_colors_rgb[label_inv_name]
                color_name = labels_colors_names[label_inv_name]
                colors_file.write('{},{},{},{}\n'.format(label.name, *color_rgb))
                col_names_file.write('{},{}\n'.format(label.name, color_name))
        ret = op.isfile(coloring_fname)
    except:
        print('Error in save_labels_coloring!')
        print(traceback.format_exc())
    return ret


def create_labels_coloring(subject, labels_names, labels_values, coloring_name, norm_percs=(3, 99),
                           norm_by_percentile=True, colors_map='jet'):
    coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
    utils.make_dir(coloring_dir)
    coloring_fname = op.join(coloring_dir, '{}.csv'.format(coloring_name))
    ret = False
    try:
        labels_colors = utils.arr_to_colors(
            labels_values, norm_percs=norm_percs, colors_map=colors_map) # norm_by_percentile=norm_by_percentile
        print('Saving coloring to {}'.format(coloring_fname))
        with open(coloring_fname, 'w') as colors_file:
            for label_name, label_color, label_value in zip(labels_names, labels_colors, labels_values):
                colors_file.write('{},{},{},{},{}\n'.format(label_name, *label_color[:3], label_value))
        ret = op.isfile(coloring_fname)
    except:
        print('Error in create_labels_coloring!')
        print(traceback.format_exc())
    return ret


def join_labels(new_name, labels):
    from functools import reduce
    import operator
    labels = list(labels)
    new_label = reduce(operator.add, labels[1:], labels[0])
    new_label.name = new_name
    return new_label


def get_lh_rh_indices(labels):
    get_hemi_delim_and_pos(labels[0])
    indices = {hemi:[ind for ind, l in enumerate(labels) if get_label_hemi(label_name(l))==hemi] for hemi in utils.HEMIS}
    labels_arr = np.array(labels)
    if sum([len(labels_arr[indices[hemi]]) for hemi in utils.HEMIS]) != len(labels):
        raise Exception('len(rh_labels) ({}) + len(lh_labels) ({}) != len(labels) ({})'.format(
            len(labels_arr[indices['rh']]), len(labels_arr[indices['lh']]), len(labels)))
    return indices


def grow_label(subject, vertice_indice, hemi, new_label_name, new_label_r=5, n_jobs=6, labels_fol='', overwrite=False):
    if labels_fol == '':
        labels_fol = op.join(MMVT_DIR, subject, 'labels')
    new_label_fname = op.join(labels_fol, '{}.label'.format(new_label_name))
    if not overwrite and op.isfile(new_label_fname):
        return new_label_fname
    new_label = mne.grow_labels(subject, vertice_indice, new_label_r, 0 if hemi == 'lh' else 1, SUBJECTS_DIR,
                                n_jobs, names=new_label_name, surface='pial')[0]
    utils.make_dir(labels_fol)
    new_label.save(new_label_fname)
    return new_label_fname


def find_clusters_overlapped_labeles(subject, clusters, data, atlas, hemi, verts, labels=None,
        min_cluster_max=0, min_cluster_size=0, clusters_label='', abs_max=True, n_jobs=6):
    cluster_labels = []
    if not op.isfile(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi))):
        from src.utils import freesurfer_utils as fu
        verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
        fu.write_surf(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi)), verts, faces)
    if labels is None:
        labels = read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi, n_jobs=n_jobs)
    if len(labels) == 0:
        print('No labels!')
        return None
    output_stc_data = np.ones((data.shape[0], 1)) * -1
    for cluster in clusters:
        x = data[cluster]
        if abs_max:
            cluster_max = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
            max_vert_ind = np.argmin(x) if abs(np.min(x)) > abs(np.max(x)) else np.argmax(x)
            if abs(cluster_max) < min_cluster_max or len(cluster) < min_cluster_size:
                continue
        else:
            cluster_max = np.max(x)
            max_vert_ind = np.argmax(x)
            if cluster_max < min_cluster_max or len(cluster) < min_cluster_size:
                continue
        output_stc_data[cluster, 0] = x
        max_vert = cluster[max_vert_ind]
        inter_labels, inter_labels_tups = [], []
        for label in labels:
            overlapped_vertices = np.intersect1d(cluster, label.vertices)
            if len(overlapped_vertices) > 0:
                if 'unknown' not in label.name:
                    inter_labels_tups.append((len(overlapped_vertices), label.name))
                    # inter_labels.append(dict(name=label.name, num=len(overlapped_vertices)))
        inter_labels_tups = sorted(inter_labels_tups)[::-1]
        for inter_labels_tup in inter_labels_tups:
            inter_labels.append(dict(name=inter_labels_tup[1], num=inter_labels_tup[0]))
        if len(inter_labels) > 0 and (clusters_label in inter_labels[0]['name'] or clusters_label == ''):
            print('Cluster max: {:.2f}, size: {}, intersected: {}'.format(
                cluster_max, len(cluster), ','.join(['{} {}'.format(t[1], t[0]) for t in inter_labels_tups])))
            # max_inter = max([(il['num'], il['name']) for il in inter_labels])
            cluster_labels.append(dict(vertices=cluster, intersects=inter_labels, name=inter_labels[0]['name'],
                coordinates=verts[cluster], max=cluster_max, hemi=hemi, size=len(cluster), max_vert=max_vert))
        # else:
        #     print('No intersected labels!')
    return cluster_labels, output_stc_data


# if __name__ == '__main__':
#     morph_annot(subject, template_brain, atlas, overwrite_subject_vertices_labels_lookup=False,
#                 overwrite_labels=False, overwrite_annot=False, n_jobs=4)
#
