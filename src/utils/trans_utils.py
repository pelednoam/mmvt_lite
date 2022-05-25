import os.path as op
import numpy as np
import nibabel as nib
import time
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREESURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')


# Check this out:
# https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/registration/Wu2017_RegistrationFusion


def get_vox2ras(fname):
    output = utils.run_script('mri_info --vox2ras {}'.format(fname))
    return read_transform_matrix_from_output(output)


def get_vox2ras_tkr(fname):
    output = utils.run_script('mri_info --vox2ras-tkr {}'.format(fname))
    return read_transform_matrix_from_output(output)


def ras_to_tkr_ras(fname):
    ras2vox = np.linalg.inv(get_vox2ras(fname))
    vox2tkras = get_vox2ras_tkr(fname)
    return np.dot(ras2vox, vox2tkras)


def read_transform_matrix_from_output(output):
    import re
    try:
        str_mat = output.decode('ascii').split('\n')
    except:
        str_mat = output.split('\n')
    for i in range(len(str_mat)):
        str_mat[i] = re.findall(r'[+-]?[0-9.]+', str_mat[i])
    del str_mat[-1]
    return np.array(str_mat).astype(float)


def apply_trans(trans, points):
    return np.array([np.dot(trans, np.append(p, 1))[:3] for p in points])


def get_talxfm(subject, subjects_dir, return_trans_obj=False):
    trans = mne.source_space._read_talxfm(subject, subjects_dir, 'nibabel')
    if not return_trans_obj:
        trans = trans['trans']
    return trans


def tkras_to_mni(points, subject, subjects_dir):
    # https://mail.nmr.mgh.harvard.edu/pipermail/freesurfer/2012-June/024293.html
    # MNI305RAS = TalXFM * orig_vox2ras * inv(orig_vox2tkras) * [tkrR tkrA tkrS 1]
    tal_xfm = get_talxfm(subject, subjects_dir)
    orig_vox2ras = get_vox2ras(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
    orig_vox2tkras = get_vox2ras_tkr(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
    trans = tal_xfm @ orig_vox2ras @ inv(orig_vox2tkras)
    points = apply_trans(trans, points)
    # points = apply_trans(tal_xfm, points)
    # points = apply_trans(orig_vox2ras, points)
    # points = apply_trans(np.linalg.inv(orig_vox2tkras), points)

    return points


def mni_to_tkras(points, subject, subjects_dir, tal_xfm=None, orig_vox2ras=None, orig_vox2tkras=None):
    # https://mail.nmr.mgh.harvard.edu/pipermail/freesurfer/2012-June/024293.html
    # MNI305RAS = TalXFM * orig_vox2ras * inv(orig_vox2tkras) * [tkrR tkrA tkrS 1]
    if tal_xfm is None:
        tal_xfm = get_talxfm(subject, subjects_dir)
    if orig_vox2ras is None:
        orig_vox2ras = get_vox2ras(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
    if orig_vox2tkras is None:
        orig_vox2tkras = get_vox2ras_tkr(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
    # Only in python 3.5:
    # trans = tal_xfm @ orig_vox2ras @ np.linalg.inv(orig_vox2tkras)
    # trans = tal_xfm.dot(orig_vox2ras).dot(np.linalg.inv(orig_vox2tkras))
    # trans = np.linalg.inv(trans)
    trans = inv(tal_xfm @ orig_vox2ras @ inv(orig_vox2tkras))
    # points = apply_trans(orig_vox2tkras, points)
    # points = apply_trans(np.linalg.inv(orig_vox2ras), points)
    # points = apply_trans(np.linalg.inv(tal_xfm), points)
    points = apply_trans(trans, points)
    return points


def inv(x):
    return np.linalg.inv(x)


def mni305_to_mni152_matrix():
    # http://freesurfer.net/fswiki/CoordinateSystems
    # The folowing matrix is V152*inv(T152)*R*T305*inv(V305), where V152 and V305 are the vox2ras matrices from the
    # 152 and 305 spaces, T152 and T305 are the tkregister-vox2ras matrices from the 152 and 305 spaces,
    # and R is from $FREESURFER_HOME/average/mni152.register.dat
    M = [[0.9975, - 0.0073, 0.0176, -0.0429],
         [0.0146, 1.0009, -0.0024, 1.5496],
         [-0.0130, -0.0093, 0.9971, 1.1840],
         [0, 0, 0, 1.0000]]
    return np.array(M)


def mni305_to_mni152(points):
    return apply_trans(mni305_to_mni152_matrix(), points)


# def mni152_mni305(points):
#     return apply_trans(np.linalg.inv(mni305_to_mni152_matrix()), points)


def mni152_mni305(voxels):
    mni152_header, mni305_header = get_mni152_and_305_headers()
    ras = apply_trans(mni152_header.get_vox2ras(), voxels)
    mni305 = np.rint(apply_trans(np.linalg.inv(mni305_header.get_vox2ras()), ras)).astype(int)
    return mni305


def mni305_mni152(voxels):
    mni152_header, mni305_header = get_mni152_and_305_headers()
    ras = apply_trans(np.linalg.inv(mni305_header.get_vox2ras()), voxels)
    mni152 = np.rint(apply_trans(np.linalg.inv(mni152_header.get_vox2ras()), ras)).astype(int)
    return mni152


def get_mni152_and_305_headers():
    nmi152_t1_fname = op.join(SUBJECTS_DIR, 'cvs_avg35_inMNI152', 'mri', 'T1.mgz')
    if not op.isfile(nmi152_t1_fname):
        nmi152_t1_fname = op.join(FREESURFER_HOME, 'subjects', 'cvs_avg35_inMNI152', 'mri', 'T1.mgz')
    if not op.isfile(nmi152_t1_fname):
        raise Exception('Can\'t find cvs_avg35_inMNI152 T1.mgz!')
    mni305_t1_fname = op.join(SUBJECTS_DIR, 'fsaverage',  'mri', 'T1.mgz')
    if not op.isfile(mni305_t1_fname):
        mni305_t1_fname = op.join(FREESURFER_HOME, 'subjects', 'fsaverage', 'mri', 'T1.mgz')
    if not op.isfile(mni305_t1_fname):
        raise Exception('Can\'t find fsaverage T1.mgz!')
    mni152_header = nib.load(nmi152_t1_fname).header
    mni305_header = nib.load(mni305_t1_fname).header
    return mni152_header, mni305_header


def tkras_to_vox(points, subject_orig_header=None, subject='', subjects_dir=''):
    if subject_orig_header is None:
        subject_orig_header = get_subject_mri_header(subject, subjects_dir)
    vox2ras_tkr = subject_orig_header.get_vox2ras_tkr()
    ras_tkr2vox = np.linalg.inv(vox2ras_tkr)
    vox = apply_trans(ras_tkr2vox, points)
    return vox


def vox_to_ras(points, subject_orig_header=None, subject='', subjects_dir=''):
    if subject_orig_header is None:
        subject_orig_header = get_subject_mri_header(subject, subjects_dir)
    vox2ras = subject_orig_header.get_vox2ras()
    ras = apply_trans(vox2ras, points)
    return ras


def get_subject_mri_header(subject, subjects_dir, image_name='T1.mgz'):
    image_fname = op.join(subjects_dir, subject, 'mri', image_name)
    if op.isfile(image_fname):
        d = nib.load(image_fname)# 'orig.mgz'))
        subject_orig_header = d.get_header()
    else:
        print("get_subject_mri_header: Can't find image! ({})".format(image_fname))
        subject_orig_header = None
    return subject_orig_header


def tal2mni(coords):
    """
    Python version of BrainMap's tal2icbm_other.m.
    This function converts coordinates from Talairach space to MNI
    space (normalized using templates other than those contained
    in SPM and FSL) using the tal2icbm transform developed and
    validated by Jack Lancaster at the Research Imaging Center in
    San Antonio, Texas.
    http://www3.interscience.wiley.com/cgi-bin/abstract/114104479/ABSTRACT
    FORMAT outpoints = tal2icbm_other(inpoints)
    Where inpoints is N by 3 or 3 by N matrix of coordinates
    (N being the number of points)
    ric.uthscsa.edu 3/14/07
    """
    coords = np.array(coords)
    if coords.ndim != 2:
        coords = np.reshape(coords, (1, 3))

    # Find which dimensions are of size 3
    shape = np.array(coords.shape)
    if all(shape == 3):
        print('Input is an ambiguous 3x3 matrix.\nAssuming coords are row vectors (Nx3).')
        use_dim = 1
    elif not any(shape == 3):
        raise AttributeError('Input must be an Nx3 or 3xN matrix.')
    else:
        use_dim = np.where(shape == 3)[0][0]

    # Transpose if necessary
    if use_dim == 1:
        coords = coords.transpose()

    # Transformation matrices, different for each software package
    icbm_other = np.array([[0.9357, 0.0029, -0.0072, -1.0423],
                           [-0.0065, 0.9396, -0.0726, -1.3940],
                           [0.0103, 0.0752, 0.8967, 3.6475],
                           [0.0000, 0.0000, 0.0000, 1.0000]])

    # Invert the transformation matrix
    icbm_other = np.linalg.inv(icbm_other)

    # Apply the transformation matrix
    coords = np.concatenate((coords, np.ones((1, coords.shape[1]))))
    coords = np.dot(icbm_other, coords)

    # Format the output, transpose if necessary
    out_coords = coords[:3, :]
    if use_dim == 1:
        out_coords = out_coords.transpose()

    return np.rint(out_coords).astype(int)


def mni2tal(coords):
    """
    Python version of BrainMap's icbm_other2tal.m.
    This function converts coordinates from MNI space (normalized using
    templates other than those contained in SPM and FSL) to Talairach space
    using the icbm2tal transform developed and validated by Jack Lancaster at
    the Research Imaging Center in San Antonio, Texas.
    http://www3.interscience.wiley.com/cgi-bin/abstract/114104479/ABSTRACT
    FORMAT outpoints = icbm_other2tal(inpoints)
    Where inpoints is N by 3 or 3 by N matrix of coordinates
    (N being the number of points)
    ric.uthscsa.edu 3/14/07
    """
    coords = np.array(coords)
    if coords.ndim != 2:
        coords = np.reshape(coords, (1, 3))

    # Find which dimensions are of size 3
    shape = np.array(coords.shape)
    if all(shape == 3):
        print('Input is an ambiguous 3x3 matrix.\nAssuming coords are row vectors (Nx3).')
        use_dim = 1
    elif not any(shape == 3):
        raise AttributeError('Input must be an Nx3 or 3xN matrix.')
    else:
        use_dim = np.where(shape == 3)[0][0]

    # Transpose if necessary
    if use_dim == 1:
        coords = coords.transpose()

    # Transformation matrices, different for each software package
    icbm_other = np.array([[0.9357, 0.0029, -0.0072, -1.0423],
                           [-0.0065, 0.9396, -0.0726, -1.3940],
                           [0.0103, 0.0752, 0.8967, 3.6475],
                           [0.0000, 0.0000, 0.0000, 1.0000]])

    # Apply the transformation matrix
    coords = np.concatenate((coords, np.ones((1, coords.shape[1]))))
    coords = np.dot(icbm_other, coords)

    # Format the output, transpose if necessary
    out_coords = coords[:3, :]
    if use_dim == 1:
        out_coords = out_coords.transpose()
    return np.rint(out_coords).astype(int)


def yale_get_driver(yale_app_url=''):
    try:
        from selenium import webdriver
    except:
        print('You should first install Selenium with Python!')
        print('https://selenium-python.readthedocs.io/installation.html')
        return False

    # Using Chrome to access web
    chrome_driver_fname = op.join(LINKS_DIR, 'chromedriver')
    if not op.isfile(chrome_driver_fname):
        print('Can\'t find chrome driver! It should be in {}'.format(chrome_driver_fname))
        print('ChromeDriver: https://sites.google.com/a/chromium.org/chromedriver/getting-started')
        return False

    driver = webdriver.Chrome(chrome_driver_fname)
    if yale_app_url == '':
        yale_app_url = 'http://sprout022.sprout.yale.edu/mni2tal/mni2tal.html'
    # Open the website
    driver.get(yale_app_url)
    time.sleep(0.3)
    return driver


def yale_tal2mni(points, driver=None, yale_app_url='', get_brodmann_areas=False):
    mni_points, brodmann_areas = [], []
    one_point = False
    if utils.is_int(points[0]) or utils.is_float(points[0]):
        one_point = True
        points = [points]
    if driver is None:
        driver = yale_get_driver(yale_app_url)
    for xyz in points:
        for val, element_name in zip(xyz, ['talx', 'taly', 'talz']):
            element_set_val(driver, element_name, val)
        time.sleep(0.1)
        driver.find_element_by_id('talgo').click()
        time.sleep(0.1)
        mni_points.append([int(element_get_val(driver, element_name)) for element_name in ['mnix', 'mniy', 'mniz']])
        if get_brodmann_areas:
            brodmann_areas.append(element_get_val(driver, 'baselectbox'))
    if one_point:
        mni_points = mni_points[0]
        if get_brodmann_areas:
            brodmann_areas = brodmann_areas[0]
    if get_brodmann_areas:
        return mni_points, brodmann_areas
    else:
        return mni_points


def element_get_val(driver, element_name):
    return driver.find_element_by_id(element_name).get_attribute('value')


def element_set_val(driver, element_name, val):
    driver.execute_script("document.getElementById('{}').value='{}'".format(element_name, str(val)))


if __name__ == '__main__':
    from src.utils import preproc_utils as pu
    SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
    subject = 'mg78'
    points = [[-17.85, -52.18, 42.08]]
    true_ras = [[-16.92, -44.43, 29.06]]
    true_vox = [[146, 86, 76]]
    # point = [-13.1962, -66.5584, 33.3018]

    h = get_subject_mri_header(subject, SUBJECTS_DIR)
    vox = tkras_to_vox(points, h)
    print('vox: {}'.format(vox))

    ras = vox_to_ras(vox, h)
    print('ras: {}'.format(ras))

    import mne
    from mne.source_space import vertex_to_mni, combine_transforms, Transform, apply_trans

    mni2 = vertex_to_mni(23633, 0, 'mg78', SUBJECTS_DIR)
    print('mni2: {}'.format(mni2))

    # Should be [[3, 5, 7], [2, 7, -1], [7, 6, 5]]
    print(yale_tal2mni([[2, 3, 9],[1, 4, 2], [6, 3, 7]]))
