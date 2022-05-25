# The setup suppose to run *before* installing python libs, so only python vanilla can be used here

import os
import os.path as op
import shutil
import traceback
from src.utils import setup_utils as utils
import glob

TITLE = 'MMVT Installation'
BLENDER_WIN_DIR = 'C:\Program Files\Blender Foundation\Blender'
EMPTY_SUBJECT_URL = 'https://www.dropbox.com/s/7wfi1rtfsknnlwg/empty_subject.blend?dl=1'


def copy_resources_files(mmvt_root_dir, overwrite=True, only_verbose=False):
    resource_dir = utils.get_resources_fol()
    utils.make_dir(op.join(op.join(mmvt_root_dir, 'color_maps')))
    files = ['aparc.DKTatlas40_groups.csv', 'atlas.csv', 'sub_cortical_codes.txt', 'FreeSurferColorLUT.txt',
             'high_level_atlas.csv']
    cm_files = glob.glob(op.join(resource_dir, 'color_maps', '*.npy'))
    all_files_exist = utils.all([op.isfile(op.join(mmvt_root_dir, file_name)) for file_name in files])
    all_cm_files_exist = utils.all([op.isfile(
        op.join(mmvt_root_dir, 'color_maps', '{}.npy'.format(utils.namebase(fname)))) for fname in cm_files])
    if all_files_exist and all_cm_files_exist and not overwrite:
        if only_verbose:
            print('All files exist!')
        return True
    if not all_cm_files_exist or overwrite:
        for color_map_file in glob.glob(op.join(resource_dir, 'color_maps', '*.npy')):
            new_file_name = op.join(mmvt_root_dir, 'color_maps', color_map_file.split(op.sep)[-1])
            # print('Copy {} to {}'.format(color_map_file, new_file_name))
            print('Coping {} to {}'.format(color_map_file, new_file_name))
            if not only_verbose:
                try:
                    shutil.copy(color_map_file, new_file_name)
                except:
                    print('Can\'t copy {} to {}!'.format(color_map_file, new_file_name))
    if not all_files_exist or overwrite:
        for file_name in files:
            print('Copying {} to {}'.format(op.join(resource_dir, file_name), op.join(mmvt_root_dir, file_name)))
            if not only_verbose:
                local_fname = op.join(resource_dir, file_name)
                if op.isfile(op.join(resource_dir, file_name)):
                    try:
                        shutil.copy(local_fname, op.join(mmvt_root_dir, file_name))
                    except:
                        print('Can\'t copy {}'.format(file_name))
                else:
                    print('{} is missing, please update your code from github (git pull)'.format(
                        op.join(resource_dir, file_name)))
    return utils.all([op.isfile(op.join(mmvt_root_dir, file_name)) for file_name in files])


def download_empty_subject(mmvt_root_dir, empty_subject_url, overwrite=False):
    import urllib.request
    empty_subject_fname = op.join(mmvt_root_dir, 'empty_subject.blend')
    if op.isfile(empty_subject_fname) and not overwrite:
        print('Empty subject is already exist')
        return True
    utils.delete_file(empty_subject_fname)
    urllib.request.urlretrieve(empty_subject_url, empty_subject_fname)
    if op.isfile(empty_subject_fname):
        print('Empty subject was download successfully!')


def install_deface():
    if utils.is_windows():
        return False
    deface_dir = utils.make_dir(op.join(get_mmvt_root_folder(), 'deface'))
    print('Installing mri_deface in {}'.format(deface_dir))
    output_files = [op.join(deface_dir, f) for f in ['mri_deface', 'talairach_mixed_with_skull.gca', 'face.gca']]
    if utils.all_files_exist(output_files):
        return True
    current_dir = os.getcwd()
    os.chdir(deface_dir)
    urls = {
        'mri_deface_linux': 'https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/mri_deface_linux',
        'mri_deface_osx': 'https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/mri_deface_osx',
        'tal': 'https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/talairach_mixed_with_skull.gca.gz',
        'face': 'https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/face.gca.gz'}
    defcae_url = urls['mri_deface_linux' if utils.is_linux() else 'mri_deface_osx']
    print('Downloading deface code from {}'.format(defcae_url))
    utils.download_file(defcae_url, op.join(deface_dir, 'mri_deface'))
    utils.download_file(urls['tal'], op.join(deface_dir, 'talairach_mixed_with_skull.gca.gz'))
    utils.download_file(urls['face'], op.join(deface_dir, 'face.gca.gz'))
    utils.run_script('chmod a+x mri_deface')
    utils.run_script('gunzip talairach_mixed_with_skull.gca.gz')
    utils.run_script('gunzip face.gca.gz')
    utils.delete_file('talairach_mixed_with_skull.gca.gz')
    utils.delete_file('face.gca.gz')
    os.chdir(current_dir)
    return utils.all_files_exist(output_files)


def create_links(links_fol_name='links', gui=True, default_folders=False, only_verbose=False,
                 links_file_name='links.csv', overwrite=True):
    links_fol = utils.get_links_dir(links_fol_name)
    if only_verbose:
        print('making links dir {}'.format(links_fol))
    else:
        utils.make_dir(links_fol)
    links_names = ['blender', 'mmvt', 'subjects', 'eeg', 'meg', 'fMRI', 'electrodes']
    # if not utils.is_windows():
    #     links_names.insert(1, 'subjects')
    if not overwrite:
        all_links_exist = utils.all([utils.is_link(op.join(links_fol, link_name)) for link_name in links_names])
        if all_links_exist:
            print('All links exist!')
            links = {link_name: utils.get_link_dir(links_fol, link_name) for link_name in links_names}
            write_links_into_csv_file(links, links_fol, links_file_name)
            return True
    if not utils.is_windows() and not utils.is_link(op.join(links_fol, 'freesurfer')):
        if os.environ.get('FREESURFER_HOME', '') != '':
            freesurfer_fol = os.environ['FREESURFER_HOME']
            if not only_verbose:
                create_real_folder(freesurfer_fol)
        # else:
        #     print('If you are going to use FreeSurfer locally, please source it and rerun')
        #     # If you choose to continue, you'll need to create a link to FreeSurfer manually")
        #     cont = input("Do you want to continue (y/n)?")
        #     if cont.lower() != 'y':
        #         return

    mmvt_message = 'Please select where do you want to put the blend files '
    subjects_message = \
        'Please select where you want to store the FreeSurfer recon-all files neccessary for MMVT.\n' + \
        '(Creating a local folder is preferred, because MMVT is going to save files to this directory) '
    blender_message = 'Please select the folder containing the Blender App'
    meg_message = 'Please select where you want to put the MEG files (Cancel if you are not going to use MEG data) '
    eeg_message = 'Please select where you want to put the EEG files (Cancel if you are not going to use EEG data) '
    fmri_message = 'Please select where you want to put the fMRI files (Cancel if you are not going to use fMRI data) '
    electrodes_message = 'Please select where you want to put the electrodes files (Cancel if you are not going to use electrodes data) '

    blender_fol = find_blender()
    if blender_fol != '':
        utils.create_folder_link(blender_fol, op.join(links_fol, 'blender'), overwrite)
    else:
        ask_and_create_link(links_fol, 'blender',  blender_message, gui, overwrite)
    default_message = "Would you like to set default links to the MMVT's folders?\n" + \
        "You can always change that later by running\n" + \
        "python -m src.setup -f create_links"
    create_default_folders = default_folders or mmvt_input(default_message, gui, 4) == 'Yes'

    messages = [mmvt_message, subjects_message, eeg_message, meg_message, fmri_message, electrodes_message]
    deafault_fol_names = ['mmvt_blend', 'subjects', 'eeg', 'meg', 'fMRI', 'electrodes']
    # if not utils.is_windows():
    #     messages.insert(0, subjects_message)
    create_default_dirs = [False] * 3 + [True] * 2 + [False] * 2

    links = {}
    if not only_verbose:
        for link_name, default_fol_name, message, create_default_dir in zip(
                links_names[1:], deafault_fol_names, messages, create_default_dirs):
            fol = ''
            if not create_default_folders:
                fol, ret = ask_and_create_link(
                    links_fol, link_name, message, gui, create_default_dir, overwrite=overwrite)
            if fol == '' or create_default_folders:
                fol, ret = create_default_link(
                    links_fol, link_name, default_fol_name, create_default_dir, overwrite=overwrite)
            if ret:
                print('The "{}" link was created to {}'.format(link_name, fol))
            links[link_name] = fol

    links = get_all_links(links, links_fol)
    write_links_into_csv_file(links, links_fol, links_file_name)
    return utils.all([utils.is_link(op.join(links_fol, link_name)) for link_name in links_names])


def mmvt_input(message, gui, style=1):
    try:
        import pymsgbox
    except:
        gui=False

    if gui:
        ret = utils.message_box(message, TITLE, style)
    else:
        ret = input(message)
    return ret


def ask_and_create_link(links_fol, link_name, message, gui=True, create_default_dir=False, overwrite=True):
    fol = ''
    ret = False
    if not overwrite and utils.is_link(op.join(links_fol, link_name)):
        fol = utils.get_link_dir(links_fol, link_name)
        ret = True
    else:
        choose_folder = mmvt_input(message, gui) == 'Ok'
        if choose_folder:
            root_fol = utils.get_parent_fol(links_fol)
            fol = utils.choose_folder_gui(root_fol, message) if gui else input()
            if link_name == 'blender' and utils.is_osx():
                fol = op.join(fol, 'blender.app','Contents','MacOS')
            if fol != '':
                create_real_folder(fol)
                ret = utils.create_folder_link(fol, op.join(links_fol, link_name), overwrite=overwrite)
                if create_default_dir:
                    utils.make_dir(op.join(fol, 'default'))
    return fol, ret


def get_mmvt_root_folder():
    return utils.get_parent_fol(levels=3)


def create_default_link(links_fol, link_name, default_fol_name, create_default_dir=False, overwrite=True):
    fol = op.join(get_mmvt_root_folder(), default_fol_name)
    create_real_folder(fol)
    ret = utils.create_folder_link(fol, op.join(links_fol, link_name), overwrite=overwrite)
    if create_default_dir:
        utils.make_dir(op.join(fol, 'default'))
    return fol, ret


def get_all_links(links={}, links_fol=None, links_fol_name='links'):
    if links_fol is None:
        links_fol = utils.get_links_dir(links_fol_name)
    all_links = [utils.namebase(f) for f in glob.glob(op.join(links_fol, '*')) if utils.is_link(f)]
    all_links = {link_name: utils.get_link_dir(links_fol, link_name) for link_name in all_links if link_name not in links}
    links = utils.merge_two_dics(links, all_links)
    return links


def write_links_into_csv_file(links, links_fol=None, links_file_name='links.csv', links_fol_name='links'):
    import csv
    if links_fol is None:
        links_fol = utils.get_links_dir(links_fol_name)
    with open(op.join(links_fol, links_file_name), 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for link_name, link_dir in links.items():
            csv_writer.writerow([link_name, link_dir])


def create_empty_links_csv(links_fol_name='links', links_file_name='links.csv'):
    links_fol = utils.get_links_dir(links_fol_name)
    links_names = ['mmvt', 'subjects', 'blender', 'eeg', 'meg', 'fMRI', 'electrodes']
    links = {link_name: '' for link_name in links_names}
    write_links_into_csv_file(links, links_fol, links_file_name)


def create_real_folder(real_fol):
    try:
        if real_fol == '':
            real_fol = utils.get_resources_fol()
        utils.make_dir(real_fol)
    except:
        print('Error with creating the folder "{}"'.format(real_fol))
        print(traceback.format_exc())


def install_reqs(do_upgrade=False, only_verbose=False):
    try:
        return utils.run_script('pip3 install --user -r requirements.txt')
    except:
        return install_reqs_loop(do_upgrade, only_verbose)


def install_reqs_loop(do_upgrade=False, only_verbose=False):
    # https://blender.stackexchange.com/questions/149944/how-to-write-my-add-on-so-that-when-installed-it-also-installs-dependencies-let/153520#153520
    # import pip
    try:
        from pip import main as pipmain
    except:
        from pip._internal import main as pipmain
    # if utils.is_windows() and not utils.is_admin():
    #     utils.set_admin()
    pipmain(['install', '--upgrade', 'pip'])
    retcode = 0
    reqs_fname = op.join(utils.get_parent_fol(levels=2), 'requirements.txt')
    with open(reqs_fname, 'r') as f:
        for line in f:
            if only_verbose:
                print('Trying to install {}'.format(line.strip()))
            else:
                if do_upgrade:
                    pipcode = pipmain(['install', '--upgrade', line.strip()])
                else:
                    pipcode = pipmain(['install', '--user', line.strip()])
                retcode = retcode or pipcode
    return retcode


def find_blender_in_linux(fol, look_for_dirs=True):
    blender_fol = ''
    if look_for_dirs:
        output = utils.get_command_output("find {} -name '*blender*' -type d".format(fol))
        blender_fols = output.split('\n')
        blender_fols = [fol for fol in blender_fols if op.isfile(
            op.join(fol, 'blender.svg')) or 'blender.app' in fol]
        if len(blender_fols) >= 1:
            # todo: let the user select which one
            blender_fol = blender_fols[0]
    else:
        output = utils.get_command_output("find {} -name '*blender*'".format(fol))
        blender_fols = output.split('\n')
        blender_fols = [fol for fol in blender_fols if utils.is_link(fol) and
                        op.isfile(op.join(fol, 'blender.svg'))]
        if len(blender_fols) >= 1:
            # todo: let the user select which one
            blender_fol = blender_fols[0]
    return blender_fol


def find_blender():
    blender_fol = ''
    if utils.is_windows():
        blender_win_fol = 'Program Files\Blender Foundation\Blender*'
        for drive in ['C', 'D']:
            if len(glob.glob(op.join('{}:\\'.format(drive), blender_win_fol))) > 0:
                blender_fol = select_file(glob.glob(op.join('{}:\\'.format(drive), blender_win_fol)))
                break
    elif utils.is_linux():
        blender_fol = find_blender_in_linux('../', False)
        if blender_fol == '':
            blender_fol = find_blender_in_linux('../../')
        if blender_fol == '':
            blender_fol = find_blender_in_linux('~/')
    elif utils.is_osx():
        blender_fol = op.join(utils.get_parent_fol(levels=3), 'blender-2.79b-macOS-10.6', 'blender.app', 'Contents', 'MacOS')
        if not op.isdir(blender_fol):
            blender_fol = '/Applications/Blender/blender.app/Contents/MacOS'
        blender_fol = blender_fol if op.isdir(blender_fol) else ''
        # output = utils.run_script("find ~/ -name 'blender' -type d")
        # if not isinstance(output, str):
        #     output = output.decode(sys.getfilesystemencoding(), 'ignore')
        # blender_fols = output.split('\n')
        # blender_fols = [fol for fol in blender_fols if 'blender.app' in fol]
        # if len(blender_fols) == 1:
        #     blender_fol = op.join(blender_fols[0], 'blender.app', 'Contents', 'MacOS', 'blender')
    print(blender_fol)
    return blender_fol


def read_blender_link():
    print('Blender link: {}'.format(utils.get_link_dir(utils.get_links_dir(), 'blender')))


def create_fsaverage_link(links_fol_name='links'):
    freesurfer_home = os.environ.get('FREESURFER_HOME', '')
    if freesurfer_home != '':
        links_fol = utils.get_links_dir(links_fol_name)
        subjects_dir = utils.get_link_dir(links_fol, 'subjects', 'SUBJECTS_DIR')
        fsaverage_link = op.join(subjects_dir, 'fsaverage')
        if not utils.is_link(fsaverage_link):
            fsveareg_fol = op.join(freesurfer_home, 'subjects', 'fsaverage')
            utils.create_folder_link(fsveareg_fol, fsaverage_link)


def get_blender_python_exe(blender_fol, gui=True):
    bin_template = op.join(blender_fol, '..', 'Resources', '2.??', 'python') if utils.is_osx() else \
        op.join(blender_fol, '2.??', 'python')
    blender_bin_folders = sorted(glob.glob(bin_template))
    if len(blender_bin_folders) == 0:
        print("Couldn't find Blender's bin folder! ({})".format(bin_template))
        blender_bin_fol = ''
        choose_folder = mmvt_input('Please choose the Blender bin folder where python file exists', gui) == 'Ok'
        if choose_folder:
            fol = utils.choose_folder_gui(blender_parent_fol, 'Blender bin folder') if gui else input()
            if fol != '':
                blender_bin_fol = glob.glob(op.join(fol, '2.??', 'python'))[-1]
        if blender_bin_fol == '':
            return '', ''
    elif len(blender_bin_folders) > 1:
        blender_bin_fol = select_file(blender_bin_folders)
    else:
        blender_bin_fol = blender_bin_folders[-1]
    python_exe = 'python.exe' if utils.is_windows() else 'python3.5m'
    return blender_bin_fol, python_exe


def select_file(files):
    if len(files) == 1:
        return files[0]
    print('More than one blender exist, please choose a folder:')
    for ind, fname in enumerate(files):
        print('{}) {}'.format(ind + 1, fname))
    input_str = 'Which one do you want to pick (1, 2, ...)? Press 0 to cancel: '
    file_num = input(input_str)
    while not is_int(file_num):
        print('Please enter a valid integer')
        file_num = input(input_str)
    if file_num == 0:
        return ''
    else:
        file_num = int(file_num) - 1
        return files[file_num]


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_pip_update_cmd(package='numpy'):
    blender_fol = utils.get_link_dir(utils.get_links_dir(), 'blender')
    blender_bin_fol, python_exe = get_blender_python_exe(blender_fol, False)
    install_cmd = '{} install --upgrade {}'.format(op.join(blender_bin_fol, 'bin', 'pip'), package)
    print(install_cmd)


def install_blender_reqs(blender_fol='', gui=True):
    # http://stackoverflow.com/questions/9956741/how-to-install-multiple-python-packages-at-once-using-pip
    try:
        if blender_fol == '':
            blender_fol = utils.get_link_dir(utils.get_links_dir(), 'blender')
        resource_fol = utils.get_resources_fol()
        blender_bin_fol, python_exe = get_blender_python_exe(blender_fol, gui)
        current_dir = os.getcwd()
        os.chdir(blender_bin_fol)
        # install blender reqs:
        # todo: take this list from the reqs file
        reqs = 'matplotlib zmq pizco scipy mne joblib tqdm nibabel pdfkit decorator Pillow scikit-learn gitpython decorator'
        pip_cmd = '{} {}'.format(op.join('bin', python_exe), op.join(resource_fol, 'get-pip.py'))
        if not utils.is_windows():
            utils.run_script(pip_cmd)
            # https://github.com/pypa/pip/issues/5226
            # https://stackoverflow.com/questions/49743961/cannot-upgrade-pip-9-0-1-to-9-0-3-requirement-already-satisfied/49758204#49758204
            # utils.run_script('curl https://bootstrap.pypa.io/get-pip.py | python3')
            install_cmd = '{} install --upgrade {} --user'.format(op.join('bin', 'pip'), reqs)
            utils.run_script(install_cmd)
        else:
            # https://stackoverflow.com/questions/130763/request-uac-elevation-from-within-a-python-script
            install_cmd = '{} install {}'.format(op.join('Scripts', 'pip'), reqs)
            print(
                'Sorry, automatically installing external python libs in python will be implemented in the future.\n' +
                'Meanwhile, you can do the following:\n' +
                '1) Open a terminal window as administrator: ' +
                'Right click on the "Command Prompt" shortcut from the start menu and choose "Run as administrator"\n' +
                '2) Change the directory to "{}".\n'.format(blender_bin_fol) +
                '3) Run "{}"\n'.format(pip_cmd) +
                '4) Run "{}"\nGood luck!'.format(install_cmd))
            # from src.mmvt_addon.scripts import install_blender_reqs
            # install_blender_reqs.wrap_blender_call(args.only_verbose)
        os.chdir(current_dir)
    except:
        print(traceback.format_exc())
        print("*** Can't install pizco ***")


def create_launcher(): ###fix for mac
    if utils.is_linux() or utils.is_osx():
        mmvt_root_path = get_mmvt_root_folder()
        launcher_fname = op.join(mmvt_root_path, 'launch_mmvt.sh')
        script_code = '#!/bin/tcsh\n'
        # freesurfer_bin = utils.run_script('which freesurfer')
        # freesurfer_home = utils.get_parent_fol(freesurfer_bin, levels=2)
        freesurfer_home = op.join(mmvt_root_path, 'freesurfer/')
        if op.isdir(freesurfer_home):
            script_code += \
                'setenv FREESURFER_HOME {}\n'.format(freesurfer_home) +\
                'source $FREESURFER_HOME/SetUpFreeSurfer.csh\n'
        elif os.environ.get('FREESURFER_HOME'):
            freesurfer_home = os.environ.get('FREESURFER_HOME')
            script_code += \
                'setenv FREESURFER_HOME {}\n'.format(freesurfer_home) + \
                'source $FREESURFER_HOME/SetUpFreeSurfer.csh\n'
        else:
            print('Can\'t find FREESURFER_HOME ({})'.format(freesurfer_home))
        if utils.is_osx():
            blender_exe = 'blender-2.79b-macOS-10.6/blender.app/Contents/MacOS/blender'
        elif utils.is_linux():
            blender_exe = 'blender-2.79b-linux-glibc219-x86_64/blender'
        script_code += '{}/{} --enable-new-depsgraph\n'.format(get_mmvt_root_folder(), blender_exe)
        print('Creating launcher: {}'.format(launcher_fname))
        with open(launcher_fname, 'w') as f:
            f.write(script_code)
        utils.run_script('chmod +x {}'.format(launcher_fname))
    else:
        print('Auto launcher for Windows is not yet implemented.')


def send_email():
    try:
        ip = utils.get_ip_address()
        utils.send_email('mmvt_setup', ip, 'MultiModalityVisualizationTool@gmail.com')
    except:
        pass


def main(args):
    # If python version is < 3.5, use Blender's python
    if sys.version_info[0] < 3 or sys.version_info[0] == 3 and sys.version_info[1] < 5:
        blender_fol = find_blender()
        blender_bin_fol, python_exe = get_blender_python_exe(blender_fol)
        blender_python_exe = op.join(blender_bin_fol, 'bin', python_exe)
        if not op.isfile(blender_python_exe):
            print('You must use python 3.5 or newer, or install first Blender')
        else:
            # rerun setup with Blender's python
            args.blender_fol = blender_fol
            call_args = utils.create_call_args(args)
            setup_cmd = '{} -m src.setup {}'.format(blender_python_exe, call_args)
            utils.run_script(setup_cmd, print_only=False)
        return

    print(args)
    links_dir = utils.get_links_dir(args.links)
    mmvt_root_dir = utils.get_link_dir(links_dir, 'mmvt')

    # 1) Install dependencies from requirements.txt (created using pipreqs)
    if utils.should_run(args, 'install_reqs'):
        install_reqs(args.upgrade_reqs_libs, args.only_verbose)

    # 2) Create links
    if utils.should_run(args, 'create_links'):
        links_created = create_links(args.links, args.gui, args.default_folders, args.only_verbose,
                                     args.links_file_name, args.overwrite_links)
        if not links_created:
            print('Not all the links were created! Make sure all the links are created before running MMVT.')

    # 2,5) Create fsaverage folder link
    if utils.should_run(args, 'create_fsaverage_link'):
        create_fsaverage_link(args.links)

    # 3) Copy resources files
    if utils.should_run(args, 'copy_resources_files'):
        resource_file_exist = copy_resources_files(mmvt_root_dir, args.overwrite, args.only_verbose)
        if not resource_file_exist:
            input('Not all the resources files were copied to the MMVT folder ({}).\n'.format(mmvt_root_dir) +
                  'Please copy them manually from the mmvt_code/resources folder.\n' +
                  'Press any key to continue...')

    # 3.5) Download empty subject from Dropbox
    if utils.should_run(args, 'download_empty_subject'):
        download_empty_subject(mmvt_root_dir, EMPTY_SUBJECT_URL, args.overwrite)

    # 4) Install the addon in Blender (depends on resources and links)
    if utils.should_run(args, 'install_addon'):
        from src.mmvt_addon.scripts import install_addon
        install_addon.wrap_blender_call(args.only_verbose)

    # 5) Install python packages in Blender
    if utils.should_run(args, 'install_blender_reqs'):
        install_blender_reqs(args.blender_fol)

    # 6) Install deface (https://surfer.nmr.mgh.harvard.edu/fswiki/mri_deface)
    if utils.should_run(args, 'install_deface'):
        install_deface()

    # 7) Create a launcher to run MMVT
    if utils.should_run(args, 'create_launcher'):
        create_launcher()

    if utils.should_run(args, 'send_email'):
        send_email()

    if 'create_links_csv' in args.function:
        create_empty_links_csv()

    if 'create_csv' in args.function:
        write_links_into_csv_file(get_all_links())

    if 'find_blender' in args.function:
        find_blender()

    if 'get_pip_update_cmd' in args.function:
        get_pip_update_cmd()

    if 'read_blender_link' in args.function:
        read_blender_link()

    # print('Finish!')


def print_help():
    str = '''
    Flags:
        -l: The links folder name (default: 'links')
        -g: Use GUI (True) or the command line (False) (default: True)
        -v: If True, just check the setup without doing anything (default: False)
        -d: If True, the script will create the default mmvt folders (default: True)
        -o: If True, the scirpt will overwrite the resources files (default: True)
        -f: Set which function (or functions) you want to run (use commas witout spacing) (default: all):
            install_reqs, create_links, copy_resources_files, install_addon, install_blender_reqs, create_launcher,
            create_links_csv, and and create_csv
    '''
    print(str)


def probles():
    # https://blender.stackexchange.com/questions/96020/libglu-so-1-error-loading-shared-library
    pass


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['h', 'help', '-h', '-help']:
        print_help()
        exit()

    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT Setup')
    parser.add_argument('-l', '--links', help='links folder name', required=False, default='links')
    parser.add_argument('-g', '--gui', help='choose folders using gui', required=False, default='1', type=au.is_true)
    parser.add_argument('-v', '--only_verbose', help='only verbose', required=False, default='0', type=au.is_true)
    parser.add_argument('-d', '--default_folders', help='default options', required=False, default='1', type=au.is_true)
    parser.add_argument('-f', '--function', help='functions to run', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('-o', '--overwrite', help='Overwrite resources', required=False, default='1', type=au.is_true)
    parser.add_argument('--blender_fol', help='', required=False, default='')
    parser.add_argument('--links_file_name', help='', required=False, default='links.csv')
    parser.add_argument('--overwrite_links', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--upgrade_reqs_libs', help='', required=False, default=0, type=au.is_true)
    args = utils.Bag(au.parse_parser(parser))
    main(args)
