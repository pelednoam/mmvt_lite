# The setup suppose to run *before* installing python libs, so only python vanilla can be used here

import os
import os.path as op
import shutil
import traceback
from src.utils import setup_utils as utils
import glob

TITLE = 'MMVT Lite Installation'


def copy_resources_files(overwrite=True, only_verbose=False):
    mmvt_root_dir = op.join(get_mmvt_root_folder(), 'mmvt_blend')
    utils.make_dir(op.join(op.join(mmvt_root_dir, 'color_maps')))
    files = ['atlas.csv', 'sub_cortical_codes.txt', 'FreeSurferColorLUT.txt']
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
    links_names = ['mmvt', 'subjects', 'eeg', 'meg', 'fMRI', 'electrodes']
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
    meg_message = 'Please select where you want to put the MEG files (Cancel if you are not going to use MEG data) '
    eeg_message = 'Please select where you want to put the EEG files (Cancel if you are not going to use EEG data) '
    fmri_message = 'Please select where you want to put the fMRI files (Cancel if you are not going to use fMRI data) '
    electrodes_message = 'Please select where you want to put the electrodes files (Cancel if you are not going to use electrodes data) '

    default_message = "Would you like to set default links to the MMVT's folders?\n" + \
        "You can always change that later by running\n" + \
        "python -m src.setup -f create_links"
    create_default_folders = default_folders or mmvt_input(default_message, gui, 4) == 'Yes'

    messages = [mmvt_message, subjects_message, eeg_message, meg_message, fmri_message, electrodes_message]
    deafault_fol_names = ['mmvt_blend', 'subjects', 'eeg', 'meg', 'fMRI', 'electrodes']
    create_default_dirs = [False] * 6

    links = {}
    if not only_verbose:
        for link_name, default_fol_name, message, create_default_dir in zip(
                links_names, deafault_fol_names, messages, create_default_dirs):
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
    links_names = ['subjects', 'eeg', 'meg', 'fMRI', 'electrodes']
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


def install_reqs_using_text(do_upgrade=False, only_verbose=False):
    try:
        # return utils.run_script('pip3 install --user -r requirements.txt')
        return utils.run_script('pip install -r requirements.txt')
    except:
        return install_reqs_loop(do_upgrade, only_verbose)


def upgrade_pip():
    return utils.run_script('pip install --upgrade pip')


def install_reqs():
    upgrade_pip()
    reqs_fname = op.join(utils.get_parent_fol(levels=2), 'requirements.txt')
    with open(reqs_fname, 'r') as f:
        for line in f:
            line_parts = line.strip().split('==')
            print('reqs line: {}'.format(line_parts))
            if len(line_parts) == 1:
                utils.run_script('pip install {}'.format(line_parts[0]))
            else:
                utils.run_script('pip install {}=={}'.format(line_parts[0], line_parts[1]))


def install_reqs_loop(do_upgrade=False, only_verbose=False):
    try:
        from pip import main as pipmain
    except:
        from pip._internal import main as pipmain
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


def create_fsaverage_link(links_fol_name='links'):
    freesurfer_home = os.environ.get('FREESURFER_HOME', '')
    if freesurfer_home != '':
        links_fol = utils.get_links_dir(links_fol_name)
        subjects_dir = utils.get_link_dir(links_fol, 'subjects', 'SUBJECTS_DIR')
        fsaverage_link = op.join(subjects_dir, 'fsaverage')
        if not utils.is_link(fsaverage_link):
            fsveareg_fol = op.join(freesurfer_home, 'subjects', 'fsaverage')
            utils.create_folder_link(fsveareg_fol, fsaverage_link)


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def main(args):
    print(args)

    # 1) Install dependencies from requirements.txt (created using pipreqs)
    if utils.should_run(args, 'install_reqs'):
        install_reqs() #args.upgrade_reqs_libs, args.only_verbose)

    # 2) Create links
    if utils.should_run(args, 'create_links'):
        links_created = create_links(
            args.links, args.gui, args.default_folders, args.only_verbose,
            args.links_file_name, args.overwrite_links)
        if not links_created:
            print('Not all the links were created! Make sure all the links are created before running MMVT.')

    # 2,5) Create fsaverage folder link
    if utils.should_run(args, 'create_fsaverage_link'):
        create_fsaverage_link(args.links)

    # 3) Copy resources files
    if utils.should_run(args, 'copy_resources_files'):
        resource_file_exist = copy_resources_files(args.overwrite, args.only_verbose)
        if not resource_file_exist:
            input('Not all the resources files were copied to the MMVT folder ({}).\n'.format(mmvt_root_dir) +
                  'Please copy them manually from the mmvt_code/resources folder.\n' +
                  'Press any key to continue...')

    # 4) Install deface (https://surfer.nmr.mgh.harvard.edu/fswiki/mri_deface)
    # if utils.should_run(args, 'install_deface'):
    if 'install_deface' in args.function:
        install_deface()

    if 'create_links_csv' in args.function:
        create_empty_links_csv()

    if 'create_csv' in args.function:
        write_links_into_csv_file(get_all_links())

    if 'get_pip_update_cmd' in args.function:
        get_pip_update_cmd()

    if 'upgrade_pip' in args.function:
        upgrade_pip()


def print_help():
    str = '''
    Flags:
        -l: The links folder name (default: 'links')
        -g: Use GUI (True) or the command line (False) (default: True)
        -v: If True, just check the setup without doing anything (default: False)
        -d: If True, the script will create the default mmvt folders (default: True)
        -o: If True, the scirpt will overwrite the resources files (default: True)
        -f: Set which function (or functions) you want to run (use commas witout spacing) (default: all):
            install_reqs, create_links, copy_resources_files, create_links_csv, and and create_csv
    '''
    print(str)


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
    parser.add_argument('--links_file_name', help='', required=False, default='links.csv')
    parser.add_argument('--overwrite_links', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--upgrade_reqs_libs', help='', required=False, default=0, type=au.is_true)
    args = utils.Bag(au.parse_parser(parser))
    main(args)
