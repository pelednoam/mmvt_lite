# FreeSurfer
#if operating system is mac
if [[ "$OSTYPE" == "darwin"* ]]
then
  cur https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-darwin-macOS-7.3.2.tar.gz -o freesurfer.tar.gz
else
  # todo: check if ubuntu or centos
  # wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-ubuntu18_amd64-7.3.2.tar.gz -O freesurfer.tar.gz
	wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-centos7_x86_64-7.3.2.tar.gz -O freesurfer.tar.gz
  sudo apt-get install csh tcsh
fi

tar -zxvpf freesurfer.tar.gz
rm freesurfer.tar.gz
mmvt_root=$(pwd)
export FREESURFER_HOME=$mmvt_root/freesurfer
export SUBJECTS_DIR=$mmvt_root/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh
recon-all -version
