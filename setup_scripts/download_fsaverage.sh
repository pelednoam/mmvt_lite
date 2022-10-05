# Getting colin27
#if operating system is mac
cd ../subjects || { echo "Failure - No mmvt_blend folder"; exit 1; }

if [[ "$OSTYPE" == "darwin"* ]]
then
  curl -L https://www.dropbox.com/s/1g1445ae5z7a6gv/fsaverage_subjects.zip?dl=1 -o fsaverage_subjects.zip

else
  wget https://www.dropbox.com/s/1g1445ae5z7a6gv/fsaverage_subjects.zip?dl=1 -O fsaverage_subjects.zip
fi

unzip fsaverage_subjects.zip
rm fsaverage_subjects.zip
cd ..