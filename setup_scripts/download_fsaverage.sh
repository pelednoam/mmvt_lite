# Getting colin27
#if operating system is mac
mkdir mmvt_blend
cd mmvt_blend || { echo "Failure - No mmvt_blend folder"; exit 1; }

if [[ "$OSTYPE" == "darwin"* ]]
then
	curl -L https://www.dropbox.com/s/g3y2uyj87hs1gka/fsaverage_dkt.blend?dl=1 -o fsaverage_dkt.blend
  curl -L https://www.dropbox.com/s/ojb49zd559jv02y/fsaverage.zip?dl=1 -o fsaverage.zip
  curl -L https://www.dropbox.com/s/1g1445ae5z7a6gv/fsaverage_subjects.zip?dl=1 -o fsaverage_subjects.zip

else
	sudo apt install unzip
	wget https://www.dropbox.com/s/g3y2uyj87hs1gka/fsaverage_dkt.blend?dl=1 -O fsaverage_dkt.blend
  wget https://www.dropbox.com/s/ojb49zd559jv02y/fsaverage.zip?dl=1 -O fsaverage.zip
  wget https://www.dropbox.com/s/1g1445ae5z7a6gv/fsaverage_subjects.zip?dl=1 -O fsaverage_subjects.zip
fi

unzip -o fsaverage.zip
rm fsaverage.zip
cd ..
mkdir subjects
cd subjects
mv ../mmvt_blend/fsaverage_subjects.zip .
unzip fsaverage_subjects.zip
rm fsaverage_subjects.zip
cd ..