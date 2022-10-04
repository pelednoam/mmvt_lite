# Getting colin27
#if operating system is mac
mkdir subjects
mkdir mmvt_blend
cd mmvt_blend || { echo "Failure - No mmvt_blend folder"; exit 1; }

if [[ "$OSTYPE" == "darwin"* ]]
then
	curl -L https://www.dropbox.com/sh/hjto15nnebrdg27/AABpdynLda9KoRkWaNB0WCDaa?dl=1 -o colin27_dropbox.zip
else
	sudo apt install unzip
  wget https://www.dropbox.com/sh/hjto15nnebrdg27/AABpdynLda9KoRkWaNB0WCDaa?dl=1 -O colin27_dropbox.zip
fi

unzip -o colin27_dropbox.zip
rm colin27_dropbox.zip
unzip -o colin27.zip

#n_files = !unzip -l colin27.zip | grep .png | wc -l
#!unzip -o colin27.zip | pv -l -s {n_files[0]} > /dev/null

rm colin27.zip
mv colin27_subjects.zip ../subjects/
cd ../subjects || { echo "Failure - No subjects folder"; exit 1; }
unzip colin27_subjects.zip
mv colin27_subjects colin27
rm colin27_subjects.zip
cd ../