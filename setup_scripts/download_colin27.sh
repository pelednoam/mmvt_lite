# Getting colin27
#if operating system is mac


if [[ "$OSTYPE" == "darwin"* ]]
then
	curl -L https://www.dropbox.com/sh/hjto15nnebrdg27/AABpdynLda9KoRkWaNB0WCDaa?dl=1 -o colin27_dropbox.zip
else
	#sudo apt install unzip
  wget https://www.dropbox.com/sh/hjto15nnebrdg27/AABpdynLda9KoRkWaNB0WCDaa?dl=1 -O colin27_dropbox.zip
fi

unzip -o colin27_dropbox.zip
rm colin27_dropbox.zip

mv colin27.zip ./mmvt_blend
cd ../mmvt_blend || { echo "Failure - No mmvt_blend folder"; exit 1; }
unzip -o colin27.zip
rm colin27.zip
cd ..

mv colin27_subjects.zip ./subjects
cd ../subjects || { echo "Failure - No subjects folder"; exit 1; }
unzip colin27_subjects.zip
mv colin27_subjects colin27
rm colin27_subjects.zip
cd ..

