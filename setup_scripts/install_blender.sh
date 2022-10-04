# Get blender
#if operating system is mac
if [[ "$OSTYPE" == "darwin"* ]]
then
	curl https://download.blender.org/release/Blender2.79/blender-2.79b-macOS-10.6.zip -o blender-2.79b-macOS-10.6.zip
	unzip blender-2.79b-macOS-10.6.zip
	rm blender-2.79b-macOS-10.6.zip
else
	wget https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2
	tar -xvjf blender-2.79b-linux-glibc219-x86_64.tar.bz2
	rm blender-2.79b-linux-glibc219-x86_64.tar.bz2
fi