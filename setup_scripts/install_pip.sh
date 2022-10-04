# if no Anaconda, install pip
#if mac
if [[ "$OSTYPE" == "darwin"* ]]
then
  # pip should be already installed, let's check
  pip3 -V
##  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
##  python get-pip.py
else
	sudo apt install python3-pip
	sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
fi
