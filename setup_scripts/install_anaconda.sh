# install Anaconda
if [[ "$OSTYPE" == "darwin"* ]]
then
  curl https://repo.anaconda.com/archive/Anaconda3-2020.11-MacOSX-x86_64.sh -o anaconda3_setup.sh
else
  wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O anaconda3_setup.sh
fi

bash anaconda3_setup.sh
rm anaconda3_setup.sh
