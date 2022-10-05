<img src=https://user-images.githubusercontent.com/35853195/42889397-52f9c75e-8a78-11e8-9da8-86ccc3a30a80.png align="right" hight=120 width=120/>
This is a lite version of <a href="https://mmvt.mgh.harvard.edu">MMVT</a> without the visualization component 

## Installation steps for Linux:  

```
mkdir mmvt_lite
cd mmvt_lite
# Clone the MMVT lite repo
git clone -b master https://github.com/pelednoam/mmvt_lite.git
# Clone the Electrodes Labeling Algorithm repo
git clone -b master https://github.com/pelednoam/electrodes_rois.git
mv mmvt_lite mmvt_code

# Create a virtual python enviroment for mmvt
python -m pip install --user --upgrade pip
python -m pip install --user virtualenv
python -m venv mmvt_env
source mmvt_env/bin/activate

# Run set setup 
cd mmvt_code
python -m src.setup
# Download the templalte brain colin27
cd ..
sh mmvt_code/setup_scripts/download_colin27.sh
```
