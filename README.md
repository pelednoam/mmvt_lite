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

# Run MMVT setup 
cd mmvt_code
python -m src.setup
# Download the templaltes brain colin27 and fsaverage
sh ./setup_scripts/download_colin27.sh
sh ./setup_scripts/download_fsaverage.sh
```

### Citations
If you are using MMVT Lite in your paper, please cite the following:

(MMVT)[https://scholar.google.com/scholar?hl=en&as_sdt=0%2C22&q=Multi-modal+neuroimaging+analysis+and+visualization+tool+%28MMVT%29&btnG=]:
Felsenstein, O., N. Peled, E. Hahn, A. P. Rockhill, L. Folsom, T. Gholipour, K. Macadams et al. "Multi-modal neuroimaging analysis and visualization tool (MMVT)." arXiv preprint arXiv:1912.10079 (2019).
	
The ELA algorithm:
Peled, N., T. Gholipour, A. C. Paulk, O. Felsenstein, D. D. Dougherty, A. S. Widge, E. N. Eskandar, S. S. Cash, M. S. Hamalainen, and S. M. Stufflebeam. "Invasive electrodes identification and labeling." GitHub Repos 10 (2017).

