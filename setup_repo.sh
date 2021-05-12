# run this code on a clean folder to download CUT/CycleGAN repo and setup the working environment.
git clone https://github.com/taesungp/contrastive-unpaired-translation CUT

# create virtual environment with python3
python3 -m venv env
source env/bin/activate

# install dependencies
pip install -r CUT/requirements.txt
