#type "source init.sh"
mkdir data models config custom
virtualenv env

source env/bin/activate
pip3 install -r requirements.txt
python3 -m ipykernel install --user --name=env --display-name=$PWD
jupyter nbextension enable --py widgetsnbextension
