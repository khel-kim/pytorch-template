#type "source init.sh"
mkdir data models config custom
virtualenv env

rm -rf .git

source env/bin/activate
pip3 install -r requirements.txt
python3 -m ipykernel install --user --name=env --display-name=$PWD
