source activate_python.sh


git clone https://github.com/NVIDIA/apex
cd apex
pip install  --no-cache-dir ./ # only python  

git clone https://github.com/NVIDIA/mellotron.git
cd mellotron
git submodule init
git submodule update

wget -N  -q https://raw.githubusercontent.com/yhgon/colab_utils/master/gfile.py
python gfile.py -u 'https://drive.google.com/open?id=1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI' -f 'mellotron_libritts.pt'
python gfile.py -u 'https://drive.google.com/open?id=1Rm5rV5XaWWiUbIpg5385l5sh68z2bVOE' -f 'waveglow_256channels_v4.pt'

touch mellotron.done
