#!/bin/bash

pip uninstall -y requirements.txt
pip install -y requirements.txt
pip install torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
