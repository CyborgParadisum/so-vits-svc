## mac ( mac is not supported )
```bash
GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=true GRPC_PYTHON_BUILD_SYSTEM_ZLIB=true pip install grpcio

pip install -r requirements-mac.txt

export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

pip install --force-reinstall 'cffi>=1.15.1'
```
- install pyaudio
https://stackoverflow.com/questions/68251169/unable-to-install-pyaudio-on-m1-mac-portaudio-already-installed
```bash
brew install portaudio
```
