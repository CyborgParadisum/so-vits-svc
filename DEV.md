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

### local inference
- linux
    ```bash
    conda create --name so-vits-svc python=3.8
    conda activate so-vits-svc
    pip install -r requirements.txt
    ```
- mac
    ```bash
    conda create --name so-vits-svc python=3.10
    conda activate so-vits-svc
    pip install -r requirements-mac.txt
    ```

