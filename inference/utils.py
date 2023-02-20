import io
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc
from utils import hubert_path

logging.getLogger('numba').setLevel(logging.WARNING)
# chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

# model_path = "logs/32k/G_50000.pth"
# config_path = "configs/config.json"


# infer_tool.mkdir(["raw", "results"])

# 支持多个wav文件，放在raw文件夹下
# clean_names = ["hello1"]
# trans = [0]  # 音高调整，支持正负（半音）
# spk_list = ['mikisayaka']  # 每次同时合成多语者音色
# slice_db = -50  # 默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
# wav_format = 'wav'  # 音频输出格式

class SoVitsSvc(object):
    def __init__(self, model_path, config_path):
        self.svc_model = Svc(model_path, config_path, hubert_path)

    # def inference_bytes():

    def inference(self, srcaudio, chara, tran=0, slice_db=-40):
        """
        :param srcaudio: (sampling_rate, audio)
        :param tran: 升降调
        :param slice_db: 切片阈值
        """
        sampling_rate, audio = srcaudio
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        tmpwav = "tmpwav.wav"
        soundfile.write(tmpwav, audio, 16000, format="wav")
        chunks = slicer.cut(tmpwav, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(tmpwav, chunks)

        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            length = int(np.ceil(len(data) / audio_sr * self.svc_model.target_sample))
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
            else:
                out_audio, out_sr = self.svc_model.infer(chara, tran, raw_path)
                _audio = out_audio.cpu().numpy()
            audio.extend(list(_audio))
        audio = (np.array(audio) * 32768.0).astype('int16')
        os.remove(tmpwav)
        return self.svc_model.target_sample, audio
