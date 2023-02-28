import hashlib
import json
import logging
import os
import time
from pathlib import Path

import librosa
import maad
import numpy as np
# import onnxruntime
import parselmouth
import soundfile
import torch
import torchaudio

from hubert import hubert_model
import utils
from models import SynthesizerTrn

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res

def get_f0(x, p_len,f0_up_key=0):

    time_step = 160 / 16000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, 16000).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    if len(f0) > p_len:
        f0 = f0[:p_len]
    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0

def clean_pitch(input_pitch):
    num_nan = np.sum(input_pitch == 1)
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch


def plt_pitch(input_pitch):
    input_pitch = input_pitch.astype(float)
    input_pitch[input_pitch == 1] = np.nan
    return input_pitch


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return f0_pitch


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)



# class SvcONNXInferModel(object):
#     def __init__(self, hubert_onnx, vits_onnx, config_path):
#         self.config_path = config_path
#         self.vits_onnx = vits_onnx
#         self.hubert_onnx = hubert_onnx
#         self.hubert_onnx_session = onnxruntime.InferenceSession(hubert_onnx, providers=['CUDAExecutionProvider', ])
#         self.inspect_onnx(self.hubert_onnx_session)
#         self.vits_onnx_session = onnxruntime.InferenceSession(vits_onnx, providers=['CUDAExecutionProvider', ])
#         self.inspect_onnx(self.vits_onnx_session)
#         self.hps_ms = utils.get_hparams_from_file(self.config_path)
#         self.target_sample = self.hps_ms.data.sampling_rate
#         self.feature_input = FeatureInput(self.hps_ms.data.sampling_rate, self.hps_ms.data.hop_length)
#
#     @staticmethod
#     def inspect_onnx(session):
#         for i in session.get_inputs():
#             print("name:{}\tshape:{}\tdtype:{}".format(i.name, i.shape, i.type))
#         for i in session.get_outputs():
#             print("name:{}\tshape:{}\tdtype:{}".format(i.name, i.shape, i.type))
#
#     def infer(self, speaker_id, tran, raw_path):
#         sid = np.array([int(speaker_id)], dtype=np.int64)
#         soft, pitch = self.get_unit_pitch(raw_path, tran)
#         pitch = np.expand_dims(pitch, axis=0).astype(np.int64)
#         stn_tst = soft
#         x_tst = np.expand_dims(stn_tst, axis=0)
#         x_tst_lengths = np.array([stn_tst.shape[0]], dtype=np.int64)
#         # 使用ONNX Runtime进行推理
#         start = time.time()
#         audio = self.vits_onnx_session.run(output_names=["audio"],
#                                            input_feed={
#                                                "hidden_unit": x_tst,
#                                                "lengths": x_tst_lengths,
#                                                "pitch": pitch,
#                                                "sid": sid,
#                                            })[0][0, 0]
#         use_time = time.time() - start
#         print("vits_onnx_session.run time:{}".format(use_time))
#         audio = torch.from_numpy(audio)
#         return audio, audio.shape[-1]
#
#     def get_units(self, source, sr):
#         source = torchaudio.functional.resample(source, sr, 16000)
#         if len(source.shape) == 2 and source.shape[1] >= 2:
#             source = torch.mean(source, dim=0).unsqueeze(0)
#         source = source.unsqueeze(0)
#         # 使用ONNX Runtime进行推理
#         start = time.time()
#         units = self.hubert_onnx_session.run(output_names=["embed"],
#                                              input_feed={"source": source.numpy()})[0]
#         use_time = time.time() - start
#         print("hubert_onnx_session.run time:{}".format(use_time))
#         return units
#
#     def transcribe(self, source, sr, length, transform):
#         feature_pit = self.feature_input.compute_f0(source, sr)
#         feature_pit = feature_pit * 2 ** (transform / 12)
#         feature_pit = resize2d_f0(feature_pit, length)
#         coarse_pit = self.feature_input.coarse_f0(feature_pit)
#         return coarse_pit
#
#     def get_unit_pitch(self, in_path, tran):
#         source, sr = torchaudio.load(in_path)
#         soft = self.get_units(source, sr).squeeze(0)
#         input_pitch = self.transcribe(source.numpy()[0], sr, soft.shape[0], tran)
#         return soft, input_pitch


class RealTimeVC:
    def __init__(self):
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = 16000  # 区块长度
        self.pre_len = 3840  # 交叉淡化长度，640的倍数

    """输入输出都是1维numpy 音频波形数组"""

    def process(self, svc_model, speaker_id, f_pitch_change, input_wav_path):
        audio, sr = torchaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return audio[-self.chunk_len:]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)
            audio, sr = svc_model.infer(speaker_id, f_pitch_change, temp_wav)
            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len:]
            self.last_o = audio
            return ret[self.chunk_len:2 * self.chunk_len]
