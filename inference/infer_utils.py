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


def clean_pitch(input_pitch):
    num_nan = np.sum(input_pitch == 1)
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch


def get_f0(x, p_len, f0_up_key=0):
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
    pad_size = (p_len - len(f0) + 1) // 2
    if (pad_size > 0 or p_len - len(f0) - pad_size > 0):
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0


class Svc(object):
    def __init__(self, net_g_path, config_path, hubert_path="hubert/hubert-soft-0d54a1f4.pt",
                 onnx=False):
        self.onnx = onnx
        self.net_g_path = net_g_path
        self.hubert_path = hubert_path
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.speakers = {}
        for spk, sid in self.hps_ms.spk.items():
            self.speakers[sid] = spk
        self.spk2id = self.hps_ms.spk
        # 加载hubert
        self.hubert_soft = hubert_model.hubert_soft(hubert_path)
        if torch.cuda.is_available():
            self.hubert_soft = self.hubert_soft.cuda()
        self.load_model()

    def load_model(self):
        # 获取模型配置
        if self.onnx:
            raise NotImplementedError
            # self.net_g_ms = SynthesizerTrnForONNX(
            #     178,
            #     self.hps_ms.data.filter_length // 2 + 1,
            #     self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            #     n_speakers=self.hps_ms.data.n_speakers,
            #     **self.hps_ms.model)
            # _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        else:
            self.net_g_ms = SynthesizerTrn(
                self.hps_ms.data.filter_length // 2 + 1,
                self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
                **self.hps_ms.model)
            _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)

    def get_units(self, source, sr):

        source = source.unsqueeze(0).to(self.dev)
        with torch.inference_mode():
            start = time.time()
            units = self.hubert_soft.units(source)
            use_time = time.time() - start
            print("hubert use time:{}".format(use_time))
            return units

    def get_unit_pitch(self, in_path, tran):
        source, sr = torchaudio.load(in_path)
        source = torchaudio.functional.resample(source, sr, 16000)
        if len(source.shape) == 2 and source.shape[1] >= 2:
            source = torch.mean(source, dim=0).unsqueeze(0)
        soft = self.get_units(source, sr).squeeze(0).cpu().numpy()
        f0_coarse, f0 = get_f0(source.cpu().numpy()[0], soft.shape[0] * 2, tran)
        return soft, f0

    def infer(self, speaker_id, tran, raw_path):
        if type(speaker_id) == str:
            speaker_id = self.spk2id[speaker_id]
        sid = torch.LongTensor([int(speaker_id)]).to(self.dev).unsqueeze(0)
        soft, pitch = self.get_unit_pitch(raw_path, tran)
        f0 = torch.FloatTensor(clean_pitch(pitch)).unsqueeze(0).to(self.dev)
        if "half" in self.net_g_path and torch.cuda.is_available():
            stn_tst = torch.HalfTensor(soft)
        else:
            stn_tst = torch.FloatTensor(soft)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.dev)
            start = time.time()
            x_tst = torch.repeat_interleave(x_tst, repeats=2, dim=1).transpose(1, 2)
            audio = self.net_g_ms.infer(x_tst, f0=f0, g=sid)[0, 0].data.float()
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1]
