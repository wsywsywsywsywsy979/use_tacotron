import sys
sys.path.append('tacotron2')
import torch
from layers import STFT


class Denoiser(torch.nn.Module):
    """ 从使用waveglow生成的音频中消除模型偏差 """

    def __init__(self, waveglow, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros'):
        super(Denoiser, self).__init__()
        self.stft = STFT(filter_length=filter_length, # 1024
                         hop_length=int(filter_length/n_overlap),# 1024/4
                         win_length=win_length).cuda() # 1024
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype, # float16
                device=waveglow.upsample.weight.device) # 'cuda'
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
