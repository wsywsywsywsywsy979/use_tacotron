from hparams import create_hparams
from text import text_to_sequence
from train import load_model
import torch
import numpy as np
import os 
import scipy.io.wavfile as wav
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig # wsy add
import sys
sys.path.append('waveglow/')
def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        # axes[i].imshow(data[i], aspect='auto', origin='bottom', 
        #                interpolation='none') # 报错：ValueError: 'bottom' is not a valid value for origin; supported values are 'upper', 'lower'
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')  # wsy fix
    savefig("1.jpg") # wsy add 
# 设置 hparams
hparams = create_hparams()
hparams.sampling_rate = 22050
MAX_WAV_VALUE=32768.0  # add
# 从检查点加载模型
check_point=r"/home/wangshiyao/wangshiyao_space/exp6/tacotron2-master/tacotron2_statedict.pt"
model=load_model(hparams)
model.load_state_dict(torch.load(check_point)['state_dict'])
_=model.cuda().eval() # fix 把half()去掉了

# 加载WaveGlow用于mel2audio 合成和去噪 
waveglow_path=r"/home/wangshiyao/wangshiyao_space/exp6/tacotron2-master/waveglow_256channels_universal_v5.pt"
waveglow=torch.load(waveglow_path)['model'] 
waveglow.cuda().eval() # fix 把half()去掉了
for k in waveglow.convinv:
    k.float()

# 准备文本输入
text = "I'm wangshiyao, I like everything!"
sequence=np.array(text_to_sequence(text,['english_cleaners']))[None,:]
sequence=torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

# 解码文本输入并打印结果
mel_outputs,mel_outputs_postnet,_,alignments=model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],mel_outputs_postnet.float().data.cpu().numpy()[0],alignments.float().data.cpu().numpy()[0].T))

output_dir=r"/home/wangshiyao/wangshiyao_space/exp6/tacotron2-master/result" # add
# 使用WaveGlow从频谱图合成音频:
with torch.no_grad():
    audio1=waveglow.infer(mel_outputs_postnet,sigma=0.666)
    audio=audio1*MAX_WAV_VALUE # add 
#---------------add -----------------------------------
audio=audio.squeeze()
audio2=audio.cpu().numpy()
audio=audio2.astype('int16') # 试一下把这一步去掉，看是什么效果
# audio=audio2
audio_path=os.path.join(output_dir,"1.wav")
wav.write(audio_path,hparams.sampling_rate,audio)
#------------------------------------------------------
"""
运行时的主观日志：
    tacotron直接运行会卡住，但调试运行，就很快，这可不行
"""
