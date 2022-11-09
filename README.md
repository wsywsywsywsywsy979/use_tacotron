# Tacotron 2 (without wavenet)
PyTorch实现[Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf).

此实现包括**分布式**和**自动混合精度**支持，并使用[LJSpeech数据集](https://keithito.com/LJ-Speech-Dataset/).

分布式和自动混合精度支持依赖于NVIDIA的[Apex]和[AMP]。

使用发布的[Tacotron 2]和[WaveGlow]模型访问我们的[website]获取音频样本。

![对齐，预测梅尔谱图，目标梅尔谱]（tensorboard.png）

## 条件 
1. NVIDIA GPU + CUDA cuDNN

## 设置
1. 下载并提取[LJ语音数据集](https://keithito.com/LJ-Speech-Dataset/)
2. 克隆此存储库：`git clone https://github.com/NVIDIA/tacotron2.git`
3. 进入仓库: `cd tacotron2`
4. 初始化子模块: `git submodule init; git submodule update`
5. 更新.wav路径: `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt`
    - 或者，在`hparams.py`中设置`load_mel_from_disk=True`，并更新mel-spectrogram路径
6. Install [PyTorch 1.0]
7. Install [Apex]
8. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## 训练
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## 使用预训练模型进行训练
使用预训练模型进行训练可以加快收敛速度
默认情况下，依赖于数据集的文本嵌入层为[忽略]

1. 下载发布的[Tacotron 2]模型
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference （其他都是原始仓库的直译，此处是我自己测试了的）
1. 下载[Tacotron 2] model
2. 下载[WaveGlow] model
3. 修改inference_bywsy.py中要修改的路径和text内容，然后运行该py文件

N.b.  当执行Mel频谱图到音频合成时，确保Tacotron 2和Mel解码器在相同的Mel频谱表示上进行训练。

## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow)用于语音合成的比实时更快的基于流的生成网络

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/比实时更快
WaveNet。

## Acknowledgements
此实现使用来自以下存储库的代码：[KeithIto](https://github.com/keithito/tacotron/)，[Prem Seetharaman](https://github.com/pseeth/pytorch-stft)如我们的代码中所述。

我们受到了[Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)Tacotron PyTorch实施的启发。

我们感谢Tacotron 2的论文作者，特别是Jonathan Shen, Yuxuan Wang and Zongheng Yang。

[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp