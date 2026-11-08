# videoGenV2

写实流水线在 `/home/yi.yang/src/videoGenV1`。本目录用固定的 Deepseek 鲸鱼娘，按提示词生成不同主题的短视频。画面是 Wan2.2 TI2V-5B 加动漫 LoRA，配音是 CosyVoice2。口型留到动作稳定之后。

角色图在 `project/deepseek/`：`deepFront.jpg` 是正面，`deepseekBack.jpg` 是背面。一次生成只用其中一张，由镜头的 `view` 决定。身份描述写在 `project/deepseek/character.json`，不交给语言模型重写。

Deepseek 只负责把主题拆成台词、场景和动作。它不生成像素，也不生成波形。

Wan 代码在 `/home/yi.yang/src/videoGenV1/repos/Wan2.2`。环境是 `videogen-v2`，视频权重在 `/gpfs/sjtu/audiocc/home/yi.yang/videogen/models/Wan2.2-TI2V-5B`，动漫 LoRA 在 `/gpfs/sjtu/audiocc/home/yi.yang/videogen-v2/models/lora/`。CosyVoice2 权重在 `/gpfs/sjtu/audiocc/home/yi.yang/videogen/models/CosyVoice2-0.5B`。

## 画面模型：Wan2.2 TI2V-5B

这条流水线用的是 5B 的密集模型，不是 Wan2.2-A14B。A14B 是高噪声专家加低噪声专家的 MoE，总参数约 27B，本目录没有加载它。

TI2V-5B 由三块组成，数字来自 `wan/configs/wan_ti2v_5B.py`：

| 模块 | 本机配置 | 作用 |
| --- | --- | --- |
| umT5-XXL 编码器 | `models_t5_umt5-xxl-enc-bf16.pth`，推理时放在 CPU | 把英文提示编码成交叉注意力条件 |
| Wan2.2 VAE | 时间 × 高 × 宽压缩 4×16×16 | 在 latent 里生成，再解码成像素 |
| DiT | 30 层，宽度 3072，24 头，FFN 14336 | 在 latent 上按 flow matching 去噪 |

patch 是 1×2×2，所以进 DiT 之前，时间再压 1 倍、空间再压 2 倍。算上 VAE，总压缩是 4×32×32。默认 24 fps、121 帧，大约 5 秒。官方支持的尺寸是 `704*1280` 和 `1280*704`。

有参考图时，第一帧用这张图的 VAE latent，后面的帧从噪声开始。没有参考图时，整段都从噪声开始。本目录的鲸鱼娘镜头总是带一张参考图。采样器是 UniPC，分类器无关引导把有条件和无条件的结果做差。配置里的默认是 50 步、guide scale 5；当前镜头用 40 步、guide scale 6。

预训练目标是 flow matching：在 VAE latent 上学习从噪声走向视频的速度场。官方说明里，5B 和 14B MoE 共用升级后的训练数据；5B 本身没有第二套专家。本集群没有重训这个底模。

动漫 LoRA 是现成权重 `wan_2.2_B_anime_v01_000153000.safetensors`。`run_lora.py` 把它的 A、B 矩阵乘进 DiT 的线性层，强度 0.6，只影响这次进程里的副本，不改底模文件。它负责赛璐璐画风。鲸鱼娘的脸和衣服来自参考图，不是这条 LoRA 单独学会的。平滑动作的第二条 LoRA 还没训。

## 语音模型：CosyVoice2-0.5B

配音走 `scripts/tts_one.py` 的 `inference_zero_shot`。参考音是 `/home/yi.yang/src/CosyVoice/asset/zero_shot_prompt.wav`，对应文本是「希望你以后能够做的比我还好呦。」音色来自这句参考，不是为鲸鱼娘单独微调的说话人。`text_frontend` 关闭，台词按原文送进去。采样率 24000。

模型分三段，结构来自权重目录里的 `cosyvoice2.yaml`：

1. 语言模型是 Qwen2-0.5B：24 层，隐藏维 896，14 个查询头，2 个键值头。它不直接出波形，而是预测 6561 类语音 token，末尾再加一个结束符。训练损失是语音 token 的交叉熵，文本 token 不计入损失，并按长度归一。采样用重复感知采样，top-p 0.8、top-k 25。
2. 流匹配解码器把语音 token 变成 80 维梅尔谱。说话人条件是 192 维的 x-vector。编码器是 6 层 Conformer，解码器是因果条件流匹配网络。训练时的 classifier-free guidance 比例是 0.2，推理时是 0.7。
3. HiFT 声码器把梅尔谱变成 24 kHz 波形。它带基频预测和对抗训练；这一步的判别器不参与我们现在的推理。

官方训练顺序是先训语言模型，再训流匹配，最后训声码器的 GAN。yaml 里的监督微调学习率是 1e-5。本集群没有做这一步，也没有做 CosyVoice 的强化学习。

## 按主题出一条片子

```bash
python /home/yi.yang/src/videoGenV2/scripts/plan_shot.py \
  --description "在下午的咖啡馆窗边，用大约一秒慢慢抬手挥一下" \
  --scene "咖啡馆窗边，暖光" \
  --shot cafe-wave
```

背身镜头加 `--view back`。不写 `--view` 时，由模型在 front 和 back 里选。默认竖屏 `704*1280`、121 帧、40 步、guide scale 6、LoRA 强度 0.6，和 deepseek-wave 里看过的设置相同。

然后提交：

```bash
sbatch /home/yi.yang/slurm/videogen-v2-lora.sbatch /home/yi.yang/src/videoGenV2/project/shots/cafe-wave
```

一次作业先合成 `voice.wav`，再加载 Wan 和 LoRA，按种子逐条出画面。画面在 `candidates/1/wan.mp4`，带配音的是 `candidates/1/voiced.mp4`。

筛掉单帧跳变：

```bash
python /home/yi.yang/src/videoGenV2/scripts/score_motion.py \
  --shot /home/yi.yang/src/videoGenV2/project/shots/cafe-wave
```

`snap` 为 true 的片子最大帧差至少是 4。deepseek-wave 的候选 3 会落在这里，候选 2 不会。留下的平滑片子才作为以后换场景和微调的正样本。

## 还原 B 站缓存

把客户端下载的缓存目录放进 `project/deepseek/`。每个目录里有 `videoInfo.json` 和两个 `.m4s`。还原时去掉客户端加在片段开头的 `000000000`，再把画面和声音封装成 mp4，不重新编码。成品用分集标题命名，写在缓存目录的旁边。同名文件已存在就跳过。

```bash
python /home/yi.yang/src/videoGenV2/scripts/restore_bilibili_cache.py \
  /home/yi.yang/src/videoGenV2/project/deepseek
```
