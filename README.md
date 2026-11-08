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

动漫 LoRA 是现成权重 `wan_2.2_B_anime_v01_000153000.safetensors`。`run_lora.py` 把它的 A、B 矩阵乘进 DiT 的线性层，强度 0.6，只影响这次进程里的副本，不改底模文件。它负责赛璐璐画风。鲸鱼娘的脸和衣服来自参考图，不是这条 LoRA 单独学会的。角色 LoRA 另训，不和这条风格 LoRA 叠在同一次生成里。做法和现有数据见下面两节。

## 角色训练数据

视频不进 git。清单在 `project/deepseek/video_data_raw/clips/`：`clips.json` 是切段，`features.json` 是每段的时长、动作、机位、场景、字幕和抽样权重。统计写于 2026-11-08，共 37 段。

干净段 25 条，合计 179.9 秒，约 3.0 分钟。画面里只有鲸鱼娘，没有别人的手、第二个人、字幕和水印。噪声段 12 条，合计 264.3 秒，约 4.4 分钟，来自 B 站成片里切剩下的实拍房间，里面会有别人的手或第二个人。训练时干净段的权重是时长乘 3，噪声段是时长乘 1。

干净段按动作：

| 动作 | 时长 | 现在的情况 |
| --- | --- | --- |
| 正面坐着 | 37.3 秒 | 沙发、地板、吃东西，已经够用 |
| 正面站立表情 | 35.7 秒 | 其中 25.7 秒是实拍房间，10.1 秒是更 Q 的赛璐璐站立 |
| 背面或转身 | 27.3 秒 | 多数走到一半会转身；一直保持背面、人也不走远的只有图书馆那 5.1 秒 |
| 张嘴说话 | 22.2 秒 | 窗边一条会张嘴，手会抬到胸前 |
| 坐下再站起 | 19.1 秒 | 室内一条，木质阳台一条 |
| 正面走路 | 13.1 秒 | 白底小步，加咖啡馆朝镜头走近 |
| 侧身站立 | 10.1 秒 | 走廊左侧面站住，头发在动，脚没有迈步 |
| 自己拿东西 | 9.1 秒 | 阳台上一只蓝色小鲸鱼，开头那一秒的重复玩偶已切掉 |
| 挥手 | 6.1 秒 | 白底 |
| 侧身走 | 0 秒 | 还没有 |

机位上，正面 142.6 秒，背面或转身 27.3 秒，侧面 10.1 秒。场景上，赛璐璐 96.7 秒，实拍房间 62.9 秒，白底 20.3 秒。

正式角色训练按 5 分钟干净视频估，还差大约 2 分钟。优先补的是侧身走、保持大小的背面走、正面走路，其次是再一条自己拿东西和一条更清楚的口型。短试点要的 2–3 分钟干净时长已经够。

`features.json` 里还有按运动幅度和白底比例分成的 4 个像素簇。它们把白底和实拍房间大致分开，分不出侧身、走路和说话。看缺什么以这张动作表为准。

## 角色 LoRA 怎么训

脚本是 `scripts/train_character_lora.py`，提交文件是 `/home/yi.yang/slurm/videogen-v2-lora-train.sbatch`。底模仍是 TI2V-5B。T5、VAE 和 DiT 原权重都冻结。只训练秩 16 的 LoRA，加在每一层的 self-attention、cross-attention 的 q、k、v、o，以及 FFN 的两层线性上。不加载现成的动漫 LoRA。

一条样本是片段里随机的 17 帧，缩到 480×832。17 等于 4n+1，边长能被 32 整除，VAE 才能编码。像素先变到 -1 到 1，再编成 latent。flow matching 用 shift 5：`x_t = (1-sigma) * 干净latent + sigma * 噪声`，网络预测的速度是 `噪声 - 干净latent`，损失是被 mask 过的均方误差。第一帧 latent 保持干净，对应 token 的时间步是 0，不计入损失，和推理时锁第一帧的方式一致。

文本是 `features.json` 里的中文 `caption`，开训前用 CPU 上的 umT5 编一次然后卸掉 T5。每步有 10% 的概率丢掉这句，留给以后的引导。优化器只看 LoRA 参数，AdamW，学习率 1e-4。注意力块用梯度检查点。权重存成 `diffusion_model.*.lora_A/B`，`run_lora.py` 可以直接读，推理时 scale 用 1.0。

抽样只看 `sample_weight`。动作名、机位、场景和聚类编号不另外参与损失，所以坐着 37 秒被抽到的次数仍然多于侧面 10 秒。当前试点是 120 步，用来确认损失是有限的、LoRA 能收到梯度。这一轮还不能说明角色已经学会。

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
