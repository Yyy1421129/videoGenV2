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

有参考图时，第一帧用这张图的 VAE latent，后面的帧从噪声开始。没有参考图时，整段都从噪声开始。按主题出片时仍带一张参考图。角色 LoRA 的检查可以在 `shot.json` 里写 `"text_only": true`，这时不传图，第一帧也不会被钉住。采样器是 UniPC，分类器无关引导把有条件和无条件的结果做差。配置里的默认是 50 步、guide scale 5；当前镜头用 40 步、guide scale 6。

预训练目标是 flow matching：在 VAE latent 上学习从噪声走向视频的速度场。官方说明里，5B 和 14B MoE 共用升级后的训练数据；5B 本身没有第二套专家。本集群没有重训这个底模。

动漫 LoRA 是现成权重 `wan_2.2_B_anime_v01_000153000.safetensors`。`run_lora.py` 把它的 A、B 矩阵乘进 DiT 的线性层，强度 0.6，只影响这次进程里的副本，不改底模文件。它负责赛璐璐画风。鲸鱼娘的脸和衣服来自参考图，不是这条 LoRA 单独学会的。角色 LoRA 另训，不和这条风格 LoRA 叠在同一次生成里。做法和现有数据见下面两节。

## 角色训练数据

训练片段不进 git。清单在 `project/deepseek/video_data_raw/clips/`：`clips.json` 是切段，`features.json` 是每段的时长、动作、机位、场景、字幕和抽样权重。统计写于 2026-11-12，共 54 段。生成结果在仓库的 `shots/`，和训练片段分开。

干净段 42 条，合计 333.6 秒，约 5.6 分钟。画面里只有鲸鱼娘，没有别人的手、第二个人、字幕和水印。噪声段 12 条，合计 264.3 秒，来自 B 站成片里切剩下的实拍房间，里面会有别人的手或第二个人。训练时干净段的权重是时长乘 3，噪声段是时长乘 1。当前训练只用干净段。

2026-11-12 新进四条：侧面拿面包 10.0 秒，背面蹲下再站起 9.0 秒，正面双手拿木盒 10.0 秒，正面拿苹果 9.0 秒。

干净段按动作：

| 动作 | 时长 | 现在的情况 |
| --- | --- | --- |
| 背面或转身 | 47.4 秒 | 转身，以及图书馆、画廊、黄昏木桥上的背面走。这几条走都会让人变小 |
| 正面坐着 | 37.3 秒 | 已经够用 |
| 正面站立表情 | 35.7 秒 | 已经够用 |
| 正面走路 | 23.2 秒 | 白底、咖啡馆，加拱柱回廊里朝镜头走 |
| 张嘴说话 | 22.2 秒 | 够用 |
| 侧身走 | 19.1 秒 | 海边栈道 12 秒，老街骑楼 7 秒。还少 |
| 坐下再站起 | 19.1 秒 | 够用 |
| 挥手 | 14.1 秒 | 白底一条，石桥栏杆一条 |
| 背面洗碗 | 11.0 秒 | 站在原地，人的大小基本不变 |
| 侧身站立 | 10.1 秒 | 脚没有迈步 |
| 侧身拿面包、双手拿木盒 | 各 10.0 秒 | 2026-11-12 补的手部接触 |
| 背面浇水、背面蹲下、拿起苹果、自己拿东西 | 9.0–9.1 秒 | 蹲下是背面蹲下再站直，人没有变小 |
| 按琴键、接落叶、蹲下系鞋带 | 各 8.0 秒 | 各有一条 |
| 侧身拿花、荡秋千 | 各 7.0 秒 | 秋千上下裁掉了一部分身体 |

机位上，正面 203.8 秒，背面或转身 76.5 秒，侧面 53.3 秒。场景上，赛璐璐 250.4 秒，实拍房间 62.9 秒，白底 20.3 秒。

干净视频已经超过 5 分钟。还缺的是侧身走，以及大小保持不变的背面走。手部接触有三条，背面蹲起有一条，下一轮训练会用上。

`features.json` 里还有按运动幅度和白底比例分成的 4 个像素簇。它们把白底和实拍房间大致分开，分不出侧身、走路和说话。看缺什么以这张动作表为准。

## 角色 LoRA 怎么训

脚本是 `scripts/train_character_lora.py`，提交文件是 `/home/yi.yang/slurm/videogen-v2-lora-train.sbatch`。底模仍是 TI2V-5B。T5、VAE 和 DiT 原权重都冻结。只训练秩 16 的 LoRA，加在每一层的 self-attention、cross-attention 的 q、k、v、o，以及 FFN 的两层线性上，共 300 层、约 4030 万个可训练参数。不加载现成的动漫 LoRA。

一条样本是片段里的 49 帧，缩到 480×832。49 等于 4n+1，边长能被 32 整除，VAE 才能编码。像素先变到 -1 到 1，再编成 latent。flow matching 用 shift 5：`x_t = (1-sigma) * 干净latent + sigma * 噪声`，网络预测的速度是 `噪声 - 干净latent`，损失是全部帧的均方误差。第一帧也加噪，也计入损失。窗口按帧号截取，不按时间戳，避免片段末尾少一帧时整次训练退出。

起点按这段视频里的帧差来抽。脚本先把片段缩成小图，每隔 4 帧取一个候选起点，窗口内变化越大，被抽到的机会越大；静止片段仍留一个很小的权重。片段本身还是按 `sample_weight` 抽，所以长的干净段仍然更常出现。动作名、机位、场景和聚类编号不进损失。

更早的训练把第一帧 latent 留成干净、时间步设为 0，并不计入损失。那样和带参考图的推理一致，但步数加到 800 以后，人几乎停在参考图的姿势上。当前脚本不锁第一帧。`run_lora.py` 在传入参考图时仍会锁第一帧；`shot.json` 写了 `text_only` 才不传图。

文本是 `features.json` 里的中文 `caption`，开训前用 CPU 上的 umT5 编一次然后卸掉 T5。每步有 10% 的概率丢掉这句，留给以后的引导。优化器只看 LoRA 参数，AdamW，学习率 1e-4。注意力块用梯度检查点。DiT 是 bfloat16，注意力块会把一部分激活转成 float32；LoRA 在相乘前把底模权重转到激活的 dtype，A、B 仍用 float32 参数。权重存成 `diffusion_model.*.lora_A/B`，`run_lora.py` 可以直接读。角色 LoRA 单独用时 scale 设为 1.0。每 200 步另存一个文件，不覆盖上一份。

当前提交用干净段、800 步，输出 `whale_character_r16_f49.safetensors`。它和下面的 33 帧权重分开存，不从那一份继续训。

## 上次的角色 LoRA 结果

三次检查用同一组中文提示词和种子 81：书店里指向高处的书，雨中月台侧面撑伞，雪地背面蹲下拢雪再站起。只加角色 LoRA，scale 1.0，不叠赛璐璐 LoRA。

120 步、锁定第一帧，权重 `whale_character_r16.safetensors`，视频在 `shots/char-lora-s120/`。人像这只鲸鱼娘，主要是因为推理钉住了参考图。书店的指向很弱，月台的伞旁多出一只手，雪地里腿消失。

800 步、仍然锁定第一帧，权重 `whale_character_r16_800_s0800.safetensors`，视频在 `shots/char-lora-s800/`。损失从大约 0.19 降到 0.155。生成里人更僵，变的主要是背景。这份权重不再续训。

33 帧、不锁第一帧，800 步已经跑完。当时用 38 条干净片段、295.4 秒。四份权重是 `whale_character_r16_f33b_s0200.safetensors` 到 `_s0800`，每份 300 层。800 步都有梯度。损失 1–100 步平均 0.301，601–800 步 0.286，几乎不降。这个数比锁定第一帧的高，因为每一帧都在损失里，两边不能直接比。

用 `_s0800` 做了纯文本生成，不传参考图。视频在 `shots/char-lora-f33b/`：

- 书店：蓝发、女仆装、鲸鱼围裙都在。身体 5 秒几乎不动，书浮在肩膀旁边，指尖插进书里。
- 月台：第一帧就已经侧身撑着伞，之后接近静帧。提示词要求站着不迈步，所以这条看不出会不会走。
- 雪地：背面、服装和身体大小稳定。没有蹲下去，雪从一开始就在手里。

身份可以不靠参考图出现。动作仍然短：33 帧大约只有 1.4 秒，而推理要生成 5 秒。下一轮改成 49 帧，并优先抽动作大的窗口，同时用上 2026-11-12 的拿手和背面蹲下。这一轮还没有生成结果。

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
sbatch /home/yi.yang/slurm/videogen-v2-lora.sbatch /home/yi.yang/src/videoGenV2/shots/cafe-wave
```

一次作业先合成 `voice.wav`，再加载 Wan 和 LoRA，按种子逐条出画面。画面在 `candidates/1/wan.mp4`，带配音的是 `candidates/1/voiced.mp4`。

筛掉单帧跳变：

```bash
python /home/yi.yang/src/videoGenV2/scripts/score_motion.py \
  --shot /home/yi.yang/src/videoGenV2/shots/cafe-wave
```

`snap` 为 true 的片子最大帧差至少是 4。deepseek-wave 的候选 3 会落在这里，候选 2 不会。留下的平滑片子才作为以后换场景和微调的正样本。

生成结果只写在仓库的 `shots/`。`plan_shot.py` 新建的镜头在 `shots/<镜头名>`。已经看过的片子按权重分在子目录里：

- `anime-lora`：现成赛璐璐 LoRA 的挥手试验和四条场景。
- `char-lora-s120`：第一版角色 LoRA，120 步，锁定第一帧。
- `char-lora-s800`：第二版角色 LoRA，800 步，仍然锁定第一帧。同样的三条中文提示词和种子 81，人物基本不动。
- `char-lora-f33b`：33 帧、第一帧也计入损失的 800 步权重，纯文本生成。书店、月台、雪地，种子 81。

## 还原 B 站缓存

把客户端下载的缓存目录放进 `project/deepseek/`。每个目录里有 `videoInfo.json` 和两个 `.m4s`。还原时去掉客户端加在片段开头的 `000000000`，再把画面和声音封装成 mp4，不重新编码。成品用分集标题命名，写在缓存目录的旁边。同名文件已存在就跳过。

```bash
python /home/yi.yang/src/videoGenV2/scripts/restore_bilibili_cache.py \
  /home/yi.yang/src/videoGenV2/project/deepseek
```
