# 已生成镜头

镜头只放在这个目录。每条镜头目录里有 `shot.json`、`voice.wav`，配音画面在 `candidates/<序号>/voiced.mp4`。

| 目录 | 内容 |
| --- | --- |
| `anime-lora` | 赛璐璐 LoRA 基线，没有角色 LoRA。挥手试验 `deepseek-wave`、`whale-wave-smooth`，以及咖啡馆、夜街、房间、背面。 |
| `char-lora-f49a` | 49 帧，75% 全加噪、25% 设定图锚定。书店、走路各有纯文本和首帧参考，雪地为纯文本。种子 81。权重 `whale_character_r16_f49a_s0800.safetensors`。 |
| `char-lora-f49c` | 49 帧，60% 全加噪、40% 用同一段视频的另一帧作参考。书店、走路、雪地为纯文本，书店和走路另有设定图。种子 81。权重 `whale_character_r16_f49c_s0800.safetensors`。 |
