# 已生成镜头

镜头只放在这个目录。`scripts/plan_shot.py` 会把新镜头直接写到 `shots/<镜头名>`。下面的子目录是已经看过的结果。

| 目录 | 内容 |
| --- | --- |
| `anime-lora` | 现成赛璐璐 LoRA 的生成。权重不在本仓库。包括挥手试验 `deepseek-wave`、`whale-wave-smooth`，以及咖啡馆、夜街、房间、背面四条场景。 |
| `char-lora-s120` | 第一版角色 LoRA，120 步，锁定第一帧。书店指向、月台打伞、雪地蹲下，种子 81，只加角色 LoRA，scale 1.0。权重 `whale_character_r16.safetensors`。 |
| `char-lora-s800` | 第二版角色 LoRA，800 步，仍然锁定第一帧。提示词和种子与第一版相同。人物几乎不动，只换了背景。权重 `whale_character_r16_800_s0800.safetensors`。 |
| `char-lora-f33b` | 33 帧训练、第一帧也计入损失。纯文本生成，不锁第一帧。书店的手不自然，月台几乎不动，雪地背面站姿稳定。权重 `whale_character_r16_f33b_s0800.safetensors`。 |

每条镜头目录里有 `shot.json`、`voice.wav`，画面在 `candidates/<序号>/wan.mp4`，配音合成后是 `voiced.mp4`。
