"""Attach view, action, scene, and a training caption to each clip.

Pixel clustering only separates white backgrounds from busy real rooms.
These labels come from the mid-frame of each clean clip and from the
known contents of the noisy leftovers.
"""

import json
from pathlib import Path

CLIPS = Path("/home/yi.yang/src/videoGenV2/project/deepseek/video_data_raw/clips")
PREFIX = "蓝色长发蓝眼睛的鲸鱼娘，女仆装，围裙上有鲸鱼图案，赛璐璐动画。"

# file -> action, view, scene, caption body
CLEAN = {
    "train_data1_02.mp4": (
        "站立表情", "正面", "实拍房间",
        "正面全身，站在床上，生气地张嘴，双手握在身前。镜头固定，背景是真实房间。",
    ),
    "train_data1_03.mp4": (
        "站立表情", "正面", "实拍房间",
        "正面全身，站在床上，生气地抿嘴。镜头固定，背景是真实房间。",
    ),
    "train_data2_01.mp4": (
        "站立表情", "正面", "实拍房间",
        "正面全身，站在明亮的房间里，手里拿着蓝色玩偶。镜头固定。",
    ),
    "train_data2_02.mp4": (
        "站立表情", "正面", "实拍房间",
        "正面全身，站着把白杯子送到嘴边。镜头固定，背景是真实房间。",
    ),
    "train_data2_04.mp4": (
        "坐着", "正面", "实拍房间",
        "正面全身，坐在地板上，手里拿着白杯子。镜头固定。",
    ),
    "train_data3_01.mp4": (
        "坐着", "正面", "实拍房间",
        "正面全身，坐在灰色沙发上，表情惊讶。镜头固定。",
    ),
    "train_data3_02.mp4": (
        "坐着", "正面", "实拍房间",
        "正面全身，坐在沙发上，轻轻微笑。镜头固定。",
    ),
    "train_data3_03.mp4": (
        "坐着", "正面", "实拍房间",
        "正面全身，坐在沙发上吃白色食物。镜头固定。",
    ),
    "train_data4_01.mp4": (
        "坐着", "正面", "实拍房间",
        "正面全身，坐在木地板上吃东西，动作很小。镜头固定。",
    ),
    "train_data5_03.mp4": (
        "坐着", "正面", "实拍房间",
        "正面全身，坐在沙发上，手里拿着蓝色小物件。镜头固定。",
    ),
    "seedance_wave_01.mp4": (
        "挥手", "正面", "白底",
        "正面全身，站在白背景前挥手。镜头固定。",
    ),
    "seedance_stand_01.mp4": (
        "走路", "正面", "白底",
        "正面全身，在白背景前小步走路。镜头固定。",
    ),
    "seedance_back_01.mp4": (
        "背面或转身", "背面或转身", "白底",
        "先是背面全身，再侧过脸回头看。白背景，镜头固定。",
    ),
    "seedance_speak_01.mp4": (
        "说话手势", "正面", "白底",
        "正面全身，在白背景前张嘴说话。镜头固定。",
    ),
    "seedance_cafe_01.mp4": (
        "走路", "正面", "赛璐璐场景",
        "正面全身，在空的咖啡馆里朝镜头走来，然后微笑。镜头固定。",
    ),
    "seedance_night_01.mp4": (
        "背面或转身", "背面或转身", "赛璐璐场景",
        "夜里路灯下，从正面走到背面，再侧身回头。镜头固定。",
    ),
    "seedance_room_01.mp4": (
        "说话手势", "正面", "赛璐璐场景",
        "正面全身，在房间里眨眼，做手势，看向杯子，然后微笑。镜头固定。",
    ),
    "seedance_hall_01.mp4": (
        "背面或转身", "背面或转身", "赛璐璐场景",
        "背面全身，沿着走廊往前走，结尾侧过身回头。镜头固定。",
    ),
    "paiwo_sit_01.mp4": (
        "坐下再站起", "正面", "赛璐璐场景",
        "正面全身，从站着坐到地上，再站起来。镜头固定。",
    ),
    "ref_stand_01.mp4": (
        "站立表情", "正面", "赛璐璐场景",
        "正面全身，站在有窗的房间里，双手放在身前，动作幅度小。镜头固定。",
    ),
    "ref_back_01.mp4": (
        "背面或转身", "背面或转身", "赛璐璐场景",
        "背面全身，沿着图书馆书架过道向前走，不回头。镜头固定。",
    ),
    "ref_speak_01.mp4": (
        "说话手势", "正面", "赛璐璐场景",
        "正面全身，站在窗边张嘴说话，双手在胸前，然后微笑。镜头固定。",
    ),
    "ref_sit_01.mp4": (
        "坐下再站起", "正面", "赛璐璐场景",
        "正面全身，在木质阳台上从站着坐下，再站起来。镜头固定。",
    ),
    "ref_plush_01.mp4": (
        "自己拿东西", "正面", "赛璐璐场景",
        "正面全身，在阳台上看着地板上的一只蓝色小鲸鱼，弯腰用双手捧起来，抱在胸口。地板上不再留下另一只。镜头固定。",
    ),
    "ref_side_01.mp4": (
        "侧身站立", "侧面", "赛璐璐场景",
        "左侧面全身，站在木地板走廊的窗边，脸朝画面左侧，只看见一只眼睛，没有迈步。镜头固定。",
    ),
    "seedance_side_01.mp4": (
        "侧身走", "侧面", "赛璐璐场景",
        "左侧面全身，在海边木栈道上朝画面左侧走，脸保持侧面，然后停住。镜头固定。",
    ),
    "seedance_backwalk_01.mp4": (
        "背面或转身", "背面或转身", "赛璐璐场景",
        "背面全身，在画廊里向画面左侧走，不回头。走的过程中人略微变小。镜头固定。",
    ),
    "seedance_flower_01.mp4": (
        "侧身拿花", "侧面", "赛璐璐场景",
        "左侧面全身，在花店里从铁桶抽出一枝白花，双手捧在身前看着。始终保持侧面。镜头固定。",
    ),
    "seedance_grass_01.mp4": (
        "蹲下系鞋带", "正面", "赛璐璐场景",
        "正面全身，在草坪上从站着蹲下给右脚系鞋带，再站起来。镜头固定。",
    ),
    "seedance_kitchen_01.mp4": (
        "背面洗碗", "背面或转身", "赛璐璐场景",
        "背面全身，站在厨房水槽前拿起一只白杯子冲洗，再放回台面，不回头。人的大小基本不变。镜头固定。",
    ),
    "seedance_piano_01.mp4": (
        "按琴键", "正面", "赛璐璐场景",
        "正面全身，站在立式钢琴旁，右手按下中间的一个白键。镜头固定。",
    ),
    "seedance_greenhouse_01.mp4": (
        "背面浇水", "背面或转身", "赛璐璐场景",
        "背面全身，在温室木架前提起洒水壶给植物浇水，不回头。人的大小基本不变。镜头固定。",
    ),
    "seedance_leaf_01.mp4": (
        "接落叶", "正面", "赛璐璐场景",
        "正面全身，站在银杏道上接住一片黄叶，拿到眼前看。双脚不迈步。镜头固定。",
    ),
    "seedance_swing_01.mp4": (
        "荡秋千", "侧面", "赛璐璐场景",
        "左侧面，坐在秋千上抓住锁链荡动。画面上下裁掉了一部分身体。镜头固定。",
    ),
    "seedance_arcade_01.mp4": (
        "侧身走", "侧面", "赛璐璐场景",
        "左侧面全身，在老街骑楼里向画面左侧走，手臂自然摆动，不转向镜头。镜头固定。",
    ),
    "seedance_colonnade_01.mp4": (
        "走路", "正面", "赛璐璐场景",
        "正面全身，在拱柱回廊里朝镜头走几步，然后停住。镜头固定。",
    ),
    "seedance_pier_01.mp4": (
        "背面或转身", "背面或转身", "赛璐璐场景",
        "背面全身，在黄昏木桥上向前走，不回头。走的过程中人变小。镜头固定。",
    ),
    "seedance_bridgewave_01.mp4": (
        "挥手", "正面", "赛璐璐场景",
        "正面全身，站在石桥栏杆边，右手在脸旁挥一次，再放下。双脚不迈步。镜头固定。",
    ),
}

NOISY_BODY = "在真实房间里活动。画面里可能出现别人的手、第二个人或道具。"


def main():
    rows = json.loads((CLIPS / "features.json").read_text(encoding="utf-8"))
    for row in rows:
        name = row["file"]
        if row["tier"] == "clean":
            action, view, scene, body = CLEAN[name]
            weight = 3.0
        else:
            action, view, scene, body = "复杂实拍", "正面", "实拍房间", NOISY_BODY
            weight = 1.0
        row["action"] = action
        row["view"] = view
        row["scene"] = scene
        row["caption"] = PREFIX + body
        row["sample_weight"] = round(row["seconds"] * weight, 4)
    (CLIPS / "features.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def total(pred):
        buckets = {}
        for row in rows:
            if not pred(row):
                continue
            buckets[row["action"]] = buckets.get(row["action"], 0.0) + row["seconds"]
        return {k: round(v, 3) for k, v in buckets.items()}

    clean = [r for r in rows if r["tier"] == "clean"]
    summary = {
        "clean_seconds": round(sum(r["seconds"] for r in clean), 3),
        "noisy_seconds": round(sum(r["seconds"] for r in rows if r["tier"] == "noisy"), 3),
        "clean_action": {},
        "clean_view": {},
        "clean_scene": {},
    }
    for key in ("action", "view", "scene"):
        buckets = {}
        for row in clean:
            buckets[row[key]] = buckets.get(row[key], 0.0) + row["seconds"]
        summary["clean_" + key] = {k: round(v, 3) for k, v in sorted(buckets.items())}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
