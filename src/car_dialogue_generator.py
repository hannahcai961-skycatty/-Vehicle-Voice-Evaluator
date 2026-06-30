"""
车载对话数据集生成器 - 含LLM多轮query生成
生成多轮对话，场景和user_query均由LLM生成以保证多样性
使用通义千问 / Kimi 等 DashScope 兼容模型
"""

import json
import random
import time
from openai import OpenAI
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ── 配置区（可通过 .env 覆盖）────────────────────────────────────────────────
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("LLM_MODEL_NAME", "qwen3.5-plus")
'''
TOTAL_CONVERSATIONS = 1 # 先设成1测试，成功后再改大（例如500）  
'''
TIER_CONFIG = [
    {"count": 0,"turn_range": (2, 5)},
    {"count": 0, "turn_range": (5, 10)},
    {"count": 2, "turn_range": (10, 20)},
]
OUTPUT_FILE = str(PROJECT_ROOT / "data" / "input" / os.getenv("OUTPUT_DATASET", "car_dialogue_dataset_generated.json"))

MAX_RETRIES = 3
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 数据定义（保持不变）
INTENTS = [
    "vehicle_basic_info", "real_time_vehicle_status", "location_and_navigation", "time_and_schedule",
    "user_preference_and_habit", "cabin_perception", "entertainment_media", "cabin_internal_perception", "communication"
]

SLOT_MAP = {
    "real_time_vehicle_status": [
        "speed_kmh", "range_remaining_km", "ac.target_function", "ac.position",
        "ac_temperature_setting_celsius", "windows.target_function", "windows_status",
        "light_status", "ventilation_mode", "temperature_celsius"
    ],
    "location_and_navigation": [
        "destination", "destination_name", "destination_type",
        "distance_to_destination_km", "time_to_destination_minutes",
        "route_total_distance_km", "waypoint", "poi_search_results",
        "traffic_condition", "traffic_event"
    ],
    "time_and_schedule": [
        "schedule.event", "schedule.date", "schedule.location",
        "schedule_type", "schedule_status", "reminder_time_minutes_before",
        "is_working_day", "is_holiday"
    ],
    "user_preference_and_habit": [
        "preferred_driving_style", "preferred_ac_temperature_celsius",
        "preferred_ventilation_mode", "preferred_music_genres",
        "preferred_radio_stations", "preferred_podcasts", "preferred_audiobooks",
        "work_location", "home_location", "preferred_streaming_platform",
        "spatial_audio_mode", "mood_based_music", "drive_scene_audio_boost"
    ],
    "entertainment_media": [
        "audio.target_function", "audio.position", "audio.set_type", "audio.value",
        "current_media_type", "current_track_title", "current_track_artist",
        "volume_level_percent", "preferred_music_genres", "preferred_streaming_platform",
        "spatial_audio_mode", "mood_based_music", "drive_scene_audio_boost",
        "music_behavior_on_call"
    ],
    "cabin_perception": [
        "weather.type", "visibility", "road_surface.type", "temperature_celsius",
        "wind_speed_kmh", "traffic_condition", "traffic_event"
    ],
    "cabin_internal_perception": [
        "gender", "age", "safety_seats", "non_dangerous behavior", "emotion"
    ],
    "vehicle_basic_info": [
        "vehicle_id", "vehicle_model"
    ]
}

DOMAIN_LIST = ["导航", "媒体", "车辆控制", "时间日程", "外部信息", "用户偏好"]

# ── 结构化信息生成 ────────────────────────────────────────────────────────────
def build_conversation_skeleton(conv_id: int, turn_range: tuple) -> dict:
    domains = random.sample(DOMAIN_LIST, k=random.randint(2, 4))
    turn_count = random.randint(turn_range[0], turn_range[1])  # 替换原来的 random.choices 那行
    

    turns = []
    for i in range(1, turn_count + 1):
        intent_count = random.choices([1, 2], weights=[70, 30])[0]
        non_comm_intents = [it for it in INTENTS if it != "communication"]
        selected_intents = random.sample(non_comm_intents, k=intent_count)

        all_attrs = []
        for it in selected_intents:
            if it in SLOT_MAP:
                pool = SLOT_MAP[it]
                all_attrs.extend(random.sample(pool, k=min(2, len(pool))))

        reference = "无指代"
        if i > 1 and random.random() > 0.5:
            reference = f"可能对第{i-1}轮内容有指代或追问"

        turns.append({
            "turn": i,
            "ground_truth_intent": " + ".join(selected_intents),
            "ground_slots": {"attribute": " / ".join(all_attrs)},
            "reference_hint": reference,
            "user_query": ""
        })

    return {
        "conv_id": f"C{str(conv_id).zfill(3)}",
        "domain": " + ".join(domains),
        "scenario": "",
        "turns": turns
    }

# ── Prompt ─────────────────────────────────────────────────────────────────────

SCENARIO_SYSTEM_PROMPT = """你是一个车载语音交互数据集的场景生成专家。

任务：每次调用时，根据传入的【领域】和【intent + slots】（可能多轮），生成一个高度真实的、适合车载语音对话的完整场景描述（scenario）。

用户人设：
物设定：
- 核心标签：48岁专业野外摄影师、野生动植物专家
- 常驻城市：成都
- 家庭状况：已婚（子女已成年或独立，家庭负担较轻）
- 职业：自由/专业野生动植物摄影师与专家（兼职科普或拍摄任务）
- 车辆类型：硬派越野SUV或改装越野车（强调通过性、空间、设备固定）
- 核心用车场景：用车时间自由（无固定上下班），以周末/工作日灵活出发的长途自驾 + 野外拍摄任务为主，经常进川西、川南山区或其他自然保护区，拍摄任务中长途占比高，可避开节假日高峰
- 用车核心属性：长途可靠性 + 越野能力 + 野外生存支持 + 专业拍摄辅助，依赖车载助手查询天气、路况、补给点、航班/火车备选、信号覆盖等，痛点包括野外信号弱、天气多变、设备保护、补给规划、安全预警
- 代表人群：中年专业户外/探险人士、长途越野爱好者、自然领域专家

场景描述要求：
- 用1-3个中文短语描述，中间用"+"链接像真实用户生活中的与车交互
- 简单概括场景，如： "商场停车场找车 + 买咖啡","突发爆胎求助", "搜索火车航班信息+预定行程+查找酒店"。
- 场景要能自然引出后续的intent和slots，让对话有因果连贯性和生活气息
- 长度控制在5-20字，避免过于冗长
- 多样化：覆盖航班火车酒店联网搜索、导航、多媒体、车辆控制、舒适调节、紧急事件处理、闲聊、通勤、长途等场景
- 体现真实驾驶约束：用户在开车，不能低头操作，需要语音、手眼安全

输出规范：
- 只返回场景描述文本，不要任何多余内容

现在根据以下信息生成：

【领域】{domain}
【intent和slots】{intents_and_slots}"""


QUERY_SYSTEM_PROMPT = """

你是一个车载多轮对话数据集的构造专家,你可以参照剧本构造出前后逻辑紧密的多轮query,目的是测试车载模型结合对话上下文识别用户意图的能力

任务:根据传入的【场景描述】、【领域】、本轮【intent】和【slots】,和车载智能交互对话。

剧本：
- 开场:帮我打开空调/导航至公司，顺路去买一杯咖啡/查一下今天的最新新闻但不要娱乐信息，并帮我在公司附近找一家奶茶店
- 复杂意图理解:打开除了主驾侧以外的所有车窗/不要关闭抬头显示(HUD),而是保持开启状态
- 多轮对话:1.打开氛围灯 2.把亮度设为30 3.太暗了,调到70 4.再亮一点 5.稍微调暗一点
- 用户感知车控:我快要冻僵了/我的背好酸/起雾了，我看不清路况/车里闻起来像是腐烂食物味道
- 新闻:上海今天有什么重大新闻吗？/能给我讲一下谷歌大模型的最新动态吗？
- 天气:明天冷吗？/这周五北京会下雪吗？
- 联网搜导航:我想找些玩的地方，你有什么推荐的吗？/这附近有公园吗？/拉斯维加斯的地标是什么？
- 联网搜电影:过去五年奥斯卡最佳影片获奖名单有哪些？/给我介绍一下《绿皮书》/给我播放猫和老鼠
- 联网搜音乐：《Flowers》这首歌获得格莱美奖了吗？/播放周杰伦的《依然范特西》专辑/播放一些适合婚礼的歌单
- 火车和航班：帮我查一下明天飞成都的航班/目的地换成那个云南/出发日期改为12月31日/有这周六去深圳的火车吗？/改成从杭州出发

人物设定：
- 核心标签：48岁专业野外摄影师、野生动植物专家
- 常驻城市：成都
- 家庭状况：已婚（子女已成年或独立，家庭负担较轻）
- 职业：自由/专业野生动植物摄影师与专家（兼职科普或拍摄任务）
- 车辆类型：硬派越野SUV或改装越野车（强调通过性、空间、设备固定）
- 核心用车场景：用车时间自由（无固定上下班），以周末/工作日灵活出发的长途自驾 + 野外拍摄任务为主，经常进川西、川南山区或其他自然保护区，拍摄任务中长途占比高，可避开节假日高峰
- 用车核心属性：长途可靠性 + 越野能力 + 野外生存支持 + 专业拍摄辅助，依赖车载助手查询天气、路况、补给点、航班/火车备选、信号覆盖等，痛点包括野外信号弱、天气多变、设备保护、补给规划、安全预警
- 代表人群：中年专业户外/探险人士、长途越野爱好者、自然领域专家
- 情绪：有时疲惫/匆忙/困倦/烦躁/撒娇,身体感受（好困、饿死了、热死了、超烦、开心死了、眼睛酸）
- 用语习惯：语气词（嗯、额、呃、哎、哦对、对对对、那个、就是、你晓得），使用模糊指代（刚才那个、你知道的、那个平时喝的、就前面那个、红色那个）
- 说话特点：逻辑性强，前后两句话有关联。
- 喜好：把车载智能当做工具和聊天对象，喜欢就一个问题展开联想和衍生

人机交互习惯：
- 命令语气：打开空调、帮我切掉某某歌手的歌、给我导航途中的咖啡店、别放这个了
- 任务密度：一次只提出1条发问或命令或牢骚
- 交互偏好：强烈依赖上下文，默认车载记得之前的对话历史。使用指代词代替前文出现过的对象。
- 追问习惯：就上一个话题深入追问，如："1.现在是什么歌？2.歌手是谁？3.放点他最火的歌4.他和我平时听的歌手有合作吗？";"1.现在导航去公司2.距离多远，走高速还是什么路？3.我想先去买杯咖啡，好加途经点吗？4.油够不够，还是先去加油？5.那现在到公司要多久？"

能力测试点：
- 测试车载模型对于前文推理能力，信息融合能力，指代澄清能力，对话修正能力
- 示例场景：
  * 用户提到地点A后，下一轮说"导航到那里"（指代澄清能力）
  * 用户说"调到20度"，下一轮说"太冷了"（前文推理能力）
  * 用户问"离这儿多远"，需要结合前文的目的地（信息融合能力）
  * 用户改口："算了，还是去刚才说的那个地方"（对话修正能力）
- 至少45%的轮数必须依赖前文上下文才能正确理解用户意图和槽位提取

上下文依赖:
- 参照剧本生成上下关联的对话
- 能够结合上一条query预测设计本轮query，使得上下文相互关联，逻辑性强


输出规范：
只返回一个严格的 JSON 数组，每个元素格式为 {"turn": 轮次编号, "user_query": "生成的话"}
示例：[{"turn": 1, "user_query": "哎导航去最近的加油站"}, {"turn": 2, "user_query": "还有多远啊"}]
不要任何多余文字或 markdown 代码块。

重要约束:
1. 只返回JSON数组，不要任何markdown代码块或多余文字
2. 不能在query中提供目标答案
3. 对话中至少45%的轮数应该包含上下文依赖（指代、修正、融合等）
4. 适度的停顿、重复、自我纠正是可以的（更真实）

日常聊天记录：
{
["turn": 1,
"user_query": "嗯...现在几点了啊？"],
["turn": 2,
"user_query": "帮我导航到公司，顺便看看下午有没有日程待办"],
["turn": 3,
"user_query": "距离目的地还有多远"],
["turn": 4,
"user_query": "中途有没有我常喝的咖啡店"],
["turn": 5,
"user_query": "嗯...如果有合适的店就加个中途点"],
["turn": 6,
"user_query": "放点音乐呗，打开空调"],
["turn": 7,
"user_query": "平时都放的是什么歌！来首周杰伦"],
["turn": 8,
"user_query": "哎，这首给我听困了，换一首稻香吧"],
["turn": 9,
"user_query": "哎哎哎...现在是导航去哪呢，那个像大裤衩的建筑是什么？"],
["turn": 10,
"user_query": "下午4点我有其他事吗，没有就帮我约一个4点半的会议"],
["turn": 11,
"user_query": "哎哎...算了给我改到五点的会议"],
["turn": 12,
"user_query": "念一下我下午的日程"],
["turn": 13,
"user_query": "现在多少度啊...好热啊...你调一下"],
["turn": 14,
"user_query": "哎呀呀腰疼...帮我整一下"],
["turn": 15,
"user_query": "现在续航还剩多少，要加油不，快不行了提醒我"],
["turn": 16,
"user_query": "开不动了，前面什么情况啊，净耽误事，我几点能到目的地啊"],
["turn": 17,
"user_query": "给我换条最快的路"]
}
"""

def build_query_prompt(skeleton: dict) -> str:
    lines = [
        f"场景：{skeleton['scenario']}",
        f"领域：{skeleton['domain']}",
        "",
        "请为以下每一轮生成user_query（口语化中文），返回格式：",
        '[{"turn": 1, "user_query": "..."}, {"turn": 2, "user_query": "..."}, ...]',
        "",
        "各轮结构信息："
    ]
    for t in skeleton["turns"]:
        lines.append(
            f"  第{t['turn']}轮 | intent: {t['ground_truth_intent']} "
            f"| slots: {t['ground_slots']['attribute']} "
            f"| {t['reference_hint']}"
        )
    return "\n".join(lines)

# ── LLM 调用 ─────────────────────────────────────────────────────────────────

def call_llm_for_scenario(domains: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
                    {"role": "user", "content": f"领域组合：{domains}\n请生成一个对应的驾车场景："}
                ],
                temperature=0.7,
                max_tokens=80
            )
            content = response.choices[0].message.content  # 先不 strip
            if content is None:                             # ← 加这个判断
                print(f"  [场景] content 为 None，重试 {attempt}")
                time.sleep(2 ** attempt)
                continue
            scenario = content.strip()
            if 5 < len(scenario) < 50:
                return scenario
            print(f"  [场景] 长度异常，重试 {attempt}")
        except Exception as e:
            print(f"  [场景生成] 失败 (attempt {attempt}): {str(e)}")
            time.sleep(2 ** attempt)
    return f"日常驾车场景（{domains}）"

def call_llm_for_queries(skeleton: dict) -> list | None:
    prompt = build_query_prompt(skeleton)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.75,
                max_tokens=800
            )
            raw = response.choices[0].message.content
            if raw is None:
                print(f"  [Query] content 为 None，重试 {attempt}")
                continue
            raw = raw.strip()

            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == len(skeleton["turns"]):
                if all(isinstance(item, dict) and "turn" in item and "user_query" in item for item in parsed):
                    return parsed
                else:
                    print(f"  [Query] 格式错误：元素不是字典或缺少字段，重试{attempt}")
            else:
                print(f"  [Query] 轮数不匹配，期望{len(skeleton['turns'])}，得到{len(parsed)}，重试{attempt}")

        except json.JSONDecodeError as e:
            print(f"  [Query] JSON解析失败 (attempt {attempt}): {e}")
        except Exception as e:
            print(f"  [Query] API异常 (attempt {attempt}): {str(e)}")
            time.sleep(2 ** attempt + random.uniform(0, 1))
    return None

def fill_queries(skeleton: dict, llm_turns: list) -> dict:
    query_map = {item["turn"]: item["user_query"] for item in llm_turns}
    for t in skeleton["turns"]:
        t["user_query"] = query_map.get(t["turn"], "[生成失败]")
        if "reference_hint" in t:
            del t["reference_hint"]
    return skeleton

# ── 主流程 ────────────────────────────────────────────────────────────────────
def save_checkpoint(dataset: dict, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"  [checkpoint] 已保存 {len(dataset['conversations'])} 条至 {output_file}")

def main():
    if not API_KEY:
        print("[错误] 未配置 LLM_API_KEY，请复制 .env.example 为 .env 并填写")
        return

    dataset = {"conversations": []}
    failed = []
    conv_id = 1
    valid_count = 0  # 新增：有效数据计数器

    for tier in TIER_CONFIG:
        print(f"\n── 生成 {tier['count']} 条，轮数范围 {tier['turn_range']} ──")
        for _ in range(tier["count"]):
            skeleton = build_conversation_skeleton(conv_id, tier["turn_range"])
            scenario = call_llm_for_scenario(skeleton["domain"])
            skeleton["scenario"] = scenario

            print(f"[进度: {valid_count + 1}/{sum(t['count'] for t in TIER_CONFIG)}] ID: {skeleton['conv_id']} | "
                  f"{len(skeleton['turns'])}轮 | {scenario[:40]}...")

            llm_result = call_llm_for_queries(skeleton)

            if llm_result:
                filled = fill_queries(skeleton, llm_result)
                dataset["conversations"].append(filled)
                valid_count += 1  # 只有成功才计数

                # 每50条有效数据保存一次
                if valid_count % 50 == 0:
                    save_checkpoint(dataset, OUTPUT_FILE)
            else:
                print(f"  [失败] {skeleton['conv_id']} query生成失败")
                for t in skeleton["turns"]:
                    t["user_query"] = "[LLM生成失败]"
                    t.pop("reference_hint", None)
                dataset["conversations"].append(skeleton)
                failed.append(skeleton["conv_id"])

            conv_id += 1
            time.sleep(0.8 + random.uniform(0, 0.7))

    # 最终保存（处理不足50条的尾部）
    save_checkpoint(dataset, OUTPUT_FILE)

    print(f"\n完成！生成 {len(dataset['conversations'])} 条（有效：{valid_count} 条）")
    print(f"保存至：{os.path.abspath(OUTPUT_FILE)}")
    if failed:
        print(f"失败条目（{len(failed)}条）：{failed}")
if __name__ == "__main__":
    main()