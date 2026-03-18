import json
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

import time
from datetime import datetime
from dotenv import load_dotenv  # 新增导入

# 在所有逻辑开始前加载环境变量
load_dotenv()

# ==================== 数据加载（仅作备选，不参与默认逻辑） ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 绝对路径
RECIPE_DB_PATH = os.path.join(BASE_DIR, "recipe_database.json")  # 相对路径

try:
    with open(RECIPE_DB_PATH, "r", encoding="utf-8") as f:
        RECIPE_DB: List[Dict[str, Any]] = json.load(f)
except FileNotFoundError:
    RECIPE_DB = []

# 创建日志目录
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def save_interaction_log(
    stage: str, input_data: Any, output_data: Any, error: Optional[str] = None
):
    """
    记录每一次交互，用于后续数据分析（Bad Case 分析）
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,  # 'recommend' 或 'select_recipe'
        "input": input_data,
        "output": output_data,
        "error": error,
        "success": error is None,
    }
    # 按天存储日志文件
    log_file = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(log_entry, ensure_ascii=False) + "\n"
        )  # ensure_ascii=False 确保中文不乱码


# ==================== 工具函数 ====================


def _normalize_ingredients_string(user_ingredients: str) -> List[str]:
    """
    将用户输入的食材字符串进行简单切分与清洗。
    例如："鸡蛋, 西红柿; 土豆" -> ["鸡蛋", "西红柿", "土豆"]
    """
    if not user_ingredients:
        return []
    text = user_ingredients
    for sep in ["，", ",", "；", ";", "、"]:
        text = text.replace(sep, " ")
    tokens = [t.strip() for t in text.split(" ") if t.strip()]
    return tokens


def _create_openai_client() -> OpenAI:
    """
    创建 OpenAI 兼容客户端，严格使用环境变量：
    - OPENAI_API_KEY
    - OPENAI_BASE_URL（指向 DeepSeek 或兼容服务）
    - OPENAI_MODEL（如 deepseek-chat）
    """
    # import httpx  # 导入底层的网络库

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法调用 LLM。")

    base_url = os.environ.get("OPENAI_BASE_URL")
    # client_kwargs: Dict[str, Any] = {"api_key": api_key}
    # if base_url:
    #     client_kwargs["base_url"] = base_url

    return OpenAI(api_key=api_key, base_url=base_url)


def recommend_recipes_by_ingredients(user_ingredients: str) -> List[Dict[str, Any]]:
    """
    使用 DeepSeek（或兼容 LLM）根据用户输入的食材，直接生成创意菜谱推荐列表。

    返回格式：
    [
      {"id": "菜名或唯一ID", "name": "展示用菜名", "match_score": 0.9},
      ...
    ]
    这里的 match_score 代表与食材的相关度或推荐置信度，由 LLM 决定。
    """
    client = _create_openai_client()
    model_name = os.environ.get("OPENAI_MODEL", "deepseek-chat")

    ingredients_text = ", ".join(
        _normalize_ingredients_string(user_ingredients)
    )  # 将用户输入的食材字符串进行简单切分与清洗，并拼接成字符串

    system_prompt = (
        "你是一名具备临床背景的专业精准营养师。规则非常严格：\n"
        "1. **必须以用户提供的食材作为绝对主料**，不允许脱离主料。\n"
        "2. **最多只能额外增加 1～2 种最常见、最基础的辅助食材**（如葱、姜、蒜、盐、油）。\n"
        "3. 严禁凭空增加大量主材。\n"
        "4. 输出一个 JSON 对象，仅包含字段 recipes 数组。\n"
        "每个元素必须包含：id, name, match_score, health_tag, calorie_estimate 字段：\n"
        "1. id：菜名。\n"
        "2. name：展示用菜名。\n"
        "3. match_score：与食材相关度 (0-1)。\n"
        "4. health_tag：健康标签（如：低GI、高蛋白、低脂、控糖）。\n"
        "5. calorie_estimate：该分量下估算的千卡数值（如 '350kcal'）。\n"
        "请只返回 JSON，保持严谨和科学。"
    )

    user_prompt = (
        f"用户可用的主要食材为：{ingredients_text}。\n"
        "请基于这些食材，推荐 3 个有创意的中式菜名，"
        "可以是家常菜或融合菜，但要尽量与食材相关。"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    recipes = data.get("recipes", [])

    result: List[Dict[str, Any]] = []
    for item in recipes:
        rid = str(item.get("id") or item.get("name") or "").strip()
        name = str(item.get("name") or rid).strip()
        if not rid or not name:
            continue
        try:
            score = float(item.get("match_score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        result.append(
            {
                "id": rid,
                "name": name,
                "match_score": round(score, 3),
            }
        )

    return result[:3]


# V1.0: 无COT
# def generate_structured_recipe(recipe_name: str) -> Dict[str, Any]:
#     """
#     使用 OpenAI API 生成结构化菜谱。

#     返回格式：
#     {
#       "name": "",
#       "ingredients": [],
#       "seasonings": [],
#       "steps": [],
#       "cooking_time": "",
#       "tips": ""
#     }

#     要求：数量精确，带单位，例如：盐：3g，油：10ml 等。
#     """
#     """
#     升级版：结合本地 RAG 知识生成的营养干预菜谱
#     """
#     client = _create_openai_client()
#     model_name = os.environ.get("OPENAI_MODEL", "deepseek-chat")

#     # 【RAG 检索逻辑】：尝试从本地 JSON 找专业定义
#     local_data = _find_recipe_by_id(recipe_name)
#     context_str = ""
#     if local_data and "clinical_definition" in local_data:
#         context_str = f"\n参考本地权威定义：{local_data['clinical_definition']}"

#     system_prompt = (
#         "你是一名专业临床营养师。请为指定菜谱生成结构化数据，JSON 字段严格为：\n"
#         "name, ingredients, seasonings, steps, cooking_time, nutrition_facts, tips。\n\n"
#         "要求：\n"
#         "1. ingredients 和 seasonings 必须包含精确数量（如 3g, 10ml）且ingredients 和 seasonings 数组中的每个元素必须是对象，包含 {'name': '...', 'amount': '...', 'unit': '...'} 字段，严禁直接返回字符串。。\n"
#         "2. steps 需体现少盐少油的健康烹饪手法，其中调味品 seasonings 需体现少盐少油的干预原则。\n"
#         "3. nutrition_facts 为对象，包含：calories, protein, fat, carbohydrates（必须估算数值）且nutrition_facts 必须包含 calories, protein, fat, carbohydrates 字段。。\n"
#         "4. tips 必须包含一条针对特定人群（如糖尿病或健身者）的专业食用建议。\n"
#     )
#     # 示例：将本地参考数据塞给 Prompt
#     user_prompt = (
#         f"菜名：《{recipe_name}》。{context_str}\n请生成精准的结构化膳食建议。"
#     )

#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         temperature=0.3,
#         response_format={"type": "json_object"},  # 要求返回严格 JSON
#     )

#     content = response.choices[0].message.content
#     data = json.loads(content)

#     return data


# V2.0：COT
def generate_structured_recipe(
    recipe_name: str, user_type: str = "general"
) -> Dict[str, Any]:
    """
    升级版：引入思维链（CoT）逻辑，确保营养计算与干预原则的严谨性。
    """
    client = _create_openai_client()
    model_name = os.environ.get("OPENAI_MODEL", "deepseek-chat")

    # 【RAG 检索逻辑】
    local_data = _find_recipe_by_id(recipe_name)
    context_str = ""
    if local_data and "clinical_definition" in local_data:
        context_str = f"\n参考本地权威定义：{local_data['clinical_definition']}"

    # 1. 定义身份上下文（Identity Context）
    identity_map = {
        "diabetes": "用户是糖尿病患者，严格控制升糖指数（GI），严禁加糖，控制精制淀粉，建议增加膳食纤维。",
        "fitness": "用户是健身增肌人群，追求高蛋白质，中等碳水，严格控制劣质脂肪，注重微量元素摄入。",
        "hypertension": "用户是高血压患者，严格执行‘低钠盐’准则，严禁高盐调料，建议增加富含钾的食材描述。",
        "general": "用户是健康成年人，追求营养均衡，口味适中，遵循中国居民膳食指南。",
    }
    identity_context = identity_map.get(user_type, identity_map["general"])

    # --- 核心改进：在 Prompt 中嵌入 CoT 逻辑 ---
    system_prompt = (
        f"你是一名资深临床营养师。当前干预目标：{identity_context}\n\n"
        "在生成数据前，请务必遵循以下思维链路（Chain of Thought）：\n"
        "1. 【分析】：分析菜品原始特征。\n"
        "2. 【干预】：基于上述“干预目标”对调味进行针对性替换，食材必须严格使用数据库里的，不能新增数据库里没有的食材。\n"
        "3. 【计算】：精确估算营养数据。\n"
        "4. 【校验】：确保方案完全符合干预目标的要求。\n\n"
        "请返回 JSON 对象，字段包括：thoughts, name, ingredients, seasonings, steps, cooking_time, nutrition_facts, tips。\n"
        "其中 ingredients, seasonings 必须为对象数组 {'name': '...', 'amount': '...', 'unit': '...'}。"
        "\n严格约束：必须以用户提供的食材为主，只能补充油盐葱姜蒜等基础辅料。"
    )

    user_prompt = f"菜名：《{recipe_name}》。{context_str}\n请基于你的思维链路，生成精准的结构化膳食建议。"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # 保持低随机性，确保计算稳定
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    # 打印思维过程，方便我们在后台观察 AI 是怎么想的
    if "thoughts" in data:
        print(f"DEBUG - AI 思维链分析: {data['thoughts']}")

    return data


def retrieve_recipes_by_ingredients(user_ingredients: str) -> List[Dict[str, Any]]:
    """
    从本地 recipe_database.json 根据食材进行简单检索
    """
    ingredients = _normalize_ingredients_string(user_ingredients)

    scored_recipes = []

    for recipe in RECIPE_DB:

        score = 0
        recipe_text = json.dumps(recipe, ensure_ascii=False)

        for ing in ingredients:
            if ing in recipe_text:
                score += 1

        if score > 0:
            scored_recipes.append((score, recipe))

    scored_recipes.sort(reverse=True, key=lambda x: x[0])

    results = []
    for score, recipe in scored_recipes[:3]:

        results.append(
            {
                "id": recipe["id"],
                "name": recipe["name"],
                "match_score": round(score / len(ingredients), 2),
            }
        )

    return results


# ==================== Pydantic 模型 ====================


class RecommendRequest(BaseModel):
    ingredients: str


class RecommendItem(BaseModel):
    id: str
    name: str
    match_score: float


class RecommendResponse(BaseModel):
    results: List[RecommendItem]


class SelectRecipeRequest(BaseModel):
    recipe_id: str  # 菜谱ID
    user_type: Optional[str] = "general"  # 新增字段，默认为普通模式


class RecipeDetail(BaseModel):
    id: str
    name: str
    ingredients: List[Dict[str, Any]]
    seasonings: List[Dict[str, Any]]
    steps: List[str]
    cooking_time: str
    difficulty: str


# ==================== Pydantic 模型（已更新以匹配营养干预需求） ====================


class NutritionFacts(BaseModel):
    calories: str
    protein: str
    fat: str
    carbohydrates: str


class RecipeDetail(BaseModel):
    id: str
    name: str
    ingredients: List[Dict[str, Any]]
    seasonings: List[Dict[str, Any]]
    steps: List[str]
    cooking_time: str
    difficulty: str
    # 新增营养与专家字段
    nutrition_facts: Optional[NutritionFacts] = None
    tips: Optional[str] = None
    clinical_definition: Optional[str] = None


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="Cooking AI Assistant",
    description="基于 FastAPI 的 Cooking AI 推荐示例。",
    version="0.1.0",
)

# CORS：方便前端页面直接调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许全部，生产环境建议收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _find_recipe_by_id(recipe_id: str) -> Optional[Dict[str, Any]]:
    for recipe in RECIPE_DB:
        if recipe.get("id") == recipe_id:
            return recipe
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root():
    """
    根路径直接返回 index.html，方便前端访问。
    """
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.post("/recommend", response_model=RecommendResponse)
def api_recommend(body: RecommendRequest):
    """
    输入食材字符串，返回 Top3 推荐菜谱。
    增加了完善的错误处理与日志记录。
    """
    try:
        # 先从数据库检索
        results = retrieve_recipes_by_ingredients(body.ingredients)

        # 如果数据库没有结果，再调用LLM
        if not results:
            results = recommend_recipes_by_ingredients(body.ingredients)
        # 成功时记录：输入是什么，AI 推荐了哪 3 个菜
        save_interaction_log("recommend", body.ingredients, results, None)
        return RecommendResponse(results=results)
    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG: Recommend 发生错误 -> {error_msg}")

        # 失败记录：输入食材 + 错误信息
        save_interaction_log("recommend", body.ingredients, None, error_msg)

        # 抛出 HTTP 异常让前端知道
        raise HTTPException(status_code=500, detail=f"推荐失败: {error_msg}")


@app.post("/select_recipe", response_model=RecipeDetail)
def api_select_recipe(body: SelectRecipeRequest):
    start_time = time.time()  # 记录开始时间
    recipe_name = body.recipe_id  # 菜谱ID
    local_ref = _find_recipe_by_id(recipe_name)

    try:
        # 将前端传来的 user_type 传给生成函数
        data = generate_structured_recipe(body.recipe_id, body.user_type)

        # --- 新增：食材格式清洗逻辑 ---
        def clean_list_to_dicts(items):
            if not isinstance(items, list):
                return []
            cleaned = []
            for item in items:
                if isinstance(item, dict):
                    cleaned.append(item)
                elif isinstance(item, str):
                    # 如果是 "西兰花 200g"，简单切分为字典
                    parts = item.split(" ")
                    cleaned.append(
                        {
                            "name": parts[0],
                            "amount": parts[1] if len(parts) > 1 else "",
                            "unit": "",
                        }
                    )
            return cleaned

        # 清洗 ingredients 和 seasonings
        safe_ingredients = clean_list_to_dicts(data.get("ingredients", []))
        safe_seasonings = clean_list_to_dicts(data.get("seasonings", []))

        # --- 之前的营养数据转换逻辑 ---
        raw_nf = data.get("nutrition_facts", {})
        # 如果不是字典，强制转为空
        if not isinstance(raw_nf, dict):
            raw_nf = {}
        safe_nutrition_facts = NutritionFacts(
            calories=str(raw_nf.get("calories", "N/A")),
            protein=str(raw_nf.get("protein", "N/A")),
            fat=str(raw_nf.get("fat", "N/A")),
            carbohydrates=str(raw_nf.get("carbohydrates", "N/A")),
        )

        response_data = RecipeDetail(
            id=str(data.get("name", recipe_name)),
            name=str(data.get("name", recipe_name)),
            ingredients=safe_ingredients,
            seasonings=safe_seasonings,
            steps=data.get("steps", []),
            cooking_time=str(data.get("cooking_time", "20")),
            difficulty="medium",
            nutrition_facts=safe_nutrition_facts,
            tips=data.get("tips") or data.get("professional_tips", "暂无建议"),
            clinical_definition=(
                local_ref.get("clinical_definition")
                if local_ref
                else "个性化营养膳食方案"
            ),
        )
        # # 2. 【核心改进】成功时记录：输入菜名，输出完整的菜谱详情
        # # 使用 .model_dump() 将 Pydantic 对象转为字典记录
        # save_interaction_log(
        #     "select_recipe", body.recipe_id, response_data.model_dump(), None
        # )

        # 在最后计算耗时
        duration = round(time.time() - start_time, 2)

        # 存入日志时带上 duration
        log_data = response_data.model_dump()
        log_data["latency"] = duration  # 记录到输出中
        save_interaction_log("select_recipe", body.recipe_id, log_data, None)

        return response_data

    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG: 发生错误 -> {error_msg}")

        # 降级逻辑保持不变
        # fallback_ingredients = []
        if local_ref:
            # 如果本地有数据，直接用本地的
            fallback_data = RecipeDetail(
                id=local_ref["id"],
                name=local_ref["name"],
                ingredients=local_ref.get("ingredients", []),
                seasonings=local_ref.get("seasonings", []),
                steps=local_ref.get("steps", []),
                cooking_time=local_ref.get("cooking_time", "20"),
                difficulty=local_ref.get("difficulty", "easy"),
                nutrition_facts=NutritionFacts(
                    calories="N/A", protein="N/A", fat="N/A", carbohydrates="N/A"
                ),
                tips="已加载本地缓存方案",
            )
            # 记录降级情况，标记错误原因
            save_interaction_log(
                "select_recipe",
                body.recipe_id,
                fallback_data.model_dump(),
                f"Fallback used: {error_msg}",
            )
            return fallback_data

        # 彻底失败时记录
        save_interaction_log("select_recipe", body.recipe_id, None, error_msg)
        raise HTTPException(status_code=500, detail=f"生成失败: {error_msg}")


# 静态资源挂载：用于访问 index.html 以外的静态文件（如将来扩展 JS/CSS）
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR),
    name="static_root",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
