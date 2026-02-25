import json
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI


# ==================== 数据加载（仅作备选，不参与默认逻辑） ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPE_DB_PATH = os.path.join(BASE_DIR, "recipe_database.json")

try:
    with open(RECIPE_DB_PATH, "r", encoding="utf-8") as f:
        RECIPE_DB: List[Dict[str, Any]] = json.load(f)
except FileNotFoundError:
    RECIPE_DB = []


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

    ingredients_text = ", ".join(_normalize_ingredients_string(user_ingredients))

    system_prompt = (
        "你是一名创意十足的中文厨师助手，需要输出一个 JSON 对象，"
        "仅包含一个字段 recipes，类型为数组。\n"
        "每个元素必须包含：id, name, match_score 三个字段：\n"
        "1. id：字符串，作为唯一标识，推荐直接使用菜名。\n"
        "2. name：字符串，用于在前端展示的菜名。\n"
        "3. match_score：0-1 的浮点数，表示与用户食材的相关度或推荐置信度。\n"
        "请只返回 JSON，不要包含其它任何说明文字。"
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


def generate_structured_recipe(recipe_name: str) -> Dict[str, Any]:
    """
    使用 OpenAI API 生成结构化菜谱。

    返回格式：
    {
      "name": "",
      "ingredients": [],
      "seasonings": [],
      "steps": [],
      "cooking_time": "",
      "tips": ""
    }

    要求：数量精确，带单位，例如：盐：3g，油：10ml 等。
    """
    client = _create_openai_client()
    model_name = os.environ.get("OPENAI_MODEL", "deepseek-chat")

    system_prompt = (
        "你是一名专业的中文厨师助手，需要输出一个 JSON 对象，"
        "字段严格为：name, ingredients, seasonings, steps, cooking_time, tips。\n\n"
        "要求：\n"
        "1. name 为菜名。\n"
        '2. ingredients 为数组，每个元素包含 name, amount, unit，例如：{"name": "鸡蛋", "amount": "2", "unit": "个"}。\n'
        "3. seasonings 为数组，结构与 ingredients 相同，必须给出精确数量和单位，如：3g, 10ml 等。\n"
        "4. steps 为数组，分步描述做法，每一步清晰简洁。\n"
        '5. cooking_time 为字符串，单位为分钟，例如 "20" 表示 20 分钟。\n'
        "6. tips 为字符串，给出烹饪小提示。\n"
        "7. 所有字段必须存在，且符合 JSON 格式，不能包含多余字段。\n"
    )

    user_prompt = f"请为菜名《{recipe_name}》生成一个完整的结构化菜谱。"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},  # 要求返回严格 JSON
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    # 简单校验字段存在性
    expected_keys = {
        "name",
        "ingredients",
        "seasonings",
        "steps",
        "cooking_time",
        "tips",
    }
    if not expected_keys.issubset(data.keys()):
        raise ValueError("LLM 返回的结构中缺少必要字段。")

    return data


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
    recipe_id: str


class RecipeDetail(BaseModel):
    id: str
    name: str
    ingredients: List[Dict[str, Any]]
    seasonings: List[Dict[str, Any]]
    steps: List[str]
    cooking_time: str
    difficulty: str


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
    """
    results = recommend_recipes_by_ingredients(body.ingredients)
    return RecommendResponse(results=results)


@app.post("/select_recipe", response_model=RecipeDetail)
def api_select_recipe(body: SelectRecipeRequest):
    """
    根据 recipe_id（此处直接视为菜名）调用 LLM 生成完整结构化菜谱信息。
    若 LLM 调用失败，则尝试从本地 JSON 数据库中查找同名菜谱作为备选。
    """
    recipe_name = body.recipe_id

    try:
        data = generate_structured_recipe(recipe_name)
    except Exception as e:
        fallback = _find_recipe_by_id(recipe_name)
        if not fallback:
            raise HTTPException(status_code=500, detail=f"生成菜谱失败: {e}") from e
        return RecipeDetail(
            id=fallback["id"],
            name=fallback["name"],
            ingredients=fallback.get("ingredients", []),
            seasonings=fallback.get("seasonings", []),
            steps=fallback.get("steps", []),
            cooking_time=fallback.get("cooking_time", ""),
            difficulty=fallback.get("difficulty", "easy"),
        )

    return RecipeDetail(
        id=data.get("name", recipe_name),
        name=data.get("name", recipe_name),
        ingredients=data.get("ingredients", []),
        seasonings=data.get("seasonings", []),
        steps=data.get("steps", []),
        cooking_time=data.get("cooking_time", ""),
        difficulty="easy",
    )


# 静态资源挂载：用于访问 index.html 以外的静态文件（如将来扩展 JS/CSS）
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR),
    name="static_root",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
