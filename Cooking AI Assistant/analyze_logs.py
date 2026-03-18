import json
import os


def run_data_cleaning_experiment():
    # 动态获取今日日志文件名
    from datetime import datetime

    log_file = f"logs/log_{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    if not os.path.exists(log_file):
        print(f"找不到日志文件: {log_file}")
        return

    bad_cases = []
    total_requests = 0
    total_recommendations = 0
    latencies = []  # <--- 新增：用于存储所有请求的耗时

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 统计推荐阶段
            if log["stage"] == "recommend":
                total_recommendations += 1

            # 统计详情阶段
            if log["stage"] == "select_recipe":
                total_requests += 1
                output = log.get("output") or {}  # 防止 output 为 None

                # --- 核心改进：累加耗时数据 ---
                # 只有当 output 里确实记录了 latency 时才统计
                if "latency" in output:
                    latencies.append(output["latency"])

                # PM 逻辑：检查关键字段是否存在或是否为空
                ingredients = output.get("ingredients", [])
                steps = output.get("steps", [])

                if not steps or not ingredients:
                    bad_cases.append({"input": log["input"], "reason": "内容缺失"})
                elif output.get("tips") == "已加载本地缓存方案":
                    bad_cases.append(
                        {"input": log["input"], "reason": "LLM 报错，使用了本地降级"}
                    )

    # --- 打印报告 ---
    print(f"\n--- 📊 每日产品运营实验报告 ({datetime.now().strftime('%Y-%m-%d')}) ---")
    print(f"1. 搜索推荐次数: {total_recommendations}")
    print(f"2. 菜谱生成次数: {total_requests}")
    print(f"3. 异常案例 (Bad Cases): {len(bad_cases)}")

    if total_requests > 0:
        error_rate = (len(bad_cases) / total_requests) * 100
        print(f"4. 整体服务异常率: {error_rate:.2f}%")

        # --- 打印平均耗时 ---
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"5. 平均生成耗时: {avg_latency:.2f}s")  # 使用 .2f 保留两位小数
        else:
            print(f"5. 平均生成耗时: 暂无数据 (请检查 main.py 是否已添加 latency 记录)")

        if bad_cases:
            print("\n--- 异常原因明细 ---")
            for case in bad_cases:
                print(f" - 食材: {case['input']} | 原因: {case['reason']}")
    else:
        print("暂无生成的详情数据可分析")


if __name__ == "__main__":
    run_data_cleaning_experiment()
