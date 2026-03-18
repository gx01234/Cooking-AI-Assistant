import json
import os


def clean_for_finetuning(input_log_file, output_train_file):
    cleaned_data = []
    total_lines = 0  # <--- 修复点 1：初始化总行数计数器

    if not os.path.exists(input_log_file):
        print(f"源日志文件不存在: {input_log_file}")
        return

    with open(input_log_file, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1  # <--- 修复点 2：每读一行，计数加 1
            try:
                line = line.strip()
                if not line:
                    continue
                log = json.loads(line)

                # --- STAGE 1: 格式与状态校验 (Format Check) ---
                # 必须成功且没有错误信息
                if not log.get("success") or log.get("error"):
                    continue

                # 我们只清洗“详情生成”阶段的数据作为训练集
                if log.get("stage") != "select_recipe":
                    continue

                output = log.get("output", {})

                # --- STAGE 2: 逻辑与质量校验 (Logic Check) ---
                ingredients = output.get("ingredients", [])
                steps = output.get("steps", [])

                # 剔除内容太单薄的数据（步骤少于3步，或食材少于2样）
                if len(steps) < 3 or len(ingredients) < 2:
                    continue

                # --- STAGE 3: 性能指标筛选 (Metric Filtering) ---
                # 只选取生成速度快的数据（假设 latency 在 output 中）
                latency = output.get("latency", 0)
                if latency > 15:  # 超过15秒的可能逻辑较乱，不作为优质教材
                    continue

                # --- STAGE 4: 数据对齐 (Alignment) ---
                # 转化为标准微调格式
                training_sample = {
                    "instruction": "你是一名精准营养师，请根据菜名生成结构化的膳食方案。",
                    "input": log.get("input", ""),
                    "output": json.dumps(output, ensure_ascii=False),
                }

                cleaned_data.append(training_sample)

            except Exception as e:
                # 记录一下解析失败的情况
                continue

    # 保存精选后的训练集
    with open(output_train_file, "w", encoding="utf-8") as f:
        for entry in cleaned_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✨ 数据清洗报告 ✨")
    print(f"---------------------------------")
    print(f"原始日志记录: {total_lines} 条")
    print(f"精选训练样本: {len(cleaned_data)} 条")
    print(
        f"数据留存率: {(len(cleaned_data)/total_lines*100):.2f}%"
        if total_lines > 0
        else "0%"
    )
    print(f"输出文件: {output_train_file}")


if __name__ == "__main__":
    # 确保文件名和你本地 logs 文件夹里的一致
    clean_for_finetuning("logs/log_2026-03-10.jsonl", "train_data_v1.jsonl")
