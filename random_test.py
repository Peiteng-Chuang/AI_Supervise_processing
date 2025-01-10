import random

# 測試多次以檢驗權重
results = [random.choices([True, False], weights=[90, 10], k=1)[0] for _ in range(10000)]

# 計算出現的比例
true_count = results.count(True)
false_count = results.count(False)

print(f"True 出現次數: {true_count}")
print(f"False 出現次數: {false_count}")
print(f"True 出現比例: {true_count / len(results):.4f}")
print(f"False 出現比例: {false_count / len(results):.4f}")