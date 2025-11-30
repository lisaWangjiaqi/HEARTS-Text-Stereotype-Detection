# import requests

# API_ENDPOINT = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
# TEAM_ID = "Jiaqi"
# API_TOKEN = "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc"

# response = requests.post(
#     API_ENDPOINT,
#     headers={
#         "Content-Type": "application/json",
#         "X-Team-ID": TEAM_ID,
#         "X-API-Token": API_TOKEN
#     },
#     json={
#         "team_id": TEAM_ID,
#         "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
#         "messages": [
#         {"role": "user", "content": "Hello, how are you?"}
#         ],
#         "max_tokens": 1024
#     }
# )

# data = response.json()
# print(response)
# print(data)
# # print(data["content"][0]["text"])


import requests
import json

# 请求地址
url = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"

# 请求头
headers = {
    "Content-Type": "application/json",
    "X-Team-ID": "Jiaqi",
    # 实际使用时建议改为从环境变量读取
    "X-API-Token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
}

# 请求体（Python 字典，会在下面用 json.dumps 序列化）
payload = {
    "team_id": "Jiaqi",
    "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "api_token":"GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 1024,
}

def main():
    """
    调用远程模型接口并打印结果

    输入：
        无（直接使用上面定义好的 url / headers / payload）

    输出：
        在控制台打印 HTTP 状态码和返回内容（JSON 或文本）
    """
    # 发送 POST 请求
    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),  # 将 Python dict 序列化为 JSON 字符串
        timeout=60,
    )

    print("Status code:", response.status_code)

    # 尝试按 JSON 解析返回
    try:
        data = response.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except ValueError:
        # 如果不是 JSON，就直接打印文本
        print("Response text:")
        print(response.text)


if __name__ == "__main__":
    main()
