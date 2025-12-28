import requests

# 目标 URL
service = "http:%2F%2Flanunion.cqu.edu.cn%2Frepair%2Flogin.php%3Fredirect_uri%3D%252Frepair%252Fadmin%252Findex.php%252Fuser%252Floginpost"
ticket = "ST-7781393-kRXs-tNg1hrzeL5RyjN8P-6Zx10rg-sso-9499fb667-v97rg"
url = f"https://sso.cqu.edu.cn/p3/serviceValidate?service={service}&ticket={ticket}"

try:
    # 发送 GET 请求
    response = requests.get(url)

    # 检查响应状态码是否为 200 (OK)
    if response.status_code == 200:
        print("✅ 请求成功！")

        # 获取响应内容
        # 1. 以文本形式获取
        print("\n--- 响应文本 (response.text) ---")
        print(response.text)


    else:
        print(f"❌ 请求失败，状态码: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"网络错误或其他异常: {e}")