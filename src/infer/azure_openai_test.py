# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024/12/21 19:14
#    @Description   : AZURE OPENAI API demo
#
# ===============================================================


import os
import base64
from openai import AzureOpenAI


endpoint = os.getenv("AZURE_ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")


# 使用基于密钥的身份验证来初始化 Azure OpenAI 客户端
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

# 准备聊天提示
chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "你是一个帮助用户查找信息的 AI 助手。"
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "soul是什么app"
            },
            # {
            #     "type": "text",
            #     "text": "\n"
            # }
        ]
    }
]

chat_prompt = [
    {'role': 'system', 'content': '你是一个帮助用户查找信息的 AI 助手'},
    {'role': 'user', 'content': 'soul是什么app'}

]

# 如果已启用语音，则包括语音结果
messages = chat_prompt

# 生成完成
completion = client.chat.completions.create(
    model=deployment,
    messages=messages,
    max_tokens=800,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    stream=False
)

print(completion.to_json())
print('-' * 20)
print(completion.choices[0].message.content)

