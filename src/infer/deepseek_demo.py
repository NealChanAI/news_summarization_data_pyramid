# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2025-03-31 10:22
#    @Description   : DeepSeek demo
#
# ===============================================================


from openai import OpenAI

API_KEY = ''

client = OpenAI(api_key=API_KEY, base_url='https://api.deepseek.com')

response = client.chat.completions.create(
    model='deepseek-chat',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': 'Hello'},
    ],
    stream=False
)

print(response.choices[0].message.content)
