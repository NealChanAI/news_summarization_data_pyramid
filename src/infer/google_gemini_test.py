# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-24 16:05
#    @Description   : Google Gemini 2.0 Flash API调用
#
# ===============================================================


from google import genai
import os


gemini_api_key = os.getenv("GEMINI_API_KEY")


client = genai.Client(
    vertexai=False,
    api_key=gemini_api_key
)
response = client.models.generate_content(
    # model='gemini-2.0-flash-exp',
    model='gemini-2.0-flash-thinking-exp-1219',
    contents='soul的冷启动策略是怎么样的'
)
print(response.text)
