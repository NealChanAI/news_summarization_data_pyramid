# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-24 16:05
#    @Description   : Google Gemini 2.0 Flash API调用
#
# ===============================================================


from google import genai
from google.genai import types
import os


gemini_api_key = os.getenv("GEMINI_API_KEY")
text = '男子头部中弹数年后才发现。　　□新华社今日上午电。　　德国一名男子因持续头疼前往医院就诊，本以为是患了脑瘤，但医生却惊讶地在他头皮与头骨间发现一枚弹头。。　　这名男子现年35岁，德国西部城市波鸿的警方24日说，医生从这名男子后脑头皮下取出一枚点22口径的弹头。。　　警方发言人说，这名男子得知头皮下有弹头才回想起，2004年或2005年的新年前夜庆祝活动中，他在酩酊大醉之时头部曾受击打，但伤口不久便愈合。最近以来，他频繁头痛，到医院检查后医生告诉他，他的头皮下有颗弹头，所幸弹头没有射穿头盖骨。。　　警方发言人说：“可能是有人朝天射击，子弹下落时打入他的头部。 ”。。'
user_prompt = '''
            # 角色：你是一名华语新闻摘要撰写专家
            # 场景：根据提供的新闻内容，撰写一份高质量的中文摘要
            # 需要分析的【新闻内容】：${CONTENT}

            # 任务：
            1. 理解与分析提供给你的【新闻内容】；
            2. 根据【内容】撰写一篇高质量的【新闻摘要】；

            # 【字段定义】
            请严格按照如下格式仅输出JSON，不要输出Python代码或其他信息：
            {
              "text": 【新闻内容】,
              "summary": 【新闻摘要】
            }

            # 注意事项：
            1. 必须严格按照【字段定义】中的格式输出；
            2. 必须严格避免出现政治敏感、色情、赌博相关的词汇；
            3. 用精炼的语言撰写摘要，确保语言流畅、逻辑清晰；
            4. 确保摘要贴合新闻重点，无冗余或遗漏。
        '''


client = genai.Client(
    vertexai=False,
    api_key=gemini_api_key
)
user_prompt = user_prompt.replace('${CONTENT}', text)
print(user_prompt)

response = client.models.generate_content(
            model='gemini-2.0-flash-thinking-exp-1219',
            # model='gemini-2.0-flash-exp',
            config=types.GenerateContentConfig(
                system_instruction='你是一个资深的新闻摘要撰写专家，擅长为华语新闻撰写高质量的摘要(Abstractive Summary)',
                top_p=0,
                temperature=0,
                max_output_tokens=1024,
                # response_mime_type='application/json',
                # response_schema={
                #     'required': ["text", "summary"],
                #     'properties': {
                #         "text": {"type": "string"},
                #         "summary": {"type": "integer"}
                #     },
                #     'type': 'OBJECT',
                # }
            ),
            contents=user_prompt
        )
print(response.text)


# import google.generativeai as genai
#
# import typing_extensions as typing
#
# class Recipe(typing.TypedDict):
#     recipe_name: str
#     ingredients: list[str]
#
# model = genai.GenerativeModel("gemini-1.5-pro-latest")
# result = model.generate_content(
#     "List a few popular cookie recipes.",
#     generation_config=genai.GenerationConfig(
#         response_mime_type="application/json", response_schema=list[Recipe]
#     ),
# )
# print(result)
