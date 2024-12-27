# -*- coding: utf-8 -*-
# ===============================================================
#
#    @Create Author : chenyongming
#    @Create Time   : 2024-12-25 16:39
#    @Description   : Token相关数据
#    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
#
# ===============================================================


import tiktoken


def num_tokens_from_string(text, encoding_name='o200k_base'):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


if __name__ == '__main__':
    text = '这是八个中文文字'

    print(num_tokens_from_string("tiktoken is great!", "o200k_base"))
    print(num_tokens_from_string(text, "o200k_base"))

    encoding = tiktoken.get_encoding('o200k_base')
    print([encoding.decode([token]) for token in encoding.encode(text)])
