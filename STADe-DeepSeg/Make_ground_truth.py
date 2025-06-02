# -*- coding:utf-8 -*-
"""
作者：LENOVO
日期：2022年 10月16日
题目：
解释：
"""

import json
import numpy as np
import pandas as pd



df_anno = pd.DataFrame(pd.read_csv(r"C:\Users\LENOVO\Desktop\硕士毕业论文\结果\击键识别\All-Amend_Backbone_I3D-num0-9_public\\test_Behave_Public.csv")).values[:]

Behave_Public_25 = {"database": {}}

for anno in df_anno:
    video_name = anno[0]
    Behave_Public_25["database"].update({video_name:{"subset": "test", "annotations": []}})

for anno in df_anno:
    start_frame = anno[-2]
    end_frame = anno[-1]
    class_idx = anno[1]
    video_name = anno[0]
    Behave_Public_25["database"][video_name]['annotations'].append(
        {"segment": [str(start_frame), str(end_frame)], "label": str(class_idx)})
# print(NumKeyStroke_18)

Behave_Public_25 = json.dumps(Behave_Public_25,indent=4,separators=(',', ': '))  # ,indent=4,separators=(',', ': ')

f = open('Behave_Public_25.json', 'w')
f.write(Behave_Public_25)
f.close()

# print(NumKeyStroke_18)

