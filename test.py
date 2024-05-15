
import json
with open('./fault_result_analysis1.json') as fin:
    example = json.load(fin)
    print(example[0])
    