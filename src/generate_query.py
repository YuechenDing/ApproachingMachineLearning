def dfs_generate(template_string_list, current_index, current_generate_string):
    if current_index >= len(template_string_list):
        if current_generate_string[-1] != "了":
            print(current_generate_string)
        return None
    for string in slot_dict[template_string_list[current_index]]:
        dfs_generate(
                template_string_list, 
                current_index + 1, 
                current_generate_string + string)

slot_dict = {
    # "Entity": ["灯泡", "白炽灯"],
    # "Entity": ["中国最后一个状元", "最后一个状元", "清朝最后一个状元", "历史上最后一个状元"],
    # "Entity": ["第一颗原子弹", "第一颗核弹", "历史上第一颗原子弹", "历史上第一颗核弹"],
    "Entity": ["敦刻尔克大撤退"],
    "IsWho": ["是谁", "是哪个", "是哪位", "是哪个人", "是什么人", "是哪些人"],
    "Who": ["谁", "哪些人"],
    "WhoIs": ["谁是", "哪位是", "哪个是", "哪些人是"],
    "Action": ["发明了", "发明的", "研究出了", "研究出的", "创造出了", "创造出的", "造出了", "造出的"],
    "Question-Action2Noun": ["哪位发明家", "哪位科学家", "哪位研究者"],
    "IsQuestion-Action2Noun": ["是哪位发明家", "是哪位科学家", "是哪位研究者"],
    "Action2Noun": ["的发明家", "的创造者"],
    "Why-Begin": ["为什么"],
    "Why-End": ["的原因", "的原因是", "有什么原因", "的关键", "的关键原因", "的经验教训"],
    "Result": ["成功", "能成功", "成功实施", "得以成功实施"]
}
template_list = [
    "[Entity][Why-Begin][Result]",
    "[Entity][Result][Why-End]",
    "[Why-Begin][Entity][Result]"
]
for template_string in template_list:
    dfs_generate(template_string[1: -1].split("]["), 0, "")