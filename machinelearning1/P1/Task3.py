"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

"""
任务3:
(080)是班加罗尔的固定电话区号。
固定电话号码包含括号，
所以班加罗尔地区的电话号码的格式为(080)xxxxxxx。

第一部分: 找出被班加罗尔地区的固定电话所拨打的所有电话的区号和移动前缀（代号）。
 - 固定电话以括号内的区号开始。区号的长度不定，但总是以 0 打头。
 - 移动电话没有括号，但数字中间添加了
   一个空格，以增加可读性。一个移动电话的移动前缀指的是他的前四个
   数字，并且以7,8或9开头。
 - 电话促销员的号码没有括号或空格 , 但以140开头。

输出信息:
"The numbers called by people in Bangalore have codes:"
 <list of codes>
代号不能重复，每行打印一条，按字典顺序排序后输出。

第二部分: 由班加罗尔固话打往班加罗尔的电话所占比例是多少？
换句话说，所有由（080）开头的号码拨出的通话中，
打往由（080）开头的号码所占的比例是多少？

输出信息:
"<percentage> percent of calls from fixed lines in Bangalore are calls
to other fixed lines in Bangalore."
注意：百分比应包含2位小数。
"""

def is_bangalore_phone(phone):
    """判断是否是班加罗尔的固定电话
    :param phone:电话号码,字符串
    :return: True:是班加罗尔的固定电话,False:不是
    """
    return phone[:5] == "(080)"

def is_fixed_phone(phone):
    """判断是否是固定电话
    :param phone: 电话号码,字符串
    :return: True:是固定电话,False:不是
    """
    return "(" in phone and ")" in phone

def is_saler_phone(phone):
    """判断是否是电话促销员的号码
    :param phone: 电话号码,字符串
    :return: True:是电话促销员,False:不是
    """
    return str(phone).startswith("140")

def is_mobile_phone(phone):
    """判断是否是移动号码
    :param phone: 电话号码,字符串
    :return: True:是移动号码,False:不是
    """
    return not is_fixed_phone(phone) and " " in phone

def get_area_code(phone):
    """获取固定电话的区号
    :param phone: 电话号码,字符串
    :return: 区号
    """
    if is_fixed_phone(phone):
        end_index = phone.index(")")
        return phone[1:end_index]
    else:
        raise ValueError("not a fixed phone, no area code")

def get_mobile_phone_tag(phone):
    """获取移动号码的前缀
    :param phone: 电话号码,字符串
    :return: 移动号码前缀
    """
    if is_mobile_phone(phone):
        return phone[:4]
    else:
        raise ValueError("not a mobile phone, no tag")

def find_phone():
    """找出被班加罗尔地区的固定电话所拨打的所有电话的区号和移动前缀"""
    code_or_tags = set()
    for call in calls:
        from_phone = call[0]
        to_phone = call[1]
        if is_bangalore_phone(from_phone): # 主叫号码是班加罗尔地区
            if is_fixed_phone(to_phone): # 被叫是固定号码
                code_or_tags.add(get_area_code(to_phone))
            elif is_mobile_phone(to_phone): # 被叫是移动号码
                code_or_tags.add(get_mobile_phone_tag(to_phone))
    code_tags_list = list(code_or_tags)
    code_tags_list.sort()
    return code_tags_list

print("The numbers called by people in Bangalore have codes:")
phones = find_phone()
for phone in phones:
    print(phone)

def find_rate():
    """计算由班加罗尔固话打往班加罗尔固话在班加罗尔固话打出的所有号码中所占比例"""
    bangalore_phone = []
    total_phone = []
    for call in calls:
        from_phone = call[0]
        to_phone = call[1]
        if is_bangalore_phone(from_phone): # 主叫号码是班加罗尔地区
            total_phone.append(to_phone)
            if is_bangalore_phone(to_phone): # 被叫号码是班加罗尔地区
                bangalore_phone.append(to_phone)
    return round(len(bangalore_phone) / len(total_phone) * 100, 2)

print("{} percent of calls from fixed lines in Bangalore are calls to other fixed lines in Bangalore."
      .format(find_rate()))



