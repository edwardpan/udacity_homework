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
任务2: 哪个电话号码的通话总时间最长? 不要忘记，用于接听电话的时间也是通话时间的一部分。
输出信息:
"<telephone number> spent the longest time, <total time> seconds, on the phone during
September 2016.".

提示: 建立一个字典，并以电话号码为键，通话总时长为值。
这有利于你编写一个以键值对为输入，并修改字典的函数。
如果键已经存在于字典内，为键所对应的值加上对应数值；
如果键不存在于字典内，将此键加入字典，并将它的值设为给定值。
"""
total_times = {}
max_second = 0
max_second_phone = ""
for call in calls:
    call_second = int(call[3])

    call_from_time = total_times.get(call[0], 0)
    call_from_time_new = call_from_time + call_second
    total_times[call[0]] = call_from_time_new
    if call_from_time_new > max_second:
        max_second = call_from_time_new
        max_second_phone = call[0]

    call_to_time = total_times.get(call[1], 0)
    call_to_time_new = call_to_time + call_second
    total_times[call[1]] = call_to_time_new
    if call_to_time_new > max_second:
        max_second = call_to_time_new
        max_second_phone = call[1]

print("{} spent the longest time, {} seconds, on the phone during September 2016."
      .format(max_second_phone, max_second))
