import os
import subprocess


# 获取当前目录下的所有.py文件
# py_files = [file for file in os.listdir() if file.endswith('.py')]

# 排除当前脚本文件，如果当前文件名是 'run_scripts.py'
# py_files.remove('script.py')
# py_files.remove('proposed(1).py')
# py_files.remove('1031.py')

# 得到参数列表
# arg_list_1 = []
# wer_list = [0.005, 0.008, 0.01]



# for w in range(len(wer_list)):
#     for i in range(len(count)):
#         arguments = ['--wer', f'{wer_list[w]}', '--Tmax', f'{2.0}','--Emax', f'{1.5}','--num_clients','10','--num_epoch','200','--count_py',f'{count[i]}']
#         arg_list_1.append(arguments)

# for i in range(len(count)):
#     arguments = ['--wer', f'{0.0065}', '--Tmax', f'{2.0}','--Emax', f'{1.5}','--num_clients','5','--num_epoch','200','--count_py',f'{count[i]}']
#     arg_list_1.append(arguments)
    
        


# # 遍历所有的.py文件并运行它们
# for ind in range(len(arg_list_1)):
#     # for file in py_files:
#     #     print(f"Running {file} with argument {arg_list_1[ind]}")
#     #     subprocess.run(['python', file] + arg_list_1[ind])
#     print(f"Running with argument {arg_list_1[ind]}")



wer_list = [60000, 50000,100000, 10000]
T_maxs = [6.5, 9, 3, 25]
pat_list = ['exp1','exp2']
count = [c for c in range(2)]
arg_list_2 = []
# for i in range(len(wer_list)):
#     for j in range(len(pat_list)):
#         for z in range(len(count)):
#             arguments = ['--wer', f'{wer_list[i]}', '--Tmax', f'{T_maxs[i]}','--pattern', f'{pat_list[j]}', '--count_py', f'{count[z]}']
#             arg_list_2.append(arguments)
for i in range(len(wer_list)):
    for j in range(len(pat_list)):
        arguments = ['--wer', f'{wer_list[i]}', '--Rate', f'{wer_list[i]}', '--Tmax', f'{T_maxs[i]}','--pattern', f'{pat_list[j]}']
        arg_list_2.append(arguments)


file = './src/main.py'
for ind in range(len(arg_list_2)):
    print(f"Running {file} with argument {arg_list_2[ind]}")
    subprocess.run(['python', file] + arg_list_2[ind])

# file = 'avg_fresh.py'
# for ind in range(len(arg_list_1)):
#     print(f"Running {file} with argument {arg_list_1[ind]}")
#     subprocess.run(['python', file] + arg_list_1[ind])