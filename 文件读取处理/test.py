# 读取全部内容
with open("example.txt",'r',encoding='utf-8') as file:
    content = file.read()
    print(content)

print("============================================")

# 逐行读取内容
with open("example.txt",'r',encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        print(line,end='')

# 文件写入
with open("example.txt",'w',encoding='utf-8') as file:
    file.write("hello world")

with open("example.txt",'r',encoding='utf-8') as file:
    contents = file.read()
    print(contents)

lines = ["hello world", "world"]
with open("example.txt",'w',encoding='utf-8') as file:
    file.writelines(line + '\n' for line in lines)

with open("example.txt",'r',encoding='utf-8') as file:
    contents = file.read()
    print(contents)