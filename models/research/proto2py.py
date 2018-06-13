import os

init_path = os.getcwd()
print(init_path)

b_path = 'object_detection\protos\\'
proto_path = os.path.join(init_path,b_path)
f_list = os.listdir(proto_path)

for f in f_list:
    if '.proto' in f:
        print('process:' + f)
        cmd_line = 'protoc ' + b_path + f + ' --python_out=.'
        #print(cmd_line)
        os.system(cmd_line)
        
print('process done!')