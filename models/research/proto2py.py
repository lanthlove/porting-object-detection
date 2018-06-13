import os

init_path = os.getcwd()
path = 'object_detection\pycocotools\'
proto_path = os.path.join(init_path,path)
f_list = os.listdir(proto_path)

for f in f_list:
    if '.proto' in f:
        print('process:' + f)
        cmd_line = 'protoc ' + os.path.join(path, f) + ' --python_out=.'
        #print(cmd_line)
        os.system(cmd_line)
        
print('process done!')