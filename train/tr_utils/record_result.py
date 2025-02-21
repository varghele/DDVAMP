import os


def record_result(string, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            print("successfully create record file!")
            f.write(string + "\n")
    with open(file_name, 'a') as f:
        f.write(string+"\n")
