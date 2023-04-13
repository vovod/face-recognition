import datetime
import cv2
import os

def write_logs(names, frame):
    today = datetime.datetime.now()
    day = today.strftime("%Y-%m-%d")
    with open('check.txt') as f:
        lines = f.read().splitlines()
    list_log = []
    for i in range(len(lines)):
        lines[i] = lines[i].split(" ")
        if str(day) == lines[i][1]:
            if lines[i][0] not in list_log:
                list_log.append(lines[i][0])
    for name in names:
        full_logs = str(name) + " " + str(today)
        # print(full_logs)
        if name not in list_log:
            path_log_images = "log_images"
            cv2.imshow("catch", frame)
            print("New log: " + full_logs)
            path_out = path_log_images + "/" + str(day)
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            file_name = name + ".jpg"
            save_path = path_out + "/" + file_name
            # print(save_path)
            cv2.imwrite(save_path, frame)
            f = open("check.txt", "a")
            f.write(full_logs + "\n")
            f.close()

# print(day)