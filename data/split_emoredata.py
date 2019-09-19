import os, shutil, random

root = "E:\\faces_emore_sub\\train"


def getinfo(root):
    """
    root/1
    root/2
    ...
    :param root:
    :return:{"root/1":[img1, img2, ...], ...}
    """
    pathinfo = {}
    for dir in os.listdir(root):
        dic = os.path.join(root, dir)
        imgs = os.listdir(dic)
        pathinfo.update({dic: imgs})
    return pathinfo

# x = [1, 2, 3, 4]
# y = random.choice(x)
# yy = random.sample(x, 2)
# print(y, yy)


def emore_sub(pathinfo, rate=0.4):
    # move rate img from train set to val set
    for key, value in pathinfo.items():
        num = int(len(value)*rate)
        if num > 1:
            choosed = random.sample(value, num)
        else:
            choosed = [random.choice(value)]
        newdir = os.path.join(os.path.split(root)[0], "val", os.path.split(key)[1])
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        if len(choosed) > 1:
            for img in choosed:
                shutil.move(os.path.join(key, img), os.path.join(newdir, img))
        else:
            shutil.copyfile(os.path.join(key, choosed[0]), os.path.join(newdir, choosed[0]))


if __name__ == "__main__":
    pathinfo = getinfo(root)
    # print(pathinfo)
    emore_sub(pathinfo, rate=0.3)

