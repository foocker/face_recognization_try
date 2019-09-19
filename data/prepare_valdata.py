from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
import numpy as np


def balance_index(val_data, val_label, rate=0.005, sd=42):
    np.random.seed(sd)
    index = np.arange(len(val_label))
    np.random.shuffle(index)
    # index = np.random.choice(index, len(val_label), replace=False)
    index_even = index[0::2]
    index_odd = index[1::2]
    ra = np.sum(val_label[index_even] == val_label[index_odd]) / len(index)
    # print("ra:", ra)
    """
    while ra >= rate:
        val_data, val_label, ra = balance_index(val_data, val_label, rate=rate)
        # print("final radnom rate:", ra)
    """
    """
    for _ in range(20):
        val_data, val_label, ra = balance_index(val_data, val_label, rate=rate)
        if ra < rate:
            break
    """
    val_data, val_label = val_data[index], val_label[index]
    return val_data, val_label, ra


def make_pair_index(label, sd=2):
    np.random.seed(sd)
    cla_count, choose_count = [], []    # class's number count
    # random choose class's number when the sum is equal to the half of all number of labels
    la_i, c, choose_sum = 0, 0, 0
    # get real count
    for i, la in enumerate(label):
        c += 1
        if la != la_i:
            cla_count.append(c)
            la_i += 1
            c = 0
        try:
            label[i+1]
        except:
            cla_count.append(c+1)
    cla_count[0] -= 1
    # get choosed count
    for c in cla_count:
        choosed = int(np.random.choice(c, 1)) + 1
        if 1 < choosed < c:
            if choosed % 2 == 1:
                choosed -= 1
        elif choosed == c:
            if choosed % 2 == 1:
                choosed -= 1
        # elif c == 1:
        #     continue
        else:
            choosed += 1    # all folder have more than two image
        choose_count.append(choosed)
        choose_sum += choosed
    # print("choosed:", choose_count, "\n", "sum of choose_count:",  np.sum(choose_count), "\n", "cla_count:", cla_count, "\n", "sum of cla_count:",  np.sum(cla_count))
    cut = len(cla_count) - len(choose_count)    # when the sum reach to half but the number of index is less
    cla_count, choose_count = np.array(cla_count), np.array(choose_count)
    if cut != 0:
        choose_count = np.hstack((choose_count, np.array([2]*cut)))    # all folder must have more than two image
    return cla_count, choose_count


def get_banalance_pair(val_data, val_label, cla_count, choose_count, seed=42, rate=0.005):
    val_data_u, val_label_u = np.zeros(shape=(0,3,112,112), dtype=val_data.dtype), np.zeros(shape=(0,), dtype=val_label.dtype)
    val_data_d, val_label_d = np.zeros(shape=(0,3,112,112), dtype=val_data.dtype), np.zeros(shape=(0,), dtype=val_label.dtype)
    cur = 0
    for cla, chc in zip(cla_count, choose_count):
        val_data_u = np.concatenate((val_data_u, val_data[cur:cur+chc]), axis=0)
        val_label_u = np.concatenate((val_label_u, val_label[cur:cur+chc]), axis=0)
        val_data_d = np.concatenate((val_data_d, val_data[cur+chc:cur+cla]), axis=0)
        val_label_d = np.concatenate((val_label_d, val_label[cur+chc:cur+cla]), axis=0)
        cur += cla
    val_data_d, val_label_d, ra = balance_index(val_data_d, val_label_d, rate=rate, sd=seed)
    # print("before val_data_d:", val_data_d.shape, val_label_d.shape)
    same_index = np.where(val_label_d[0::2] == val_label_d[1::2])
    # print("same index:", same_index)
    # print(val_label_d[2*same_index[0]], val_label_d[2*same_index[0]+1])
    diff_index = np.where(val_label_d[0::2] != val_label_d[1::2])
    # print("++++++", val_label_d[same_index].shape, "\n", val_data_d[same_index].shape)

    val_label_u = np.concatenate((val_label_u, val_label_d[2*same_index[0]], val_label_d[2*same_index[0]+1]), axis=0)

    val_data_u = np.concatenate((val_data_u, val_data_d[2*same_index[0]], val_data_d[2*same_index[0]+1]), axis=0)
    val_data_d = np.concatenate((val_data_d[2*diff_index[0]], val_data_d[2*diff_index[0]+1]), axis=0)
    val_label_d = np.concatenate((val_label_d[2*diff_index[0]], val_label_d[2*diff_index[0]+1]), axis=0)
    # print("after val_data_d", val_label_d.shape, val_data_d.shape)
    return val_data_u, val_label_u, val_data_d, val_label_d, ra


def gcd(m, n):
    if not isinstance(m, int) or not isinstance(n, int):
        raise TypeError
    if n == 0:
        m, n = n, m
    while m:
        m, n = n % m, m
    return n


def get_val_dataset(imgs_folder, sd=42, rate=0.005):
    np.random.seed(sd)
    val_data = []
    val_label = []
    val_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    val_data_folder = ImageFolder(imgs_folder, val_transform)
    
    i = 0
    for data, label in val_data_folder:
        val_data.append(data.numpy())
        val_label.append(label)
        i += 1
        if i == 10000:
            break
    
    val_label = np.array(val_label)
    val_data = np.array(val_data)
    print("++++++", val_label.shape, val_data.shape)

    cla_count, choose_count = make_pair_index(val_label, sd=sd)
    # print(cla_count, "\n", choose_count, len(cla_count), len(choose_count))
    # print("sum:", "real:", np.sum(cla_count), "choose:", np.sum(choose_count))

    val_data_u, val_label_u, val_data_d, val_label_d, ra = \
        get_banalance_pair(val_data, val_label, cla_count, choose_count, seed=sd, rate=rate)
    print(val_data_u.shape, val_data_d.shape, val_label_u.shape, val_label_d.shape)
    print("final rate is :{}".format((val_label_d.shape[0] * ra + val_label_u.shape) / val_label.shape[0]))

    d, u = val_label_d.shape[0] ,val_label_u.shape[0]
    assert d % 2 ==0 and u % 2 == 0
    if d > u:
        # print("d > u")
        val_data = np.concatenate((val_data_u, val_data_d[:u]), axis=0)
        val_label = np.array([True]*(u//2) + [False]*(u//2))
    else:
        # print("d <= u")
        val_data = np.concatenate((val_data_u[:d], val_data_d), axis=0)
        val_label = np.array([True]*(d//2) + [False]*(d//2))
    # print("?????", val_label.shape, val_data.shape)
    index = np.arange(len(val_label))
    np.random.shuffle(index)
    index_data = np.zeros(2*len(index), dtype=np.int64)
    index_data[0::2], index_data[1::2] = 2*index, 2*index+1
    val_data, val_label = val_data[index_data], val_label[index]
    # print("shape:::::", val_label.shape, val_data.shape)
    return val_data, val_label


def show_pair(val_data, val_label, index):
    # index :0-6000, 7000
    label = val_label[index]
    pair_img = val_data[2*index:2*index+2]

    max_x_0 = np.max(pair_img[0])
    min_x_0 = np.min(pair_img[0])
    max_x_1 = np.max(pair_img[1])
    min_x_1 = np.min(pair_img[1])
    """
    try:
        x_0 = (pair_img[0] - min_x_0) / (max_x_0 - min_x_0 + 1e-1)
        x_1 = (pair_img[1] - min_x_1) / (max_x_1 - min_x_1 + 1e-1)
    except:
        x_0 = pair_img[0]
        x_1 = pair_img[1]
    """
    if max_x_0 - min_x_0 != 0 and max_x_1 - min_x_1 != 0:
        x_0 = (pair_img[0] - min_x_0) / (max_x_0 - min_x_0)
        x_1 = (pair_img[1] - min_x_1) / (max_x_1 - min_x_1)
    else:
        x_0, x_1 = pair_img[0], pair_img[1]
    """
    plt.subplot(121)
    plt.imshow(np.transpose(x_0, (1, 2, 0)))
    plt.subplot(122)
    plt.imshow(np.transpose(x_1, (1, 2, 0)))
    
    
    plt.show()
    """
    print(label)


"""
val_data, val_label = get_val_dataset("./data/CASIA/train/")
print("shape++++", val_data.shape, val_label.shape)

for i in range(200, 230):
    show_pair(val_data, val_label, i)
"""




