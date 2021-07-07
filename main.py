import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sko.GA import GA

from ui import timewindow, try1, windowstime_cal, cover, draw_cover, cover_cumulate, draw_cover_cumulate

xyset = []
polygonset_all = [list() for x in range(9)]
windows_time = []  # 具体时间窗口 9个为一组
windows_time_digit = []  # 时间窗口大小
windows_time_differ = []  # 具体时间间隙
windows_time_last = []
windows_time_text = []
windows_time_differ_text = []
windows_max_time = ""
windows_min_time = ""
windows_sum_time = ""
windows_max_differtime = ""
windows_sum_differtime = ""
windows_max_index = 0
windows_min_index = 0
windowsdiffer_max_index = 0
satellite_rec_range = []
cover_rate_list = []
cover_rate_list_cumulate = []
num_satellite = 9
onehour_list_all = []
twohour_list_all = []
threehour_list_all = []
namelist = ["坎普尔 印度", "堪萨斯城 美国", "高雄 中国", "卡拉奇 巴基斯坦", "加德满都餐厅 尼泊尔", "考纳斯 立陶宛", "川崎 日本", "喀山 俄罗斯", "喀士穆 苏丹", "孔敬 泰国",
            "库尔纳 孟加拉国", "基加利 卢旺达", "京斯敦 澳大利亚"]


class Point:
    lng = ''
    lat = ''

    def __init__(self, lng, lat):
        self.lng = lng  # 经度
        self.lat = lat  # 纬度


def split_num_l(num_lst):
    num_lst_tmp = [int(n) for n in num_lst]
    sort_lst = sorted(num_lst_tmp)  # ascending
    len_lst = len(sort_lst)
    i = 0
    split_lst = []

    tmp_lst = [sort_lst[i]]
    while True:
        if i + 1 == len_lst:
            break
        next_n = sort_lst[i + 1]
        if sort_lst[i] + 1 == next_n:
            tmp_lst.append(next_n)
        else:
            split_lst.append(tmp_lst)
            tmp_lst = [next_n]
        i += 1
    split_lst.append(tmp_lst)
    return split_lst


def read(x):
    count = 0
    polyList = []
    polygonset = []
    with open(r"E://Data(1)/SatCoverInfo_%d.txt" % x, "rb") as f:
        temp = []
        for line in f:
            line = line.strip(b"\n")
            temp.append(line)
            count = count + 1
            if count % 22 == 0:
                polyList.append(temp)
                temp = []
    for each in polyList:
        points = []
        for i in each[1::]:
            i = i.strip(b"\t")
            strlist = i.split(b"\t")
            temp = Point(float(strlist[0]), float(strlist[1]))
            points.append(temp)
        polygonset.append(points)
    polygonset_all[x] = polygonset


def time_cal(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def time_cal_reverse(time):
    resultlist = time.split(":")
    return int(resultlist[0]) * 3600 + int(resultlist[1]) * 60 + int(resultlist[2])


def getPolygonBounds(points):
    length = len(points)
    top = down = left = right = points[0]
    for i in range(1, length):
        if points[i].lng > top.lng:
            top = points[i]
        elif points[i].lng < down.lng:
            down = points[i]
        else:
            pass
        if points[i].lat > right.lat:
            right = points[i]
        elif points[i].lat < left.lat:
            left = points[i]
        else:
            pass

    point0 = Point(top.lng, left.lat)
    point1 = Point(top.lng, right.lat)
    point2 = Point(down.lng, right.lat)
    point3 = Point(down.lng, left.lat)
    polygonBounds = [point0, point1, point2, point3]
    return polygonBounds


# 判断点是否在外包矩形外
def isPointInRect(point, polygonBounds):
    if polygonBounds[3].lng <= point.lng <= polygonBounds[0].lng and polygonBounds[3].lat <= point.lat <= polygonBounds[
        2].lat:
        return True
    else:
        return False


def isPointInRect1(point, polygonBounds):
    if polygonBounds[3].lng <= point[0] <= polygonBounds[0].lng and polygonBounds[3].lat <= point[1] <= polygonBounds[
        2].lat:
        return True
    else:
        return False


# 采用射线法判断点集里的每个点是否在多边形集内，返回在多边形集内的点集
def isPointsInPolygons(polygonset, j):
    global windows_time
    polygonBounds_list = satellite_rec_range[j]
    temp_xyset = xyset
    # for points in polygonset:
    #     polygonBounds_list.append(getPolygonBounds(points))
    for point in temp_xyset:  # 每一个点
        count = 0
        windows_time_each = []
        for points in polygonset:  # 一颗卫星共包含84000个多边形
            # 求外包矩形
            # polygonBounds = getPolygonBounds(points)
            # 判断是否在外包矩形内，如果不在，直接返回false
            if not isPointInRect(point, polygonBounds_list[count]):
                count = count + 1
                continue
            length = len(points)
            p = point
            p1 = points[0]
            flag = False
            for i in range(1, length):
                p2 = points[i]
                # 点与多边形顶点重合
                if (p.lng == p1.lng and p.lat == p1.lat) or (p.lng == p2.lng and p.lat == p2.lat):
                    windows_time_each.append(count)
                    break
                # 判断线段两端点是否在射线两侧
                if (p2.lat < p.lat <= p1.lat) or (p2.lat >= p.lat > p1.lat):
                    # 线段上与射线 Y 坐标相同的点的 X 坐标
                    if p2.lat == p1.lat:
                        x = (p1.lng + p2.lng) / 2
                    else:
                        # x = p2.lng + (p.lat - p2.lat)*(p1.lng - p2.lng)/(p1.lat -p.lat)
                        x = p2.lng - (p2.lat - p.lat) * (p2.lng - p1.lng) / (p2.lat - p1.lat)  # 重要,射线的坐标
                    # 点在多边形的边上
                    if x == p.lng:
                        windows_time_each.append(count)
                        break
                    # 射线穿过多边形的边界
                    if x > p.lng:
                        flag = not flag
                p1 = p2
            if flag:
                windows_time_each.append(count)
            count = count + 1
        windows_time.append(windows_time_each)


def recal_time(single_windows_time):  # 单个地点计算时间窗口和时间间隙
    xyset_time_windows = []
    for i in range(9):
        xyset_time_windows.extend(single_windows_time[i])
    xyset_time_windows = list(set(xyset_time_windows))
    xyset_time_windows.sort()
    time_all = [i for i in range(84001)]
    xyset_time_windows_differ = list(set(time_all) - set(xyset_time_windows))  # 单个地点的时间间隙
    xyset_time_windows = split_num_l(xyset_time_windows)  # 分组
    windows_time_last.append(xyset_time_windows)
    xyset_time_windows_differ = split_num_l(xyset_time_windows_differ)
    windows_time_differ.append(xyset_time_windows_differ)


def init():
    global xyset
    start = time.time()

    for x in range(num_satellite):
        # profile = LineProfiler(read)
        # profile.runcall(read, x)
        # profile.print_stats()
        read(x)
    end = time.time()
    print(end - start)
    xyList = [  # 80.14,26.27
        "80.14 26.27", "265.67 39.02", "120.27 23.03", "67.02 24.51", "85.19 27.42", "23.54 54.54", "139.43 35.32",
        "49.1 55.45", "32.36 15.34", "102.5 16.25", "89.34 22.49", "30.05 -1.59", "167.58 -29.03"]
    for line in xyList:
        line = line.strip()
        xy = line.split()
        if len(xy) != 2:
            continue
        try:
            x = float(xy[0])
            y = float(xy[1])
        except ValueError:
            continue
        point = Point(x, y)
        xyset.append(point)
    global satellite_rec_range
    for each in polygonset_all:
        temp = []
        for points in each:
            temp.append(getPolygonBounds(points))
        satellite_rec_range.append(temp)


def analysize():  # 求时间窗口最大，最小，累计值，时间间隙最大，累计值
    global windows_max_time
    global windows_min_time
    global windows_max_index
    global windows_min_index
    global windows_sum_time
    global windows_max_differtime
    global windowsdiffer_max_index
    global windows_sum_differtime
    count = 0
    time_list = []
    for i in windows_time_last:  # 每个点
        single_sum = 0
        print(namelist[count], ":")
        temp_list = []
        for each in i:
            single_sum = single_sum + each[-1] - each[0]
            # print(time_cal(each[0]), "->", time_cal(each[-1]))
            temp = time_cal(each[0]), "->", time_cal(each[-1])
            temp_list.append(temp)
        windows_time_text.append(temp_list)
        time_list.append(single_sum)
        count = count + 1
    # print("其中，时间窗口最大值为：")
    windows_max_time = time_cal(max(time_list))
    windows_min_time = time_cal(min(time_list))
    windows_max_index = time_list.index(max(time_list))
    windows_min_index = time_list.index(min(time_list))
    # print(time_cal(max(time_list)))
    # print("其中，时间窗口最小值为：")
    # print(time_cal(min(time_list)))
    sum_time = 0
    for i in time_list:
        sum_time = sum_time + i
    # print("其中，时间窗口累计值为：")
    windows_sum_time = time_cal(sum_time)
    # print(time_cal(sum_time))
    # print("时间间隙为:")
    count = 0
    time_list_differ = []
    for i in windows_time_differ:  # 每个点
        single_sum_differ = 0
        # print(namelist[count], ":")
        temp_list = []
        for each in i:
            single_sum_differ = single_sum_differ + each[-1] - each[0]
            # print(time_cal(each[0]), "->", time_cal(each[-1]))
            temp = time_cal(each[0]), "->", time_cal(each[-1])
            temp_list.append(temp)
        windows_time_differ_text.append(temp_list)
        time_list_differ.append(single_sum_differ)
        count = count + 1
    # print("其中，时间窗口间隙最大值为：")
    windows_max_differtime = time_cal(max(time_list_differ))
    windowsdiffer_max_index = time_list_differ.index(max(time_list_differ))
    # print(time_cal(max(time_list_differ)))
    # print("其中，时间窗口间隙累计值为：")
    sum_time = 0
    for i in time_list_differ:
        sum_time = sum_time + i
    # print(time_cal(sum_time))
    windows_sum_differtime = time_cal(sum_time)


def windows():
    start = time.time()  # len(polygonset_all)
    for i in range(len(polygonset_all)):
        print("第%d颗卫星" % i)
        # profile = LineProfiler(isPointsInPolygons)
        # profile.runcall(isPointsInPolygons, polygonset_all[i], i)
        # profile.print_stats()
        isPointsInPolygons(polygonset_all[i], i)  # 求出对应地点的所有时间窗口
    temp_list = []
    for i in range(len(xyset)):
        temp_list.append(windows_time[i::13])
    for i in temp_list:  # 计算时间窗口和间隙
        recal_time(i)
    print("判断时间为：", time.time() - start)
    analysize()


def isPointsInPolygons_2(point, polygonset, windows_time2, j):
    polygonBounds_list = satellite_rec_range[j]
    # for points in polygonset:
    #     polygonBounds_list.append(getPolygonBounds(points))
    # for point in allset:  # 每一个点
    count = 0
    windows_time_each = []
    for points in polygonset:  # 一颗卫星共包含84000个多边形
        # 求外包矩形
        # 判断是否在外包矩形内，如果不在，直接返回false
        if not isPointInRect(point, polygonBounds_list[count]):
            count = count + 1
            continue
        length = len(points)
        p = point
        p1 = points[0]
        flag = False
        for i in range(1, length):
            p2 = points[i]
            # 点与多边形顶点重合
            if (p.lng == p1.lng and p.lat == p1.lat) or (p.lng == p2.lng and p.lat == p2.lat):
                windows_time_each.append(count)
                break
            # 判断线段两端点是否在射线两侧
            if (p2.lat < p.lat <= p1.lat) or (p2.lat >= p.lat > p1.lat):
                # 线段上与射线 Y 坐标相同的点的 X 坐标
                if p2.lat == p1.lat:
                    x = (p1.lng + p2.lng) / 2
                else:
                    x = p2.lng - (p2.lat - p.lat) * (p2.lng - p1.lng) / (p2.lat - p1.lat)  # 重要,射线的坐标
                # 点在多边形的边上
                if x == p.lng:
                    windows_time_each.append(count)
                    break
                # 射线穿过多边形的边界
                if x > p.lng:
                    flag = not flag
            p1 = p2
        if flag:
            windows_time_each.append(count)
        count = count + 1
    windows_time2.append(windows_time_each)


def recal_time2(windows_time2):
    xyset_time_windows = []
    for i in range(9):
        xyset_time_windows.extend(windows_time2[i])
    xyset_time_windows = list(set(xyset_time_windows))
    xyset_time_windows.sort()
    time_all = [i for i in range(84001)]
    xyset_time_windows_differ = list(set(time_all) - set(xyset_time_windows))  # 单个地点的时间间隙
    # windows_time_last.append(xyset_time_windows)
    windows_differ = split_num_l(xyset_time_windows_differ)
    max_value = 0
    # max_content = []
    count = 0
    for each in windows_differ:  # 每个点
        if len(each) > max_value:
            max_value = len(each)
            # max_content = each
        count = count + 1
    return max_value - 1
    # print(time_cal(each[0]), "->", time_cal(each[-1]))
    #     temp_list.append(temp)
    #     windows_time_differ_text.append(temp_list)
    #     time_list_differ.append(single_sum_differ)
    #     count = count + 1
    #     # print("其中，时间窗口间隙最大值为：")
    # windows_max_differtime = time_cal(max(time_list_differ))
    # windowsdiffer_max_index = time_list_differ.index(max(time_list_differ))
    pass


def recal_time3(windows_time2):
    xyset_time_windows = []
    for i in range(9):
        xyset_time_windows.extend(windows_time2[i])
    xyset_time_windows = list(set(xyset_time_windows))
    xyset_time_windows.sort()
    time_all = [i for i in range(84001)]
    xyset_time_windows_differ = list(set(time_all) - set(xyset_time_windows))  # 单个地点的时间间隙
    # windows_time_last.append(xyset_time_windows)
    windows_differ = split_num_l(xyset_time_windows_differ)
    sum_value = 0
    for each in windows_differ:  # 每个点
        sum_value = sum_value + ((each[-1] - each[0]) ** 2)
    return sum_value
    # print(time_cal(each[0]), "->", time_cal(each[-1]))
    #     temp_list.append(temp)
    #     windows_time_differ_text.append(temp_list)
    #     time_list_differ.append(single_sum_differ)
    #     count = count + 1
    #     # print("其中，时间窗口间隙最大值为：")
    # windows_max_differtime = time_cal(max(time_list_differ))
    # windowsdiffer_max_index = time_list_differ.index(max(time_list_differ))
    pass


def schaffer(p):
    """
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    """
    x1, x2 = p

    point = Point(x1, x2)
    windows_time2 = []
    start = time.time()
    for i in range(9):
        isPointsInPolygons_2(point, polygonset_all[i], windows_time2, i)  # 求时间窗口
    print(time.time() - start)
    start = time.time()
    # windows_differ = recal_time2(windows_time2)
    value = recal_time2(windows_time2)
    print(time.time() - start)
    return value


def schaffer_2(p):
    """
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    """
    x1, x2 = p

    point = Point(x1, x2)
    windows_time2 = []
    start = time.time()
    for i in range(9):
        isPointsInPolygons_2(point, polygonset_all[i], windows_time2, i)  # 求时间窗口
    print(time.time() - start)
    start = time.time()
    # windows_differ = recal_time2(windows_time2)
    value = recal_time3(windows_time2)
    print(time.time() - start)
    return value


def start_to_cal_windowstime_pow():
    start = time.time()
    ga = GA(func=schaffer_2, n_dim=2, size_pop=20, max_iter=50, lb=[75, 0], ub=[135, 55], precision=1e-1)
    best_x, best_y = ga.run()
    print(time.time() - start)
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min().cummin().plot(kind='line')
    plt.show()
    return time.time() - start


def start_to_cal_windowstime():  # 92 49 2450 2464.68 97.4 48.1
    # temp_point = Point(97.4, 48.1)
    # windows_time2 = []
    # start = time.time()
    # for i in range(9):
    # isPointsInPolygons_2(temp_point, polygonset_all[i], windows_time2, i)  # 求时间窗口
    # value= recal_time2(windows_time2)
    # print(time.time() - start)
    # print(windows_time2)
    # print(value)
    # print(content)
    start = time.time()
    # windows_differ = recal_time2(windows_time2)
    # value = recal_time2(windows_time2)
    ga = GA(func=schaffer, n_dim=2, size_pop=20, max_iter=20, lb=[75, 0], ub=[135, 55], precision=1e-1)
    best_x, best_y = ga.run()
    print(time.time() - start)
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min().cummin().plot(kind='line')
    plt.show()
    return time.time() - start

    # 在经度75°E-135°E，纬度范围0°N-55°N的区域范围内，寻找时间间隙最大值最小的点的坐标，以及时间间隙的平方和最小的点的坐标，经纬度精确到0.1°。
    # windows_time2 = []
    # points2 = []
    # start = time.time()
    # for i in np.arange(75, 135, 0.1):
    #     for j in np.arange(0, 55, 0.1):
    #         i = round(i, 2)
    #         j = round(j, 2)
    #         point = Point(i, j)
    #         points2.append(point)
    # print(len(points2))  # 生成所有的点
    # print(time.time() - start)
    # start = time.time()
    # for i in range(len(polygonset_all)):
    #     isPointsInPolygons_2(points2, polygonset_all[i], windows_time2)
    # print(time.time() - start)
    # pass


class windowstime_cal1(QDialog, windowstime_cal.Ui_windowstime_cal):
    def __init__(self):
        super(windowstime_cal1, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.start_cal)
        self.pushButton_2.clicked.connect(self.start_cal_pow)

    def start_cal(self):
        time = start_to_cal_windowstime()
        self.label_time1.setText(str(time))

        pass

    def start_cal_pow(self):
        time = start_to_cal_windowstime_pow()
        self.label_time2.setText(str(time))
        pass


def isPointsInPolygons_3(point, time):
    # for points in polygonset:
    #     polygonBounds_list.append(getPolygonBounds(points))
    # for point in allset:  # 每一个点
    count1 = 0
    # windows_time_each = []
    xyset = [[point[0], point[1]], [point[0] + point[2], point[1]], [point[0], point[1] + point[2]],
             [point[0] + point[2], point[1] + point[2]]]
    polygonset = []
    for i in range(num_satellite):
        polygonset.append(polygonset_all[i][time])  # 9颗卫星
    for xy in xyset:
        count = 0
        for points in polygonset:  # 每颗卫星同一时刻
            # 求外包矩形
            # 判断是否在外包矩形内，如果不在，直接返回false
            polygonBounds_list = satellite_rec_range[count][time]
            if not isPointInRect1(xy, polygonBounds_list):
                count = count + 1
                continue
            length = len(points)
            p = xy
            p1 = points[0]
            flag = False
            for i in range(1, length):
                p2 = points[i]
                # 点与多边形顶点重合
                if (p[0] == p1.lng and p[1] == p1.lat) or (p[0] == p2.lng and p[1] == p2.lat):  # 在里面，被覆盖
                    count1 = count1 + 1
                    break
                # 判断线段两端点是否在射线两侧
                if (p2.lat < p[1] <= p1.lat) or (p2.lat >= p[1] > p1.lat):
                    # 线段上与射线 Y 坐标相同的点的 X 坐标
                    if p2.lat == p1.lat:
                        x = (p1.lng + p2.lng) / 2
                    else:
                        x = p2.lng - (p2.lat - p[1]) * (p2.lng - p1.lng) / (p2.lat - p1.lat)  # 重要,射线的坐标
                    # 点在多边形的边上
                    if x == p[0]:
                        count1 = count1 + 1
                        break
                    # 射线穿过多边形的边界
                    if x > p[0]:
                        flag = not flag
                p1 = p2
            if flag:
                count1 = count1 + 1
                break
            count = count + 1
    if count1 == 0:  # 该网格四个点都不在内部
        return 0
    elif count1 == 4:  # 该网格四个点都在内部
        return 1
    else:
        return 2  # 不确定该网格四个点在内部


def isPointsInPolygons_4(point, time):
    # for points in polygonset:
    #     polygonBounds_list.append(getPolygonBounds(points))
    # for point in allset:  # 每一个点
    count1 = 0
    # windows_time_each = []
    xyset = [[point[0], point[1]], [point[0] + point[2], point[1]], [point[0], point[1] + point[2]],
             [point[0] + point[2], point[1] + point[2]]]
    polygonset = []
    for i in range(num_satellite):
        polygonset.append(polygonset_all[i][time])  # 9颗卫星
    for xy in xyset:
        count = 0
        for points in polygonset:  # 每颗卫星同一时刻
            # 求外包矩形
            # 判断是否在外包矩形内，如果不在，直接返回false
            polygonBounds_list = satellite_rec_range[count][time]
            if not isPointInRect1(xy, polygonBounds_list):
                count = count + 1
                continue
            length = len(points)
            p = xy
            p1 = points[0]
            flag = False
            for i in range(1, length):
                p2 = points[i]
                # 点与多边形顶点重合
                if (p[0] == p1.lng and p[1] == p1.lat) or (p[0] == p2.lng and p[1] == p2.lat):  # 在里面，被覆盖
                    count1 = count1 + 1
                    break
                # 判断线段两端点是否在射线两侧
                if (p2.lat < p[1] <= p1.lat) or (p2.lat >= p[1] > p1.lat):
                    # 线段上与射线 Y 坐标相同的点的 X 坐标
                    if p2.lat == p1.lat:
                        x = (p1.lng + p2.lng) / 2
                    else:
                        x = p2.lng - (p2.lat - p[1]) * (p2.lng - p1.lng) / (p2.lat - p1.lat)  # 重要,射线的坐标
                    # 点在多边形的边上
                    if x == p[0]:
                        count1 = count1 + 1
                        break
                    # 射线穿过多边形的边界
                    if x > p[0]:
                        flag = not flag
                p1 = p2
            if flag:
                count1 = count1 + 1
                break
            count = count + 1
    if count1 == 0:  # 该网格四个点都不在内部
        return 0
    elif count1 == 4:  # 该网格四个点都在内部
        return 1
    else:
        return 2  # 不确定该网格四个点在内部


# windows_time2.append(windows_time_each)


def whether_is_cover(list, time, sum_area):
    unsure_list = []
    cover_list = []
    notcover_list = []
    # temp_list = []
    for i in list:
        # print(isPointsInPolygons_3([255.4, -5.4, 5], time))
        num = isPointsInPolygons_3(i, time)
        if num == 2:
            unsure_list.append(i)
        if num == 1:
            cover_list.append(i)
        if num == 0:
            notcover_list.append(i)
    if len(unsure_list) == 0:
        if len(cover_list) == 0:
            return 0
        else:
            return (len(cover_list) * (cover_list[0][2] ** 2)) / sum_area
    unsure_area = len(unsure_list) * (unsure_list[0][2] ** 2)
    # while unsure_area / sum_area >= 0.01:
    for i in unsure_list:
        temp_list = [[i[0], i[1], i[2] / 2], [i[0] + i[2] / 2, i[1], i[2] / 2], [i[0], i[1] + i[2] / 2, i[2] / 2],
                     [i[0] + i[2] / 2, i[1] + i[2] / 2, i[2] / 2]]
        unsure_list.pop(0)
        # unsure_list = []
        for j in temp_list:
            # print(isPointsInPolygons_3([255.4, -5.4, 5], time))
            num = isPointsInPolygons_3(j, time)
            if num == 2:
                unsure_list.append(j)
            if num == 1:
                cover_list.append(j)
            if num == 0:
                notcover_list.append(j)
        for j in unsure_list:
            unsure_area = unsure_area + j[2] ** 2
        if unsure_area / sum_area <= 0.01:
            break
        unsure_area = 0
    cover_area = 0
    for i in cover_list:
        cover_area = cover_area + i[2] ** 2
    list = [cover_area / sum_area, unsure_list, notcover_list, cover_list]
    return list


def whether_is_cover2(list, time, sum_area):
    unsure_list = []
    cover_list = []
    notcover_list = []
    # temp_list = []
    for i in list:
        # print(isPointsInPolygons_3([255.4, -5.4, 5], time))
        num = isPointsInPolygons_4(i, time)
        if num == 2:
            unsure_list.append(i)
        if num == 1:
            cover_list.append(i)
        if num == 0:
            notcover_list.append(i)
    if len(unsure_list) == 0:
        if len(cover_list) == 0:
            return 0
        else:
            return (len(cover_list) * (cover_list[0][2] ** 2)) / sum_area
    unsure_area = len(unsure_list) * (unsure_list[0][2] ** 2)
    while unsure_area / sum_area >= 0.01:
        unsure_area = 0
        temp_list = []
        for i in unsure_list:
            temp_list.append([i[0], i[1], i[2] / 2])
            temp_list.append([i[0] + i[2] / 2, i[1], i[2] / 2])
            temp_list.append([i[0], i[1] + i[2] / 2, i[2] / 2])
            temp_list.append([i[0] + i[2] / 2, i[1] + i[2] / 2, i[2] / 2])
            unsure_list = []
            # unsure_list = []
            for j in temp_list:
                # print(isPointsInPolygons_3([255.4, -5.4, 5], time))
                num = isPointsInPolygons_4(j, time)
                if num == 2:
                    unsure_list.append(j)
                if num == 1:
                    cover_list.append(j)
                if num == 0:
                    notcover_list.append(j)
        for j in unsure_list:
            unsure_area = unsure_area + j[2] ** 2
        if unsure_area / sum_area <= 0.01:
            break
    cover_area = 0
    for i in cover_list:
        cover_area = cover_area + i[2] ** 2
    list = [cover_area / sum_area, unsure_list, notcover_list, cover_list]
    return list


def whether_is_cover1(list, time, sum_area):
    unsure_list = []
    cover_list = []
    notcover_list = []
    # temp_list = []
    for i in list:
        # print(isPointsInPolygons_3([255.4, -5.4, 5], time))
        num = isPointsInPolygons_3(i, time)
        if num == 2:
            unsure_list.append(i)
        if num == 1:
            cover_list.append(i)
        if num == 0:
            notcover_list.append(i)
    if len(unsure_list) == 0:
        if len(cover_list) == 0:
            return 0
        else:
            return (len(cover_list) * (cover_list[0][2] ** 2)) / sum_area
    unsure_area = len(unsure_list) * (unsure_list[0][2] ** 2)
    while unsure_area / sum_area >= 0.01:
        unsure_area = 0
        temp_list = []
        for i in unsure_list:
            temp_list.append([i[0], i[1], i[2] / 2])
            temp_list.append([i[0] + i[2] / 2, i[1], i[2] / 2])
            temp_list.append([i[0], i[1] + i[2] / 2, i[2] / 2])
            temp_list.append([i[0] + i[2] / 2, i[1] + i[2] / 2, i[2] / 2])
            unsure_list = []
            # unsure_list = []
            for j in temp_list:
                # print(isPointsInPolygons_3([255.4, -5.4, 5], time))
                num = isPointsInPolygons_3(j, time)
                if num == 2:
                    unsure_list.append(j)
                if num == 1:
                    cover_list.append(j)
                if num == 0:
                    notcover_list.append(j)
        for j in unsure_list:
            unsure_area = unsure_area + j[2] ** 2
        if unsure_area / sum_area <= 0.01:
            break
    cover_area = 0
    for i in cover_list:
        cover_area = cover_area + i[2] ** 2
    list = [cover_area / sum_area, unsure_list, notcover_list, cover_list]
    return list


def whether_is_cover_cumulate(list, time, sum_area):
    unsure_list = []
    cover_list = []
    notcover_list = []
    # temp_list = []
    for i in list:
        # print(isPointsInPolygons_3([255.4, -5.4, 5], time))
        num = isPointsInPolygons_3(i, time)
        if num == 2:
            unsure_list.append(i)
        if num == 1:
            cover_list.append(i)
        if num == 0:
            notcover_list.append(i)
    if len(unsure_list) == 0:
        if len(cover_list) == 0:
            return 0
        else:
            return (len(cover_list) * (cover_list[0][2] ** 2)) / sum_area
    unsure_area = len(unsure_list) * (unsure_list[0][2] ** 2)
    # while unsure_area / sum_area >= 0.01:
    while unsure_area / sum_area >= 0.01:
        unsure_area = 0
        temp_list = []
        for i in unsure_list:
            temp_list.append([i[0], i[1], i[2] / 2])
            temp_list.append([i[0] + i[2] / 2, i[1], i[2] / 2])
            temp_list.append([i[0], i[1] + i[2] / 2, i[2] / 2])
            temp_list.append([i[0] + i[2] / 2, i[1] + i[2] / 2, i[2] / 2])
            unsure_list = []
            # unsure_list = []
            for j in temp_list:
                # print(isPointsInPolygons_3([255.4, -5.4, 5], time))
                num = isPointsInPolygons_3(j, time)
                if num == 2:
                    unsure_list.append(j)
                if num == 1:
                    cover_list.append(j)
                if num == 0:
                    notcover_list.append(j)
        for j in unsure_list:
            unsure_area = unsure_area + j[2] ** 2
        if unsure_area / sum_area <= 0.01:
            break
    cover_area = 0
    for i in cover_list:
        cover_area = cover_area + i[2] ** 2
    list = [cover_area / sum_area, unsure_list, notcover_list, cover_list]
    return list
    pass


def start_to_cal_cover():
    global cover_rate_list
    # 考虑目标区域是一个经纬度范围。经度75°E-135°E，纬度范围0°N-55°N。
    # 1.分为5度的网格,2.第一次判断，找出不确定网格3.不确定的网格进行划分为四分之一4.判断网格的面积和总网格面积，5度为1，2.5度为1/4
    init_list = []
    # cover_rate = []
    # sum_area = 0
    for i in range(75, 135, 5):
        for j in range(0, 55, 5):
            temp = [i, j, 5]
            init_list.append(temp)
    sum_area = len(init_list) * init_list[0][2] ** 2
    result = []
    # time = 0
    y1 = []
    start = time.time()
    for i in range(84001):
        temp_result = whether_is_cover(init_list, i, sum_area)
        # cover_rate.append(whether_is_cover(init_list, i, sum_area))
        print(i)
        if temp_result == 0:
            print(temp_result)  # 全部没被覆盖
            y1.append(0)
        else:
            print(temp_result[0])
            y1.append(temp_result[0])
        result.append(temp_result)
    x1 = [i for i in range(84001)]
    plt.figure(figsize=(17, 8))
    plt.plot(x1, y1, 'r', label='cover_rate', linewidth=0.2)
    # plt.plot(x1, y1, 'ro-')
    plt.title('rate of cover')
    plt.xlabel('time')
    plt.ylabel('rate')
    plt.legend()
    plt.show()
    print(time.time() - start)
    cover_rate_list = result
    pass


def start_to_cal_cover_cumulate(cumulate_time):
    global cover_rate_list_cumulate
    # 考虑目标区域是一个经纬度范围。经度75°E-135°E，纬度范围0°N-55°N。
    # 1.分为5度的网格,2.第一次判断，找出不确定网格3.不确定的网格进行划分为四分之一4.判断网格的面积和总网格面积，5度为1，2.5度为1/4
    init_list = []
    # cover_rate = []
    # sum_area = 0
    for i in range(75, 135, 5):
        for j in range(0, 55, 5):
            temp = [i, j, 5]
            init_list.append(temp)
    sum_area = len(init_list) * init_list[0][2] ** 2
    result = []
    # time = 0
    # y1 = []
    start = time.time()
    for i in range(cumulate_time):
        print(i)
        temp_result = whether_is_cover_cumulate(init_list, i, sum_area)
        # cover_rate.append(whether_is_cover(init_list, i, sum_area))
        # print(i)
        # if temp_result == 0:
        #     print(temp_result)  # 全部没被覆盖
        #     y1.append(0)
        # else:
        #     print(temp_result[0])
        #     y1.append(temp_result[0])
        result.append(temp_result)
    print(time.time() - start)
    cover_rate_list_cumulate = result
    # cover_rate_list = result
    pass


def init_for_drawcover():
    init_list = []
    # cover_rate = []
    # sum_area = 0
    for i in range(75, 135, 5):
        for j in range(0, 55, 5):
            temp = [i, j, 5]
            init_list.append(temp)
    sum_area = len(init_list) * init_list[0][2] ** 2
    return whether_is_cover1(init_list, time_cal_reverse("11:20:38"), sum_area)
    pass


def cal_list(templist):
    result_list = []
    for i in templist:
        temp_list = []
        count = 0
        for j in i:
            value = j * 13
            if count == 0:
                value = value - 973
            if count == 1:
                value = value + 120
            temp_list.append(value)
            count = count + 1
        result_list.append(temp_list)
    return result_list
    pass


def cal_satellite(list):
    result_list = []
    for i in list:
        temp_list = []
        x = 13 * i.lng - 973
        y = 13 * i.lat + 120
        temp_list.append(x)
        temp_list.append(y)
        result_list.append(temp_list)
    return result_list
    # for i in list:
    #     temp_list = []
    #     count = 0
    #     for j in i:
    #         value = j * 13
    #         if count == 0:
    #             value = value - 973
    #         if count == 1:
    #             value = value + 120
    #         temp_list.append(value)
    #         count = count + 1
    #     result_list.append(temp_list)


def splitlist(list):
    split_list1 = []
    result_list = []
    for i in list:
        if i[2] == 5:
            split_list1.append([i[0], i[1], i[2] / 2])
            split_list1.append([i[0] + i[2] / 2, i[1], i[2] / 2])
            split_list1.append([i[0], i[1] + i[2] / 2, i[2] / 2])
            split_list1.append([i[0] + i[2] / 2, i[1] + i[2] / 2, i[2] / 2])
            for j in split_list1[0:4]:
                split_list1.append([j[0], j[1], j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1], j[2] / 2])
                split_list1.append([j[0], j[1] + j[2] / 2, j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1] + j[2] / 2, j[2] / 2])
            for j in split_list1[4:8]:
                split_list1.append([j[0], j[1], j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1], j[2] / 2])
                split_list1.append([j[0], j[1] + j[2] / 2, j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1] + j[2] / 2, j[2] / 2])
            for j in split_list1[8:12]:
                split_list1.append([j[0], j[1], j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1], j[2] / 2])
                split_list1.append([j[0], j[1] + j[2] / 2, j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1] + j[2] / 2, j[2] / 2])
        if i[2] == 2.5:
            split_list1.append([i[0], i[1], i[2] / 2])
            split_list1.append([i[0] + i[2] / 2, i[1], i[2] / 2])
            split_list1.append([i[0], i[1] + i[2] / 2, i[2] / 2])
            split_list1.append([i[0] + i[2] / 2, i[1] + i[2] / 2, i[2] / 2])
            for j in split_list1[0:4]:
                split_list1.append([j[0], j[1], j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1], j[2] / 2])
                split_list1.append([j[0], j[1] + j[2] / 2, j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1] + j[2] / 2, j[2] / 2])
            for j in split_list1[4:8]:
                split_list1.append([j[0], j[1], j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1], j[2] / 2])
                split_list1.append([j[0], j[1] + j[2] / 2, j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1] + j[2] / 2, j[2] / 2])
        if i[2] == 1.25:
            split_list1.append([i[0], i[1], i[2] / 2])
            split_list1.append([i[0] + i[2] / 2, i[1], i[2] / 2])
            split_list1.append([i[0], i[1] + i[2] / 2, i[2] / 2])
            split_list1.append([i[0] + i[2] / 2, i[1] + i[2] / 2, i[2] / 2])
            for j in split_list1[0:4]:
                split_list1.append([j[0], j[1], j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1], j[2] / 2])
                split_list1.append([j[0], j[1] + j[2] / 2, j[2] / 2])
                split_list1.append([j[0] + j[2] / 2, j[1] + j[2] / 2, j[2] / 2])
        if i[2] == 0.625:
            split_list1.append([i[0], i[1], i[2] / 2])
            split_list1.append([i[0] + i[2] / 2, i[1], i[2] / 2])
            split_list1.append([i[0], i[1] + i[2] / 2, i[2] / 2])
            split_list1.append([i[0] + i[2] / 2, i[1] + i[2] / 2, i[2] / 2])
    for id in list:
        if id not in split_list1:
            result_list.append(id)
    return result_list
    pass


def start():
    start_to_cal_cover_cumulate(10800)
    print("start to cal")
    onehour = 0
    twohour = 0
    threehour = 0
    onehour_list = []
    twohour_list = []
    threehour_list = []
    onehour_list_unsure = []
    twohour_list_unsure = []
    threehour_list_unsure = []
    onehour_list_uncover = []
    twohour_list_uncover = []
    threehour_list_uncover = []
    sum_area = 3300
    # print(cover_rate_list_cumulate)
    temp_list1 = []
    temp_list2 = []
    temp_list3 = []
    temp_list1_unsure = []
    temp_list2_unsure = []
    temp_list3_unsure = []
    temp_list1_uncover = []
    temp_list2_uncover = []
    temp_list3_uncover = []
    count = 0
    for i in cover_rate_list_cumulate:
        if count < 3600:
            if i != 0:
                if len(i[3]) != 0:
                    temp_list1.append(i[3])
                    temp_list1_unsure.append(i[1])
                    temp_list1_uncover.append(i[2])
        if count < 7200:
            if i != 0:
                if len(i[3]) != 0:
                    temp_list2.append(i[3])
                    temp_list2_unsure.append(i[1])
                    temp_list2_uncover.append(i[2])
        if count < 10800:
            if i != 0:
                if len(i[3]) != 0:
                    temp_list3.append(i[3])
                    temp_list3_unsure.append(i[1])
                    temp_list3_uncover.append(i[2])
        count = count + 1
    for i in temp_list1:
        for j in i:
            onehour_list.append(j)
    for i in temp_list2:
        for j in i:
            twohour_list.append(j)
    for i in temp_list3:
        for j in i:
            threehour_list.append(j)
    for i in temp_list1_unsure:
        for j in i:
            onehour_list_unsure.append(j)
    for i in temp_list2_unsure:
        for j in i:
            twohour_list_unsure.append(j)
    for i in temp_list3_unsure:
        for j in i:
            threehour_list_unsure.append(j)
    for i in temp_list1_uncover:
        for j in i:
            onehour_list_uncover.append(j)
    for i in temp_list2_uncover:
        for j in i:
            twohour_list_uncover.append(j)
    for i in temp_list3_uncover:
        for j in i:
            threehour_list_uncover.append(j)
    new_cover_list1 = []
    new_cover_list2 = []
    new_cover_list3 = []
    new_cover_list1_unsure = []
    new_cover_list2_unsure = []
    new_cover_list3_unsure = []
    new_cover_list1_uncover = []
    new_cover_list2_uncover = []
    new_cover_list3_uncover = []
    for id in onehour_list:
        if id not in new_cover_list1:
            new_cover_list1.append(id)
    for id in twohour_list:
        if id not in new_cover_list2:
            new_cover_list2.append(id)
    for id in threehour_list:
        if id not in new_cover_list3:
            new_cover_list3.append(id)
    for id in onehour_list_unsure:
        if id not in new_cover_list1_unsure:
            new_cover_list1_unsure.append(id)
    for id in twohour_list_unsure:
        if id not in new_cover_list2_unsure:
            new_cover_list2_unsure.append(id)
    for id in threehour_list_unsure:
        if id not in new_cover_list3_unsure:
            new_cover_list3.append(id)
    for id in onehour_list_uncover:
        if id not in new_cover_list1_uncover:
            new_cover_list1_uncover.append(id)
    for id in twohour_list_uncover:
        if id not in new_cover_list2_uncover:
            new_cover_list2_uncover.append(id)
    for id in threehour_list_uncover:
        if id not in new_cover_list3_uncover:
            new_cover_list3_uncover.append(id)

    one_hour_list1 = splitlist(new_cover_list1)
    two_hour_list1 = splitlist(new_cover_list2)
    three_hour_list1 = splitlist(new_cover_list3)
    one_hour_list1_unsure = splitlist(new_cover_list1_unsure)
    two_hour_list1_unsure = splitlist(new_cover_list2_unsure)
    three_hour_list1_unsure = splitlist(new_cover_list3_unsure)
    one_hour_list1_uncover = splitlist(new_cover_list1_uncover)
    two_hour_list1_uncover = splitlist(new_cover_list1_uncover)
    three_hour_list1_uncover = splitlist(new_cover_list1_uncover)

    onehour_list_all.append(one_hour_list1)
    onehour_list_all.append(one_hour_list1_unsure)
    onehour_list_all.append(one_hour_list1_uncover)
    twohour_list_all.append(two_hour_list1)
    twohour_list_all.append(two_hour_list1_unsure)
    twohour_list_all.append(two_hour_list1_uncover)
    threehour_list_all.append(three_hour_list1)
    threehour_list_all.append(three_hour_list1_unsure)
    threehour_list_all.append(three_hour_list1_uncover)

    for i in range(len(one_hour_list1)):
        onehour = onehour + (one_hour_list1[i][2] ** 2)
    for i in range(len(two_hour_list1)):
        twohour = onehour + (two_hour_list1[i][2] ** 2)
    for i in range(len(three_hour_list1)):
        threehour = onehour + (three_hour_list1[i][2] ** 2)

    print(onehour / sum_area, twohour / sum_area, threehour / sum_area)


class cover_cumulate_dialog(QDialog, cover_cumulate.Ui_cover_cumulate):
    def __init__(self):
        super(cover_cumulate_dialog, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(start)


def init_for_drawcover1():
    # init_list = []
    # # cover_rate = []
    # # sum_area = 0
    # for i in range(75, 135, 5):
    #     for j in range(0, 55, 5):
    #         temp = [i, j, 5]
    #         init_list.append(temp)
    # sum_area = len(init_list) * init_list[0][2] ** 2
    return
    # return whether_is_cover2(init_list, time_cal_reverse("01:00:00"), sum_area)
    pass


def draw_satellite(list, qp):
    pen = QPen(QColor(255, 255, 0), 2)
    qp.setPen(pen)
    for i in range(len(list) - 1):
        x1 = list[i][0]
        y1 = list[i][1]
        x2 = list[i + 1][0]
        y2 = list[i + 1][1]
        qp.drawLine(int(x1), int(y1), int(x2), int(y2))
    qp.drawLine(int(list[0][0]), int(list[0][1]), int(list[-1][0]), int(list[-1][1]))

    pass


class drawcover_cumulate_dialog(QDialog, draw_cover_cumulate.Ui_draw_cover_cumulate):
    def __init__(self):
        super(drawcover_cumulate_dialog, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.start)
        self.time = time_cal_reverse("01:00:00")
        # self.result = []
        # self.result = init_for_drawcover1()

    def start(self):
        text = self.lineEdit.text()
        self.time = time_cal_reverse(text)
        init_list = []
        # cover_rate = []
        # sum_area = 0
        for i in range(75, 135, 5):
            for j in range(0, 55, 5):
                temp = [i, j, 5]
                init_list.append(temp)

        # self.result = whether_is_cover2(init_list, self.time, sum_area)
        self.update()
        pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.begin(self)
        self.draw_line(painter)
        painter.end()

    def draw_line(self, qp):
        pen = QPen(QColor(200, 200, 200), 1)
        qp.setPen(pen)
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(QColor(100, 100, 100))
        qp.setBrush(brush)
        unsurelist = []
        coverlist = []
        # notcoverlist = []
        templist = []
        if self.result == 0:
            for i in range(75, 135, 5):
                for j in range(0, 55, 5):
                    temp = [i, j, 5]
                    templist.append(temp)
            notcoverlist = cal_list(templist)
        else:
            unsurelist = cal_list(twohour_list_all[1])
            notcoverlist = cal_list(twohour_list_all[2])
            coverlist = cal_list(twohour_list_all[0])
        for i in notcoverlist:  # 白色 未覆盖837
            length = i[2]
            qp.drawRect(int(i[0]), int(i[1]), int(length), int(length))
        pen = QPen(QColor(100, 100, 100), 2)
        qp.setPen(pen)
        brush.setColor(QColor(0, 0, 255))
        qp.setBrush(brush)
        for i in unsurelist:  # 蓝色 不确定
            length = i[2]
            qp.drawRect(int(i[0]), int(i[1]), int(length), int(length))
        brush.setColor(QColor(255, 0, 0))
        qp.setBrush(brush)
        for i in coverlist:  # 红色 覆盖
            length = i[2]
            qp.drawRect(int(i[0]), int(i[1]), int(length), int(length))
        # temp_satellite = polygonset_all[0][self.time]
        # satellite = cal_satellite(temp_satellite)
        # self.draw_satellite(satellite, qp)
        pass


def draw_satellite(list, qp):
    pen = QPen(QColor(255, 255, 0), 2)
    qp.setPen(pen)
    for i in range(len(list) - 1):
        x1 = list[i][0]
        y1 = list[i][1]
        x2 = list[i + 1][0]
        y2 = list[i + 1][1]
        qp.drawLine(int(x1), int(y1), int(x2), int(y2))
    qp.drawLine(int(list[0][0]), int(list[0][1]), int(list[-1][0]), int(list[-1][1]))

    pass


class drawcover_dialog(QDialog, draw_cover.Ui_Dialog):
    def __init__(self):
        super(drawcover_dialog, self).__init__()
        # self.list = cover_rate_list
        self.setupUi(self)
        self.pushButton.clicked.connect(self.start)
        # self.pushButton_2.setGraphicsEffect(op)
        # self.pushButton_2.
        self.time = time_cal_reverse("11:20:38")
        self.result = []
        self.result = init_for_drawcover()

    def start(self):
        text = self.lineEdit.text()
        self.time = time_cal_reverse(text)
        init_list = []
        # cover_rate = []
        # sum_area = 0
        for i in range(75, 135, 5):
            for j in range(0, 55, 5):
                temp = [i, j, 5]
                init_list.append(temp)
        sum_area = len(init_list) * init_list[0][2] ** 2
        self.result = whether_is_cover1(init_list, self.time, sum_area)
        self.update()
        pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.begin(self)
        self.draw_line(painter)
        painter.end()

    def draw_line(self, qp):
        pen = QPen(QColor(200, 200, 200), 1)
        qp.setPen(pen)
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(QColor(100, 100, 100))
        qp.setBrush(brush)
        unsurelist = []
        coverlist = []
        # notcoverlist = []
        templist = []
        if self.result == 0:
            for i in range(75, 135, 5):
                for j in range(0, 55, 5):
                    temp = [i, j, 5]
                    templist.append(temp)
            notcoverlist = cal_list(templist)
        else:
            unsurelist = cal_list(self.result[1])
            notcoverlist = cal_list(self.result[2])
            coverlist = cal_list(self.result[3])
        # print(self.result)
        # print(unsurelist)
        # print(coverlist)
        # print(notcoverlist)

        # qp.drawRect(2,120,67,185)
        for i in notcoverlist:  # 白色 未覆盖837
            length = i[2]
            qp.drawRect(int(i[0]), int(i[1]), int(length), int(length))
        pen = QPen(QColor(100, 100, 100), 2)
        qp.setPen(pen)
        brush.setColor(QColor(0, 0, 255))
        qp.setBrush(brush)
        for i in unsurelist:  # 蓝色 不确定
            length = i[2]
            qp.drawRect(int(i[0]), int(i[1]), int(length), int(length))
        brush.setColor(QColor(255, 0, 0))
        qp.setBrush(brush)
        for i in coverlist:  # 红色 覆盖
            length = i[2]
            qp.drawRect(int(i[0]), int(i[1]), int(length), int(length))

        # for i in polygonset_all:
        #     satellite_list.append(i[self.time])
        # print(satellite_list)
        temp_satellite = polygonset_all[0][self.time]
        satellite = cal_satellite(temp_satellite)
        draw_satellite(satellite, qp)
        # print(satellite)

        pass


def start():
    start_to_cal_cover()


class coverdialog1(QDialog, cover.Ui_Dialog):
    def __init__(self):
        super(coverdialog1, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(start)


class timewindowdialog(QDialog, timewindow.Ui_TimeWindow):
    def __init__(self):
        super(timewindowdialog, self).__init__()
        self.setupUi(self)
        self.label_windowsmax.setText(windows_max_time)
        self.label_windowsmin.setText(windows_min_time)
        self.label_windowssum.setText(windows_sum_time)
        self.label_windowsdiffermax.setText(windows_max_differtime)
        self.label_windowsdiffersum.setText(windows_sum_differtime)
        self.label_windowsmaxnation.setText(namelist[windows_max_index])
        self.label_windowsminnation.setText(namelist[windows_min_index])
        self.label_windowsdiffermaxnation.setText(namelist[windowsdiffer_max_index])
        self.comboBox.currentIndexChanged.connect(self.setText)

    def setText(self):
        index = self.comboBox.currentIndex()

        temp_text = ""
        for i in windows_time_text[index]:
            temp = ""
            temp = temp.join(i)
            temp_text = temp_text + temp + "\n"
        self.textEdit.setPlainText(temp_text)
        temp_text = ""
        for i in windows_time_differ_text[index]:
            temp = ""
            temp = temp.join(i)
            temp_text = temp_text + temp + "\n"
        self.textEdit_2.setPlainText(temp_text)


class maindialog(QMainWindow, try1.Ui_MainWindow):  # ??????????????????????????
    def __init__(self):
        super(maindialog, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(windows)
        self.pushButton_2.clicked.connect(self.TimeWindowDialog)
        self.pushButton_3.clicked.connect(self.windowstime_cal_dialog)
        self.pushButton_4.clicked.connect(self.cover_dialog1)
        self.pushButton_5.clicked.connect(self.draw_cover_dialog)
        self.pushButton_6.clicked.connect(self.cover_cumulate_dialog1)
        self.pushButton_7.clicked.connect(self.draw_cover_dialog_cumulate)
        self.pushButton.clicked.connect(init)

    def cover_dialog1(self):
        self.cover_dialog = coverdialog1()
        self.cover_dialog.show()

    def TimeWindowDialog(self):
        self.TimeWindowdialog = timewindowdialog()
        self.TimeWindowdialog.show()

    def windowstime_cal_dialog(self):
        self.timecal_dialog = windowstime_cal1()
        self.timecal_dialog.show()

    def draw_cover_dialog(self):
        self.draw_cover_Dialog = drawcover_dialog()
        self.draw_cover_Dialog.show()

    def cover_cumulate_dialog1(self):
        self.cover_cumulate_Dialog = cover_cumulate_dialog()
        self.cover_cumulate_Dialog.show()

    def draw_cover_dialog_cumulate(self):
        self.draw_cover_Dialog1 = drawcover_cumulate_dialog()
        self.draw_cover_Dialog1.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = maindialog()
    MainWindow.show()
    sys.exit(app.exec_())
