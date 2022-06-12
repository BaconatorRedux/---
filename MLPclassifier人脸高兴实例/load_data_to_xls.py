import pandas
import numpy as np
from demo1 import flg_path
import xlwt

def load_data():
    print("调用load_data")
    label = pandas.read_csv(flg_path,header=None,sep=' ')
    lbls = label[0::] #提取第二、三列数据
    del lbls[0]
    lbls = np.array(lbls)#创建数组

    return lbls
a = load_data()
data = a.tolist()
wb = xlwt.Workbook(encoding='utf-8')
ws = wb.add_sheet('my sheet', cell_overwrite_ok=True)  # 新建新页面
for i in range(len(a)):
    ws.write(i,0,a[i][0])
    ws.write(i, 1, a[i][1])
    ws.write(i, 2, a[i][2])
wb.save('data.xls')
