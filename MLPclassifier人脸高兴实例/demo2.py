import xlwt

num = [1,2,6,0,0,3,3,5,6,2,1,6,2,2,6]
#tanh:0 relu:1 identity:2 logistic:3
activation = [0,0,0,1,1,2,0,0,3,2,2,2,0,1,0]
#constant:0 invscaling:1 adaptive:2
learning_rate = [0,1,0,2,1,0,2,2,2,2,0,0,1,2,2]

learning_rate_init = [0.001,0.002,0.007,0.003,0.003,0.003,0.005,0.002,0.07,0.003,0.002,0.1,0.005,0.07,0.03]

loss = [0.44471106237386543,0.6601892047874837,0.6523638302737353,0.6320869822826943,0.6588531546258939,0.5393902093614509,0.11632194974791198,0.016285099275617228,0.6766740252364878,0.192162412738693,0.5509328732692423,0.8,0.6758477807685416,0.6752175769413029,0.33842660698214483]
accs = [0.8383333333333334,0.6005555555555555,0.6116666666666667,0.6472222222222223,0.6058333333333333,0.6555555555555556,0.9697222222222223,1.0,0.6005555555555555,0.9355555555555556, 0.7319444444444444,0.39944444444444444,0.6005555555555555,0.6005555555555555,0.8869444444444444]
a = []
b = []
for i in loss:
    i = round(i,5)
    a.append(i)
for i in accs:
    i = round(i,5)
    b.append(i)

wb = xlwt.Workbook(encoding='utf-8')
ws = wb.add_sheet('my sheet', cell_overwrite_ok=True)  # 新建新页面

title = ['layer_num','activation','learning_rate','learning_rate_init','loss','assc']
for i in range(len(title)):
    ws.write(0,i,title[i])

for i in range(len(num)):
    if i ==0:
        continue
    ws.write(i,0,num[i])
    ws.write(i, 1, activation[i])
    ws.write(i, 2, learning_rate[i])
    ws.write(i, 3, learning_rate_init[i])
    ws.write(i, 4, a[i])
    ws.write(i, 5, b[i])


wb.save('result.xls')