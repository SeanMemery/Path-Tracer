from json.encoder import INFINITY
import matplotlib.pyplot as plt 
import sys

text_file = open("ErrorLog.txt", "r")
lines = text_file.read().split(',')

remove_outliers = False
outliers_mult = 2.0
if (len(sys.argv) > 1 and "ro" in str(sys.argv)):
    remove_outliers = True
    outliers_mult = float(str(sys.argv[2]))


y = []
avg = INFINITY
ySum = 0
for count, line in enumerate(lines[1:len(lines)-1]):
    val = float(line[:])
    if (remove_outliers):
        if (val < outliers_mult*avg):
            y.append(val)
            ySum += val
            avg = ySum/(count+1)
    else:
        y.append(val)

plt.plot(range(0,len(y)), y)
plt.xlabel('Epoch')
plt.ylabel('RelMSE')
plt.title(lines[0]) 
plt.show()