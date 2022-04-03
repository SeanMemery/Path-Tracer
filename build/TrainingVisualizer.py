from json.encoder import INFINITY
import matplotlib.pyplot as plt 
import sys

text_file = open("ErrorLog.txt", "r")
lines = text_file.read().split(',')

remove_outliers = False
only_min = False
outliers_mult = 1.5

if "min" in str(sys.argv):
    only_min = True

elif "ro" in str(sys.argv):
    remove_outliers = True



y = []
x = []
avg = INFINITY
ySum = 0
min = INFINITY
for count, line in enumerate(lines[1:len(lines)-2]):
    val = float(line[:])

    if only_min:
        if val > min:
            continue
        min = val
        x.append(count)
        y.append(val)
        continue
    elif remove_outliers:
        ySum += val
        avg = ySum/(count+1)
        if val > outliers_mult*avg:
            continue
    
    x.append(len(x))
    y.append(val)

plt.plot(x, y)
plt.xlabel('Epoch')
plt.ylabel('RelMSE')
if only_min:
    plt.title(lines[0] + " (Only Min Values)") 
elif remove_outliers:
    plt.title(lines[0] + " (Outliers Stripped)") 
else:
    plt.title(lines[0]) 
plt.show()