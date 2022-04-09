import matplotlib.pyplot as plt 

text_file = open("ResTest.txt", "r")
lines = text_file.read().split(',')

y1 = []
y2 = []
x = []
count = 1
avg_diff = 0
for line in lines[0:len(lines)-1]:
    val1 = float(line[0:4])
    val2 = float(line[6:10])
    avg_diff += abs(val2 - val1)
    
    x.append(count)
    y1.append(val1)
    y2.append(val2)
    count += 1

print("Avg Diff = ", avg_diff/count)
plt.plot(x, y1, c="r", label="CUDA")
plt.plot(x, y2, c="b", label="SkePU CUDA")
plt.xticks([xx for xx in list(range(0,len(x)+2))])
# plt.xlabel('Res (Percentage of 1920x1080)')
# plt.ylabel('MS for 1M Rays')
# plt.title("Comparison of affect of resolution on performance of CUDA and SkePU CUDA") 
plt.xlabel('Max Depth')
plt.ylabel('MS for 1M Rays')
plt.title("Comparison of affect of max depth on performance of CUDA and SkePU CUDA") 
plt.legend()
plt.show()