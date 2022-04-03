import matplotlib.pyplot as plt 

text_file = open("ResTest.txt", "r")
lines = text_file.read().split(',')

y1 = []
y2 = []
x = []
count = 1
for line in lines[1:len(lines)-1]:
    val1 = float(line[0:4])
    val2 = float(line[6:10])
    
    x.append(count * 0.1)
    y1.append(val1)
    y2.append(val2)
    count += 1

plt.plot(x, y1, c="r", label="CUDA")
plt.plot(x, y2, c="b", label="SkePU CUDA")
plt.xticks([xx*0.1 for xx in list(range(0,len(x)+2))])
plt.xlabel('Res (Percentage of 1920x1080)')
plt.ylabel('MS for 1M Rays')
plt.title("Comparison of affect of resolution on performance of CUDA and SkePU CUDA") 
plt.legend()
plt.show()