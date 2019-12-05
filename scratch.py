import std
import statistics as stat 

check_num = [1, 2, 4, 7]

test1 = std.standard_dev(check_num)
test2 = stat.stdev(check_num)

print(f"Stats std num {test1}")
print(f"C --- std num {test2}")
