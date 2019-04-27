from os import listdir
from os.path import isfile, join

folders = ["CP", "FD", "ITW"]
mypath = "/Users/kieranschubert/Desktop/Statistics/2nd_Semester/Data_Mining/2019/Sofamehack2019/Sub_DB_Checked/"
file_names = {}

for i in range(0, len(folders)):
	path = mypath + folders[i]
	file_names[i] = [f for f in listdir(path) if isfile(join(path, f))]

'''
print(file_names[0])
print(file_names[1])
print(file_names[2])

print(len(file_names[1]))
print(file_names[0][0])

for i in range(0, len(file_names[0])):
	reader.SetFilename(file_names[0][i])
	import example
'''
'''
print(file_names)
print(file_names[0][0])
print(len(file_names[0]))
print(len(file_names.keys()))
'''