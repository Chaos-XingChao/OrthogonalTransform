import sys
import numpy as np
import math


distance_list = {}

def loadMatrix(file_path):
	input_file = file(file_path, "r")
	temp_list = []
	for line in input_file:
		line = line.strip()
		wordList = line.split(' ')
		tempList = []
		for word in wordList:
			tempList.append(float(word))
		temp_list.append(tempList)
	input_file.close()
	return temp_list

def loadDict(file_path, key="None"):
	input_file = file(file_path, "r")
	temp_dict = {}
	temp_list = {}
	for line in input_file:		
		line = line.strip()
		wordList = line.split(' ')
		if len(wordList) == 2:
			continue
		temp_dict[wordList[0].strip()] = np.array(wordList[1:], dtype=float)
		if key != "None":
			temp_list[wordList[0]] = np.linalg.norm(np.array(wordList[1:], dtype=float))
	input_file.close()
	if key != "None":
		return temp_dict,temp_list
	else:
		return temp_dict

def getKnearest(target_vector, K, dest_dict, dest_norm):
	temp_dict = {}
	dest_matrix = np.array(dest_dict.values())
	result_list = np.dot(dest_matrix,target_vector)
	target_norm = np.linalg.norm(target_vector)

	i = 0
	for word, vector in dest_dict.items():
#		temp_dict[word] = 0.5 + 0.5  * np.dot(target_vector, np.array(vector)) / (np.linalg.norm(target_vector) * np.linalg.norm(np.array(vector)))		
#		temp_result =  np.dot(target_vector, np.array(vector))
		temp_dict[word] = result_list[i] / target_norm / dest_norm[word]
		i += 1
	temp_list = sorted(temp_dict.items(), key = lambda x:x[1], reverse = True)
	#, reverse = True)


	return temp_list[:K]

def cos(v1, v2):
	return np.dot(v1,v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

if __name__ == "__main__":
	#src test name
	file_name1 = sys.argv[1]	
	#dest test name
	file_name2 = sys.argv[2]	

	#src dict name
	file_name3 = sys.argv[3]	
	#dest dict name
	file_name4 = sys.argv[4]	

	#matrix path
	file_name5 = sys.argv[5]
	
	input_file1 = file(file_name1, 'r')
	input_file2 = file(file_name2, 'r')

	src_dict = loadDict(file_name3)
	dest_dict,dest_norm = loadDict(file_name4, key="Yes")
	matrix_list = loadMatrix(file_name5)

	src_list = []
	dest_list = []

	matrixW = np.array(matrix_list)

	for line in input_file1:
		line = line.strip()
		src_list.append(line)
	
	for line in input_file2:
		line = line.strip()
		dest_list.append(line)

	word_pair = {}

	for i in range(len(src_list)):
		if src_dict.has_key(src_list[i]) and dest_dict.has_key(dest_list[i]):
			word_pair[src_list[i]] = dest_list[i]
	err_count_1 = 0
	err_count_5 = 0
	total_count = 0

	for src_word,dest_word in word_pair.items():
		total_count += 1

		target_vector = np.dot(src_dict[src_word].T, matrixW)

		target_word_list = getKnearest(target_vector, 5, dest_dict, dest_norm)		
		temp_count = 0
		sys.stderr.write(".")
		for key, v in target_word_list:
			if key != dest_word and temp_count == 0:
				err_count_1 += 1
			elif key == dest_word:
				break
			temp_count += 1
		if temp_count == 5:
			err_count_5 += 1
	sys.stderr.write("\n")	
	print "%s\t%s\t%s\t%d\t%s\t%d\t%s\t%d" % (file_name3,file_name4,"1 error count is : ", err_count_1, "5 error count is : ", err_count_5, "total count is : ", total_count)
