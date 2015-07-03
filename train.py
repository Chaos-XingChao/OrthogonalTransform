import sphere_train as st
import sys
import numpy as np


def loadDict(file_name):
	input_file = file(file_name, "r")
	temp_dict = {}
	for line in input_file:
		line = line.strip()
		wordList = line.split(' ')
		if len(wordList) == 2:
			continue
		temp_dict[wordList[0]] = np.array(wordList[1:], dtype=float)
		temp_dict[wordList[0]] /= np.linalg.norm(temp_dict[wordList[0]])
	input_file.close()
	return temp_dict

def loadFile(file_name):
	input_file = file(file_name, "r")
	temp_list = []
	for line in input_file:
		line = line.strip()
		temp_list.append(line)
	input_file.close()
	return temp_list

def loadWordPair(file_name1, file_name2, src_dict, dest_dict):
	temp_list1 = loadFile(file_name1)
	temp_list2 = loadFile(file_name2)
	word_pair = {}

	for i in range(len(temp_list1)):
		if src_dict.has_key(temp_list1[i]) and dest_dict.has_key(temp_list2[i]):
			word_pair[temp_list1[i]] = temp_list2[i]
	
	return word_pair

def getMatrix(word_pair, src_dict, dest_dict):
	X = []
	Y = []
	for src, dest in word_pair.items():
		X.append(src_dict[src])
		Y.append(dest_dict[dest])
	return np.array(X), np.array(Y)

def showHelp():
	sys.stderr.write("\n\t-source-vector-path <file>\
							\n\t\tSet source vector file path; can't be omited.\
							\n\t-dest-vector-path <file>\
							\n\t\tSet dest vector file path; can't be omited.\
							\n\t-source-word-path <file>\
							\n\t\tSet source word file path; can't be omited.\
							\n\t-dest-word-path <file>\
							\n\t\tSet dest word file path; can't be omited.\
							\n\t-alpha <float>\
							\n\t\tSet learning rate for orthogonal transform; default is 0.25.\
							\n\t-threshold <float>\
							\n\t\tSet quit condition; default is 1e-4.\
							\n\t-output <file>\
							\n\t\tSet output file path;  can't be omited.\n\n")

def trainModel(params):
	sys.stderr.write("Loading dict ... \n")
	sys.stderr.flush()
	
	if params.has_key("src_vector_path"):
		src_dict = loadDict(params["src_vector_path"])
	else:
		exit("Need source vector path")
	
	if params.has_key("dest_vector_path"):
		dest_dict = loadDict(params["dest_vector_path"])
	else:
		exit("Need dest vector path")
	if params.has_key("output_path"):
		output_file = file(params["output_path"], "w")
	else:
		exit("Need output matrix path.")
	
	sys.stderr.write("Loading file ... \n")
	if params.has_key("src_word_path") and params.has_key("dest_word_path"):
		word_pair = loadWordPair(params["src_word_path"], params["dest_word_path"], src_dict, dest_dict)
	else:
		exit("You should insert source word and dest word in.")
	
	if params.has_key("alpha"):
		alpha = params["alpha"]
	else:
		alpha = 0.25
	
	if params.has_key("threshold"):
		threshold = params["threshold"]
	else:
		threshold = 1e-4

	X, Y = getMatrix(word_pair, src_dict, dest_dict)
	dim = X.shape[1]
	sys.stderr.flush()
	sys.stderr.write("Init Matrix ... \n")
	A = st.InitMatrix(dim)
	sys.stderr.write("Begin to training : \n")
	sys.stderr.flush()

	result = st.train(X, A, Y.T, alpha = alpha, error_rate = threshold)
	
	sys.stderr.write("Training done.\n")

	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			output_file.write(str(result[i][j]) + " ")
		output_file.write("\n")
	output_file.close()


if __name__ == "__main__":
	if len(sys.argv) == 1:
		showHelp()
	else:
		params = {}

		for i in range(len(sys.argv)):
			if sys.argv[i] == "-source-vector-path":
				params["src_vector_path"] = sys.argv[i + 1]
			elif sys.argv[i] == "-dest-vector-path":
				params["dest_vector_path"] = sys.argv[i + 1]
			elif sys.argv[i] == "-source-word-path":
				params["src_word_path"] = sys.argv[i + 1]
			elif sys.argv[i] == "-dest-word-path":
				params["dest_word_path"] = sys.argv[i + 1]
			elif sys.argv[i] == "-alpha":
				params["alpha"] = float(sys.argv[i + 1])
			elif sys.argv[i] == "-threshold":
				params["threshold"] = float(sys.argv[i + 1])
			elif sys.argv[i] == "-output":
				params["output_path"] = sys.argv[i + 1]

		trainModel(params)

