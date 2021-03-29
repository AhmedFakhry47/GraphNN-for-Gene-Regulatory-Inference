import numpy as np
import json
import sys
import os

def open_file(direc):
	file = open(direc,'r+')

	for row in file.readlines():
		row    = row.split('\n')[0]
		elems  = row.split(',')
		yield elems

class numpy_encoder(json.JSONEncoder):
	def default(self,obj):
		if isinstance(obj,np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':

	dirc = sys.argv[1]

	output = {}
	files  = {}

	for file in os.listdir(dirc):
		if 'expression' in file.lower():
			files['expression'] = os.path.join(dirc,file)

		elif 'regulation' in file.lower():
			files['regulation'] = os.path.join(dirc,file)


	#To get gene expressions
	genes_file  = open_file(files['expression'])

	genes  = {geneName:{} for geneName in next(genes_file)[1:]}
	expressions = []

	for row in genes_file:
		expressions.append(row[1:])

	expressions = np.array(expressions,dtype=np.float64).T

	for j,gene in enumerate(genes.keys()):
		genes[gene] = expressions[j]
	
	del expressions

	#To store labels 
	groundTs = open_file(files['regulation'])
	columns  = next(groundTs)[1:]
	rows     = []

	matrix   = []
	for p,groundT in enumerate(groundTs):
		rows.append(groundT[0])
		matrix.append(list(map(int,groundT[1:])))

	matrix = np.reshape(np.array(matrix),(len(rows),len(columns)))

	for i,row in enumerate(rows):
		keys = [row +'-'+ column for column in columns]
		
		for j,key in enumerate(keys):
			if key not in output:
				output[key] = {}
				output[key]['Labelx']= int(matrix[i,j])
				output[key]['Labely']= 1 - int(matrix[i,j])
		
	outfile = open('Yeastract_Preprocessed.json','w')
	json.dump(output,outfile,cls=numpy_encoder)




