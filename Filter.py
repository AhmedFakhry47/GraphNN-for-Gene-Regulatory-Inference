import numpy as np
import json
import sys
import os

def open_file(direc):
	file = open(direc,'r+')

	for row in file.readlines():
		elems  = row.split()
		yield elems

class numpy_encoder(json.JSONEncoder):
	def default(self,obj):
		if isinstance(obj,np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

	
if __name__ == '__main__':

	direc,n_files = sys.argv[1], int(sys.argv[2])
	
	output  = {'N-{}'.format(i):{} for i in range(1,n_files+1)}
	
	file1   = 'Generated Yeast-{}_goldstandard.tsv'
	file2   = 'Generated Yeast-{}_multifactorial_perturbations.tsv'
	

	for i in range(1,n_files+1):
		#To get gene expressions
		genes_file   = open_file(os.path.join(direc,file2.format(i)))

		genes  = {geneName:[] for geneName in next(genes_file)}
		expressions = []
		for row in genes_file:
			expressions.append(row)

		expressions = np.array(expressions,dtype=np.float64).T

		for j,gene in enumerate(genes.keys()):
			genes[gene] = expressions[j]
		
		del expressions

		#To store labels 
		groundTs = open_file(os.path.join(direc,file1.format(i)))
		
		for p,groundT in enumerate(groundTs):
			output['N-{}'.format(i)][p] = {}
			output['N-{}'.format(i)][p]['Gene_A'],output['N-{}'.format(i)][p]['Gene_B'] = groundT[0],groundT[1]
			output['N-{}'.format(i)][p]['Label'] = int(groundT[2])

			output['N-{}'.format(i)][p]['Expression_A'] = genes[output['N-{}'.format(i)][p]['Gene_A']]
			output['N-{}'.format(i)][p]['Expression_B'] = genes[output['N-{}'.format(i)][p]['Gene_B']]

	outfile = open('Preprocessed.json','w')
	json.dump(output,outfile,cls=numpy_encoder)








		







