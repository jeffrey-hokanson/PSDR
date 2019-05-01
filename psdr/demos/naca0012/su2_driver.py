from __future__ import print_function
# from __future__ import subprocess


import subprocess
import re
import os
import copy
import shutil
import time
import numpy as np
import datetime


import numpy as np
from tqdm import tqdm
import argparse

from contextlib import contextmanager

# Make sure that OpenAeroStruct is in the PATH
import os, sys

sys.path.append('/root/Source/')


parameters = [{'kind' : 'HICKS_HENNE', 'param' : (1,0.1), 'value' : 0.1}, {'kind' : 'HICKS_HENNE', 'param' : (0,0.5), 'value' : 0.1}]
def_file = '/root/Source/def_NACA0012.cfg'

# for index,param in enumerate(parameters):
def writeConfigFile(parameters, def_file,path):
	dv_kind_pattern = r'^DV_KIND='
	dv_param_pattern = r'^DV_PARAM='
	dv_value_pattern  = r'^DV_VALUE='
	print(def_file)
	with open(def_file, 'r+') as f:
		lines = f.readlines()
	tempDefStr = ''
	for line in lines:
		if re.search(dv_kind_pattern, line) is not None:
			tempDefStr += 'DV_KIND= '
			i = 0
			for i in range(len(parameters)-1):
				tempDefStr += parameters[i]['kind'] + ', '
				i += 1
			tempDefStr += parameters[i]['kind']
			tempDefStr += '\n'

		elif re.search(dv_param_pattern, line) is not None:
			tempDefStr += 'DV_PARAM= '
			for i in range(len(parameters)-1):
				tempDefStr += str(parameters[i]['param'])+ '; '
				i += 1
			tempDefStr +=str(parameters[i]['param'])
			tempDefStr += '\n'
		elif re.search(dv_value_pattern,line) is not None:
			tempDefStr += 'DV_VALUE= '
			for i in range(len(parameters)-1):
				tempDefStr += str(parameters[i]['value']) + ', '
				i += 1
			tempDefStr += str(parameters[i]['value'])
			tempDefStr += '\n'
		else:
			tempDefStr += line
	print('Stuck here')
	print(os.listdir(path))
	shutil.copy(os.path.join(path, 'def_NACA0012.cfg'),os.path.join(path, 'def_NACA0012.cfg.bak'))
	cfg_file = os.path.join(path, 'def_NACA0012.cfg'  )
	print(cfg_file)
	with open(cfg_file, 'w+') as f:
		f.write(tempDefStr)


def generateParameters(parameters):
	numOfParams = len(parameters)
	paramDict = {'kind': 'HICKS_HENNE', 'param': None, 'value' : None}
	paramRow = []
	returnParamters = []
	for i in range(numOfParams):
		

		tempDict = copy.deepcopy(paramDict)
		if numOfParams == 1:
			tempTuple = (1, 0.4)
		elif i < numOfParams//2:
			tempTuple = (1 , 0.1 + (i * 0.8/(numOfParams - 1)) )
		else:
			tempTuple = (0 , 0.1 + (i * 0.8/(numOfParams - 1)) )
		tempDict['param'] =  tempTuple
		paramRow.append(tempDict)
	# for row in (parameters):
	for index,col in enumerate(parameters):
		paramRow[index]['value'] = col
	returnParamters.append(copy.deepcopy(paramRow))
	return returnParamters

# parameters  = [[0.1,0.3,0.2,0.1],[0.1,0.2,0.2,0.1],[0.1,0.3,0.2,0.1],[0.1,0.3,0.2,0.1]]
# parameters = np.asarray(parameters)
# parameters = [[0.1]]


# subprocess.run(['SU2_CFD',inv_cfg])

def su2_driver(parameters):
	cwd = os.getcwd()
	inv_cfg = '/root/Source/inv_NACA0012.cfg'
	def_cfg = '/root/Source/def_NACA0012.cfg'
	inv_def_cfg = '/root/Source/inv_NACA0012_def.cfg'

	parameters = np.asarray(parameters)
	parameters_dict_form = generateParameters(parameters)
	ld_out = np.zeros((1,2))
	mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.mkdir(mydir)
	print(cwd)
	for iteration,row in enumerate(parameters_dict_form):
		try:

			path = os.path.join(cwd, 'Output')
			os.mkdir(path)
			# os.mkdir()
			print(path)
			shutil.copy('/root/Source/mesh_out.su2', path)
			print('copied')
			shutil.copy(def_cfg,path)
			print('copied def')
			shutil.copy(inv_def_cfg,path)
			print('copied inv')
			writeConfigFile(row,def_file,path)		
			print('written conf file')
			# print(path)		 

			# subprocess.run(['SU2_CFD','inv_NACA0012.cfg'],cwd=path)
			# def_path = os.path.join(path,'def_NACA0012.cfg')
			deformation_out_status = subprocess.run(['SU2_DEF','def_NACA0012.cfg'],cwd=path)
			# print('def out')
			# print(deformation_out_status)
			# time.sleep(5)

			cfd_solver_out_status  = subprocess.run(['SU2_CFD','inv_NACA0012_def.cfg'],cwd=path)
			# print('cfd out', cfd_solver_out_status)
			# time.sleep(5)
			shutil.move(path, mydir)
			with open(os.path.join(mydir, 'Output', 'forces_breakdown.dat'), 'r+') as f:
					lines = f.readlines()
			cl_pattern = r'Total CL:'
			cd_pattern = r'Total CD:'
			cl = cd = 0
			for line in lines:
					if re.search(cl_pattern, line) is not None:
						temp = re.search(r'Total CL: *(-?\d+\.?\d+) *| *Pressure', line)
						# print(temp)
						# temp = temp[1].split(' ')
						cl = float(temp.group(1))

					if re.search(cd_pattern,line) is not None:
						temp = re.search(r'Total CD: *(-?\d+\.?\d+) *| *Pressure', line)
						# print(temp)
						# temp = temp[1].split(' ')
						cd = float(temp.group(1))
						break;
			ld_out[iteration] = [cl, cd]


		except(IOError) as e:
			print('There was a file error = ', e)
		except(Exception) as e:
			print('There was an error : ', e)
	# print(np.asarray(ld_out))
	return np.asarray(ld_out)
# print(ld_out)

def main():
	parameters = np.zeros((0,0))
	su2_driver(parameters)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Run SU2')
	parser.add_argument(dest = 'infile', type = str)
	
	args = parser.parse_args()

	infile = args.infile
	# infile = 'my.input'
	# X = np.array([[0.1, 0.2, 0.3, 0.4]])
	# np.savetxt(infile,X)
	X = np.loadtxt(infile)
	# print('')
	# time.sleep(200) 
	# print(os.listdir(os.getcwd()))
	# X = np.zeros((1,3))
	cwd = os.getcwd()
	print(cwd)
	# os.chdir('/root/System')
	Y = su2_driver(X)
	# os.chdir(cwd)
	outfile = os.path.splitext(infile)[0] + '.output'

	np.savetxt(outfile, Y)
	# main()