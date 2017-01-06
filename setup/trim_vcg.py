import os, sys

for index in range(1818, 2426):
	f = 'version{:04d}.txt'.format(index)
	print f
	readFile = open(f)

	lines = readFile.readlines()

	readFile.close()
	w = open(f,'w')

	w.writelines([item for item in lines[:-10]])

	w.close()
