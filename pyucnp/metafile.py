#!/usr/bin/python
import os
import datetime
import csv
#for file in os.listdir("./B_converted"):
#    if file.endswith(".txt"):
#        print(file)

keywords = {'FILE_S': 'FILE: ','DATE_S':'DATE: ', 'FS_S':'FS: ',
			'START_S': 'START: ', 'STOP_S': 'STOP: ', 'PLACE_S': 'PLACE',
			'MEAS_S' : 'MEAS: ','SETUP_S': 'SETUP: ', 'OBS_S' : 'OBS: '};





def met_create(folder = './', cvs_filename = 'example.cvs', FS = 0, NSAMPLES = 0,START = 0, STOP = 0 ,MEAS = '\"My measure.\"' , PLACE = 'LEC', SETUP = 'mysetup', OBS = ''):
	met_filename = cvs_filename.split('.')[0]
	met_filename = met_filename + '.met'

	# VAriables to write
	today = datetime.datetime.today()

	target = open(folder+met_filename, 'w')
	target.write( '/*******************************************\n')
	target.write('* File: '+ cvs_filename + ' metafile\n')
	target.write('* Author: Juan M. Bujjamer\n')
	target.write('* This metafile provides information \n')
	target.write('* and definitions for measurement processing. \n')
	target.write('*********************************************/\n\n')
	target.write('FILE: '+ cvs_filename+'\n')
	target.write('DATE: '+today.strftime('%d/%m/%Y %H:%M:%S\n'))
	target.write('FS '+str(FS)+'\n')
	target.write('NSAMPLES '+str(NSAMPLES)+'\n')
	target.write('START '+ str(START)+'\n')
	target.write('STOP '+str(STOP)+'\n')
	target.write('PLACE '+PLACE+'\n')
	target.write('MEAS '+MEAS+'\n')
	target.write('SETUP '+SETUP+'\n')
	target.write('OBS '+OBS+'\n')
	target.close()

def met_open(folder, fname):
	met_fname = fname.split('.')[0];
	met_fname = met_fname + '.met';
	met_filename = folder+met_fname;

	with open(met_filename, 'rb') as metfile:
		met_data = csv.reader(metfile, delimiter=' ', quotechar='|')
		for i in range(7):
			met_data.next() # Omito el header
		for row in met_data:
			if row[0]=='START':
				START = int(row[1])
			if row[0]=='STOP':
				STOP = int(row[1])
		if STOP == 0: STOP = -1
	return START, STOP
