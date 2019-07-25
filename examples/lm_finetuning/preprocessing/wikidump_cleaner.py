import os
import re
from joblib import Parallel, delayed
import multiprocessing
import json
import random
import argparse
import sys

def clean_text(text):
	res = []
	for i in range(1,len(text)-1):
		before = text[i-1]; current = text[i]; after = text[i+1]
		if (before == '\n' and after == '\n') or (current == '\n'):
			continue
		current = re.sub(r'(((MR)|(mr)|(Mr)|(Hr)|(Fr)|(Frk))\.)\n',r'\1',re.sub(r'([A-Za-z0-9]{2,}[\.!?])',r'\1\n',current.replace('\n','')))
		res.append(current)
	return re.sub(r'[ ]+',' ',re.sub(r'\n[ \t]+','\n',' '.join(res).replace('_','').replace(' .','').replace('"',''))).strip('\n')

#def clean_text(text):
#	return text.strip().rstrip('\r\n').rstrip('\n ').rstrip('\n')

def get_texts(filename,dirname):
	with open(os.path.join(dirname,filename),'r') as f:
		for l in f:
			yield json.loads(l)['text'].split('\n')

def get_files(path):
	files = []
	for dir in os.listdir(path):
		dirname = os.path.join(path,dir)
		for file in sorted(os.listdir(dirname)):
			files.append(os.path.join(dir,file))
	return files

def save_files(texts,save_name):
	with open(save_name,'w') as f:
		for text in texts:
			for t in text:
				f.write(t+'\n\n')

def get_and_clean_text(filename,dirname):
	out = []
	for text in get_texts(filename,dirname):
		out.append(clean_text(text))
	return out

def main():
	dirpath = '/scratch/s144234/training_datasets/enwiki'
	save_name = '/scratch/s144234/training_datasets/enwiki.txt'
	parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-d', '--dirpath', type=str, default=dirpath, help='Dataset base path')
	parser.add_argument('-s', '--save_name', type=str, default=save_name, help='Path to save file at and filename')
	parser.add_argument('-f', '--sample_frac', type=float, default=0.6, help='Fraction of files to use from dump')
	args = parser.parse_args()
	files = get_files(args.dirpath)
	print(f'Total number of files: {len(files)}')
	num_samples = int(len(files)*args.sample_frac)
	print(f'Selected number of files: {num_samples}')
	files = random.sample(files,num_samples)
	n_jobs = multiprocessing.cpu_count() - 1
	get_and_clean_text_l = lambda file: get_and_clean_text(file,args.dirpath)
	texts = Parallel(n_jobs=n_jobs,verbose=10)(delayed(get_and_clean_text_l)(file) for file in files)
	save_files(texts,args.save_name)

if __name__ == '__main__':
	main()
