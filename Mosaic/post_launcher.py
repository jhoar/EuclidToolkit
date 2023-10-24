#!/data/euclid-c11g/focus/environments/conda_focus/bin/python
import argparse
import logging
import os
import shutil
import subprocess

# Default file names
default_post_config  = 'post_config.json'

# Parse command line arguments
parser = argparse.ArgumentParser(description=
    'Post processing')
parser.add_argument('source', help='the additional Python source code folder')
parser.add_argument('input', help='the input folder')
parser.add_argument('output', help='the output folder. ' 
                    'It and its parents will be created if needed')
parser.add_argument('aux', help='the auxiliary and configuration files folder')
parser.add_argument('-s', '--step', help='Post-processing step',
                    action='append', required=True)
parser.add_argument('-p', '--prefix', help='File prefix to be processed (without folder path)',
                    action='append')
parser.add_argument('-P', '--Prefix', help='File containing list of prefixes to be processed')
parser.add_argument('-c', '--config', help='configuration file superseeding the default',
                    default=default_post_config)
parser.add_argument('-d', '--delete', help='delete the output folder before execution',
                     action='store_true', default=False)
parser.add_argument('-m', '--memory', help='Do not write output files to disk. Do all calculations in memory',
                     action='store_true', default=False)
args = parser.parse_args()

# Verify all processing steps are valid. Exit if not
valid_file_processing_steps = {
    'THUMB', 'META'
}
if not valid_file_processing_steps.issuperset(args.step):
    exit_string = f'Invalid processing steps: {set(args.step).difference(valid_file_processing_steps)}'
    logging.critical(exit_string)

# Convert input lists to strings
steps_string = '['
for step in args.step:
    steps_string += f'"{step}",'
steps_string += ']'

prefixes_string = '['

if args.Prefix:
    with open(str(args.Prefix), 'r') as input_file:
        for line in input_file.readlines():
            if line[0] != '#':
                prefixes_string += f'"{line.strip()}",'        
elif args.prefix:
    for prefix in args.prefix:
        prefixes_string += f'"{prefix}",'
else:
    logging.critical('No file prefixes specified')

prefixes_string += ']'

# Remove and/or create output folder, if needed
if args.delete and os.path.exists(args.output):
    shutil.rmtree(args.output)
os.makedirs(args.output, exist_ok=True)

# Initialise logger
post_processing_log = 'post_file_processing.log'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename=os.path.join(args.output, post_processing_log), level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')

# Log command line arguments
for arg, value in vars(args).items():
    logging.info(f'Command line argument {arg}: {value}')

# Subprocess python method call
method_call = ['python', '-c', f'import sys; sys.path.append("{args.source}"); import post; '\
  f'post.post_process("{args.input}", "{args.output}", {prefixes_string}, {steps_string}, '\
  f'"{args.output}/{post_processing_log}", "{args.aux}/{args.config}", {not args.memory})']
logging.info(f'Method call: {method_call}')

# method_call = ['python', '-c', f'import sys; sys.path.append("{args.source}"); import vis; ' \
#   f'vis.process_vis_files("{args.input}", "{args.output}", {prefixes_string}, {steps_string}, '\
#   f'"{args.output}/{post_processing_log}", "{args.aux}/{args.config}", "{args.aux}/{args.indices}", '\
#   f'"{args.aux}/{args.dark}", "{args.aux}/{args.flat}", {not args.memory})']

# Subprocess logging function
def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        if line == '':
            return
        else:
            logging.info(f'Subprocess:{line.strip()}')

# Run subprocess
logging.info('Use machine native Python from environment')
logging.info('Execute pipeline')
process = subprocess.Popen(method_call, stdout=subprocess.PIPE, universal_newlines=True)
with process.stdout:
    log_subprocess_output(process.stdout)
exitcode = process.wait() # 0 means success
logging.info(f'Subprocess exit code: {exitcode}')

logging.info('Post processing finished')
