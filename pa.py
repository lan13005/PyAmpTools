#!/usr/bin/env python

import argparse
import argcomplete
import subprocess
import os
import sys

##################################################################
# pa = pyamptools...
# This script attempts to be a dispatch system that organizes the
# various scripts in the repository to be accessible under a single
# executable
##################################################################

class HelpOnErrorParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()  # Print the full help message
        message = message.split(',')[0] # command_args is part of the required args by default, we do not care about it
        sys.stderr.write(f'\nERROR: {message}\n\n')
        sys.exit(2)  # Exit with a non-zero status code to indicate an error

def main():

    ''' Dispatch function that calls the appropriate script and passes remaining arguments to it '''

    REPO_HOME = os.environ['REPO_HOME']

    print(f'REPO_HOME: {REPO_HOME}')

    func_map = {
        'fit'        : f'{REPO_HOME}/src/pyamptools/mle.py',
        'fitfrac'    : f'{REPO_HOME}/src/pyamptools/extract_ff.py',
        'mcmc'       : f'{REPO_HOME}/src/pyamptools/mcmc.py',
        'gen_amp'    : f'{REPO_HOME}/scripts/gen_amp.py',
        'gen_vec_ps' : f'{REPO_HOME}/scripts/gen_vec_ps.py',
    }

    choices = func_map.keys()

    parser = HelpOnErrorParser(description="Dispatch pyamptools commands")
    parser.add_argument('command', choices=choices, help=f'Command to run: [ {" | ".join(func_map.keys())} ]')
    parser.add_argument('command_args', nargs=argparse.REMAINDER, help='Remaining arguments that will be passed to command')

    # tab-autocomplete so you can type 'pa ' into terminal and tab for available functions
    #   help is implicit
    argcomplete.autocomplete(parser, exclude=['-h', '--help'])

    _args = parser.parse_args()

    cmd = _args.command
    cmd_args = _args.command_args
    cmd_path = func_map[cmd]

    assert( cmd in func_map.keys() ), f'Command {cmd} not recognized. Must be one of: {func_map.keys()}'

    # Call the script with additional arguments
    command = ['python'] + [cmd_path] + cmd_args
    print(f'\n======================================================================================================')
    print(f'Running command: {" ".join(command)}')
    print(f'======================================================================================================\n')
    subprocess.run( command )

if __name__ == "__main__":
    main()
