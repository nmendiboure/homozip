#! /usr/bin/env python
# Based on Rémy Greinhofer (rgreinho) tutorial on subcommands in docopt
# https://github.com/rgreinho/docopt-subcommands-example


"""
Gillespie stochastic simulation of homologous recombination, specifically of 
RAD51-mediated strand exchange and homology search after double strand break formation.
It is the same odel as genomatch (monte carlo) but using the Gillespie algorithm.

usage:
    genomatch-gillespie [-hv] <command> [<args>...]

options:
    -h, --help                  shows the help
    -v, --version               shows the version

The subcommands are:
    
    run             Run the pipeline


"""

import importlib.metadata

from sherpassa import commands

from docopt import DocoptExit, docopt


__version__ = importlib.metadata.version("sherpa-ssa")


def main():
    """Main entry point for the sshicstuff CLI."""

    args = docopt(__doc__, version=__version__, options_first=True)
    # Retrieve the command to execute.
    command_name = args.pop("<command>").capitalize()

    # Retrieve the command arguments.
    command_args = args.pop("<args>")
    if command_args is None:
        command_args = {}
    # After 'popping' '<command>' and '<args>', what is left in the
    # args dictionary are the global arguments.

    # Retrieve the class from the 'commands' module.
    try:
        command_class = getattr(commands, command_name)
    except AttributeError as exc:
        print("Unknown command.")
        raise DocoptExit() from exc
    # Create an instance of the command.
    command = command_class(command_args, args)
    # Execute the command.
    command.execute()


if __name__ == "__main__":
    main()
