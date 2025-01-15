import click

# from ..input import initialize_from_yaml
from ..myio.input import load_file
from ..solver import SolverMD
from .maze import maze


@maze.group()
def run():
    pass

@run.command()
@click.argument(
    'filename',
    type=click.Path(exists=True),
    required=True,
    metavar='YAML_INPUT_FILE',
    )
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Enable verbose logging',
    )
def md(filename, verbose):
    try:
        gs, os, ms = load_file(filename)
    except ValueError as e:
        click.echo(e)
        return
    if verbose:
        os.debug = True

    solver = SolverMD(gs, ms, os)
    solver.run()
