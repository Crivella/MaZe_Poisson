import click

from ..input import initialize_from_yaml
from .maze import maze


@maze.group()
def run():
    pass

# @run.command()
# # @click.argument('N_p', type=int)
# # @click.argument('N', type=int)
# # @click.argument('N_steps', type=int)
# # @click.argument('L', type=float)
# def main_maze(N_p, N, N_steps, L):
#     main_Maze(N_p, N, N_steps, L)

@run.command()
@click.argument(
    'filename',
    type=click.Path(exists=True),
    required=True,
    metavar='YAML_INPUT_FILE',
    )
def main_maze_md(filename):
    gs, os, ms = initialize_from_yaml(filename)
    _main_maze_md(gs, os, ms)

@run.command()
@click.argument(
    'filename',
    type=click.Path(exists=True),
    required=True,
    metavar='YAML_INPUT_FILE',
    )
def main_maze_md_q(filename):
    gs, os, ms = initialize_from_yaml(filename)
    _main_maze_md_q(gs, os, ms)

def _main_maze_md(*args):
    from .runners.main_Maze_md import main as main_Maze_md
    main_Maze_md(*args)

def _main_maze_md_q(*args):
    from .runners.main_Maze_md_q import main as main_Maze_md_q
    main_Maze_md_q(*args)
