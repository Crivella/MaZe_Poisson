try:
    import click as original_click
    import rich_click as click
except ImportError:
    import click
    import click as original_click
    
try:
    from rich.traceback import install as install_rich_traceback
    install_rich_traceback(show_locals=False, suppress=[click, original_click])
except ImportError:
    pass