try:
    import rich_click as click
    import click as original_click
except ImportError:
    import click
    import click as original_click
    
try:
    from rich.traceback import install as install_rich_traceback
    install_rich_traceback(show_locals=True, suppress=[click, original_click])
except ImportError:
    pass