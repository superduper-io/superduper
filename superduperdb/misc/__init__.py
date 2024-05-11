# https://stackoverflow.com/questions/39969064/how-to-print-a-message-box-in-python
# TODO: Remove the unused functions
def border_msg(msg, indent=1, width=None, title=None):
    """Print message-box with optional title.

    :param msg: Message to print
    :param indent: Indentation of the box
    :param width: Width of the box
    :param title: Title of the box
    """
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    return box
