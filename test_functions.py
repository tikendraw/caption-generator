import os
import mimetypes
from pathlib import Path
import shutil
import os
import pandas as pd
import mimetypes
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np


def count_files_by_type(directory, show_hidden_dir=False, exclude=None, include=None):
    """
    Walks through a directory and its subdirectories recursively and counts the number of files of each MIME type.

    Args:
        directory (str): The path to the directory.
        show_hidden_dir (bool): Whether to show hidden directories or not.
        exclude (list): A list of folder names to exclude.
        include (list): A list of folder names to include.

    Returns:
        pd.DataFrame: A dataframe containing the file counts.
    """
    if exclude is None:
        exclude = []
    if include is None:
        include = []
    final = []

    for root, dirs, files in os.walk(directory):
        dir_name = os.path.basename(root)

        # Exclude hidden directories
        if not show_hidden_dir:
            dirs[:] = [d for d in dirs if not d.startswith('.')]

        # Exclude directories in the exclude list
        if dir_name in exclude:
            dirs.clear()
            continue

        # Include only directories in the include list
        if include and dir_name not in include:
            # dirs.clear()
            continue

        file_counts = {
            'directory': root,
            'base': dir_name,
            'folders': len(dirs),
            'video': 0,
            'music': 0,
            'photos': 0,
            'application/zip': 0,
            'documents': 0,
            'others':0,
            'total files':len(files)
        }

        for file in files:
            file_path = os.path.join(root, file)
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                file_counts['others'] += 1

            elif mime_type.startswith('video/'):
                file_counts['video'] += 1
            elif mime_type.startswith('audio/'):
                file_counts['music'] += 1
            elif mime_type.startswith('image/'):
                file_counts['photos'] += 1
            elif mime_type == 'application/zip':
                file_counts['application/zip'] += 1
            elif mime_type.startswith('text/') or mime_type.endswith('msword') or mime_type.endswith('pdf'):
                file_counts['documents'] += 1
        final.append(file_counts)

    return pd.DataFrame(final)


def printt(*args, sep=' ', end='\n', file=sys.stdout, flush=False):
    terminal_width = shutil.get_terminal_size().columns

    # Calculate the available width for each argument
    num_args = len(args)
    arg_width = terminal_width // num_args

    # Determine if there are any plots in the arguments
    has_plots = any(isinstance(arg, plt.Figure) for arg in args)

    # Determine the number of lines needed for each argument
    max_lines = max(
        max(str(arg).count('\n') + 1 for arg in args),
        1 if has_plots else 0
    )

    # Loop through each line
    for line_num in range(max_lines):
        # Loop through each argument
        for i, arg in enumerate(args):
            # Convert argument to string
            arg_str = str(arg)

            # Check if the argument is a plot
            if isinstance(arg, plt.Figure):
                plt.figure(arg.number)
                plt.imshow(arg)
                plt.axis('off')
                plt.show()
                continue

            # Split the argument string into lines
            lines = arg_str.splitlines()

            # Get the current line or an empty string if there are no more lines
            line = lines[line_num] if line_num < len(lines) else ""

            # Calculate the width for this argument
            arg_width_adjusted = arg_width - len(sep)
            if i == num_args - 1:
                arg_width_adjusted -= len(end)

            # Truncate or pad the line to fit the width
            line_str = line[:arg_width_adjusted].ljust(arg_width_adjusted)

            # Print the line
            file.write(line_str)

            # Print the separator between arguments
            if i < num_args - 1:
                file.write(sep)

        # Print the end character/string
        file.write(end)

    # Flush the output if specified
    if flush:
        file.flush()


if __name__ == '__main__':
    print('main')