import click
import shutil
import os
import re


@click.group()
def cli():
    pass


@cli.command()
@click.argument('to_start', type=int)
def increment(to_start):
    return _increment(to_start)


def matcher(x):
    return re.findall(r'\b[\w-]+\.mdx?\b', x)


def extract_id(file, to_start, increase=True):
    id = int(re.match(r'^([0-9]+)_.*', file).groups()[0])
    if id >= to_start:
        if increase:
            modified_file = re.sub(r'^([0-9]+)_', f'{id + 1:02d}_', file)
        else:
            modified_file = re.sub(r'^([0-9]+)_', f'{id - 1:02d}_', file)
    else:
        modified_file = file
    return modified_file, id


def _increment(to_start: int):

    # create plan
    files = sorted(os.listdir('.'))
    files = [
        (file, *extract_id(file, to_start)) 
        for file in files if re.match(r'^([0-9]+)_.*', file)
    ]

    print('Would edit as:')
    print('-' * 20)
    print('\n'.join([
        f'{source} => {dst}' 
        if source != dst
        else f'{source} [no change]'
        for source, dst, _ in files
    ]))
    print('-' * 20)

    if not click.confirm('Please confirm if you\'d like to accept this edit', default=False):
        print('Aborting')
        return


    for file in files:
        with open(file[0]) as f:
            lines = f.read().split('\n')

        shutil.copy(file[0], f'{file[0]}.bak')

        if file[0] != file[1]:
            shutil.move(file[0], file[1])

        # find line matching "sidebar_position: <integer>"
        line_number, line = next((
            x 
            for x in enumerate(lines) if x[1].startswith('sidebar_position: ')
        ))
        
        # replace single line in file
        if file[2] >= to_start:
            line = re.sub(r'^sidebar_position: ([0-9]+)$', f'sidebar_position: {file[2] + 1}', line)
        lines = lines[:line_number] + [line] + lines[line_number + 1:]

        content = '\n'.join(lines)
            
        mentioned_links = set(sum([matcher(line) for line in lines], []))

        for link in list(mentioned_links):
            new_link, _ = extract_id(link, to_start)
            content = content.replace(link, new_link)

        with open(file[1], 'w') as f:
            f.write(content)


@cli.command()
@click.argument('id', type=int)
def remove(id):
    return _remove(id)


def _remove(id):
    # TODO fix this logic...

    files = sorted(
        [(f, *extract_id(f, id, increase=False)) 
        for f in os.listdir('.') if re.match(r'^([0-9]+){2,2}_.*', f)],
        key=lambda x: x[0],
    )

    to_remove = next(f for f in files if f[2] == id)
    os.remove(to_remove[0])
    files = [f for f in files if f[2] != id]

    for file in files:

        with open(file[0]) as f:
            lines = f.read().split('\n')

        shutil.copy(file[0], f'{file[0]}.bak')

        if file[0] != file[1]:
            shutil.move(file[0], file[1])

        # find line matching "sidebar_position: <integer>"
        line_number, line = next((
            x 
            for x in enumerate(lines) if x[1].startswith('sidebar_position: ')
        ))
        
        # replace single line in file
        if file[2] >= id:
            line = re.sub(r'^sidebar_position: ([0-9]+)$', f'sidebar_position: {file[2] - 1}', line)
        lines = lines[:line_number] + [line] + lines[line_number + 1:]

        content = '\n'.join(lines)
            

        mentioned_links = set(sum([matcher(line) for line in lines], []))

        for link in list(mentioned_links):
            new_link, _ = extract_id(link, id, increase=False)
            content = content.replace(link, new_link)

        with open(file[1], 'w') as f:
            f.write(content)


@cli.command()
def restore():
    files = [f for f in sorted(os.listdir('.')) if f.endswith('.bak')]
    for f in files:
        _, updated_file = extract_id(f)
        shutil.move(f, f[:-4])
        os.remove(updated_file)


@cli.command()
def clean():
    files = [f for f in sorted(os.listdir('.')) if f.endswith('.bak')]
    for f in files:
        os.remove(f)


if __name__ == '__main__':
    cli()