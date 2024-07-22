import glob
import os
import subprocess

import rich
from rich.console import Console
from rich.syntax import Syntax

console = Console()
print = console.print

commit1 = "master"
git_file_diff = "cd AmpTools; git diff --name-only {} {}"
curr_amptools_commit = "grep '^AMPTOOLS_COMMIT' Makefile"
curr_amptools_commit = subprocess.check_output(curr_amptools_commit, shell=True).decode("utf-8").strip()
curr_amptools_commit = curr_amptools_commit.split("=")[1].split("#")[0].strip()

print(f'\nCurrent Amptools Commit: "{curr_amptools_commit}"')
print(f'Comparing to "{commit1}"')

# List files that have been updated
cmd = git_file_diff.format(commit1, curr_amptools_commit)
diff_list = subprocess.check_output(cmd, shell=True).decode("utf-8").split("\n")
diff_list = [x for x in diff_list if x]
rel_diff_map = {os.path.basename(x): x for x in diff_list}
abs_diff_list = [f"AmpTools/{x}" for x in diff_list]
diff_list = [os.path.basename(x) for x in diff_list]
diff_map = {x: y for x, y in zip(diff_list, abs_diff_list)}

# List files in our PyAmpTools distribution folder
dist_files = glob.glob("Distribution/*")
abs_dist_files = dist_files
dist_files = [os.path.basename(x) for x in dist_files]
dist_map = {x: y for x, y in zip(dist_files, abs_dist_files)}

print()
needs_update = []
if len(diff_list) == 0:
    print("No files have changed.")
    exit(0)
else:
    print("These files have changed || [green]Require Update[/green] || [red]No update needed[/red]\n------------------------")
    for file in diff_list:
        if file in dist_files:
            print(f"[green]{file}[/green]")
            needs_update.append(file)
        else: 
            print(f"[red]{file}[/red]")

# Show me the diff for a file between the commits
for file in needs_update:
    print()
    print(f"Diff for {file}:\n----------------")
    cmd = f"cd AmpTools; git diff {curr_amptools_commit} {commit1} -- {rel_diff_map[file]}"
    diff = subprocess.check_output(cmd, shell=True).decode("utf-8")
    diff = Syntax(diff, "diff", theme="monokai", line_numbers=True)
    print(diff)


print()
print("Please update using the following commands:\n-------------------------------------------")
for file in needs_update:
    cmd = f"[green]code --diff {diff_map[file]} {dist_map[file]}[/green]"
    print(cmd)