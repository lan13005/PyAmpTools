#!/usr/bin/env python

import subprocess

cmd='scontrol show node sciml1902,sciml1903,sciml2101,sciml2102,sciml2103,sciml2301,sciml2302'
out=subprocess.getoutput(cmd)

nodes = out.split('\n\n')

def try_selection(line, delim1, id1, delim2, id2 , type_cast):
    try:
        output = type_cast(line.split(delim1)[id1].split(delim2)[id2])
    except:
        output = 0 # AllocTres empty means no GPUs allocated
    return output

for node in nodes:
    node = node.split('\n')
    for line in node:
        if 'NodeName' in line:
            node_name = try_selection(line, '=', 1, ' ', 0, str)
        if 'CPUAlloc' in line:
            cpu_alloc = try_selection(line, '=', 1, ' ', 0, int)
            cpu_total = try_selection(line, '=', 2, ' ', 0, int)
        if 'Gres' in line:
            gres = try_selection(line, '=', 1, ' ', 0, str)
        if "RealMemory" in line:
            free_mem = int(try_selection(line, '=', 3, ' ', 0, float)/1000)
            tot_mem  = int(try_selection(line, '=', 1, ' ', 0, float)/1000)
        if "CfgTRES" in line:
            configured_gpus = try_selection(line, ',', -1 , '=', 1, int)
        if "AllocTRES" in line:
            allocated_gpus  = try_selection(line, ',', -2 , '=', 1, int)

    output = f"{node_name}" if node_name is not None else "N/A"
    print(f"\n{'Node Name':<40}: {output}")

    output = f"{cpu_alloc}/{cpu_total} OR {(cpu_total-cpu_alloc)/cpu_total*100:.2f}% free" if cpu_alloc is not None and cpu_total is not None else "N/A"
    print(f"{'CPUAlloc / CPUTot':<40}: {output}")

    output = f"{gres}" if gres is not None else "N/A"
    print(f"{'Generic Resources':<40}: {output}")

    output = f"{free_mem} / {tot_mem} OR {float(free_mem)/float(tot_mem)*100:.2f}% free" if free_mem is not None and tot_mem is not None else "N/A"
    print(f"{'Free Memory / Total Real Memory (GB)':<40}: {output}")

    output = f"{configured_gpus-allocated_gpus}/{configured_gpus} OR {float(configured_gpus-allocated_gpus)/configured_gpus*100:.2f}% free\n" \
                if configured_gpus is not None and allocated_gpus is not None else "N/A"
    print(f"{'Free GPUs / Total GPUs':<40}: {output}")
