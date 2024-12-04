import os
import subprocess
import gc
import torch
import inspect
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout

def getCoreAffinity():
    """
    Detects the number of available CPUs.
    """
    try:
        currentjobid = os.environ["SLURM_JOB_ID"]
        currentjobid = int(currentjobid)
        command = f"squeue --Format=JobID,cpus-per-task | grep {currentjobid}"
        # Run the command as a subprocess and capture the output
        output = subprocess.check_output(command, shell=True)[5:-4].replace(b" ", b"")
        cpus = output.decode("utf-8")
        cpus2 = len(os.sched_getaffinity(0))
        cpus = min(int(cpus), cpus2)
    except:
        cpus = os.cpu_count()
    return cpus

def setOptimalWorkers():
    """
    Detects the number of available CPUs.
    """
    try:
        currentjobid = os.environ["SLURM_JOB_ID"]
        currentjobid = int(currentjobid)
        command = f"squeue --Format=JobID,cpus-per-task | grep {currentjobid}"
        # Run the command as a subprocess and capture the output
        output = subprocess.check_output(command, shell=True)[5:-4].replace(b" ", b"")
        cpus = output.decode("utf-8")
        cpus2 = len(os.sched_getaffinity(0))
        cpus = min(int(cpus), cpus2)
    except:
        cpus = 1
    return cpus

def gpuClean(local_vars=None, exclude_vars=None, verbose=False):
    """
    Automatically detects and frees GPU memory by cleaning up tensor variables,
    excluding specified variables from cleanup.

    Parameters:
    local_vars (dict, optional): Dictionary of local variables. If None, uses calling frame's locals
    exclude_vars (list, optional): List of variable names to exclude from cleanup
    verbose (bool): Whether to print information about cleaned variables

    Returns:
    tuple: (freed_count, freed_memory_mb) - Number of tensors freed and approximate memory freed in MB
    """
    console = Console()

    if local_vars is None:
        frame = inspect.currentframe().f_back
        local_vars = frame.f_locals
        caller_name = frame.f_code.co_name

    exclude_vars = set(exclude_vars or [])
    freed_count = 0
    total_memory_freed = 0

    if verbose:
        # Create table for tensor cleanup details
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Variable Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Shape", style="yellow")
        table.add_column("Memory", style="red")
        console.print(
            f"\n[bold blue]Starting cleanup in scope: [cyan]{caller_name}()[/cyan][/bold blue]"
        )

    # Process direct tensor variables
    tensor_vars = {
        name: var
        for name, var in local_vars.items()
        if isinstance(var, torch.Tensor) and var.is_cuda and name not in exclude_vars
    }

    # Process tensor variables
    for name, tensor in tensor_vars.items():
        try:
            size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
            total_memory_freed += size_mb

            if verbose:
                size_str = (
                    f"{size_mb:.2f}MB" if size_mb < 1024 else f"{(size_mb/1024):.2f}GB"
                )
                table.add_row(name, "Tensor", str(list(tensor.shape)), size_str)

            del local_vars[name]
            freed_count += 1
        except Exception as e:
            if verbose:
                console.print(f"[red]Error cleaning up tensor '{name}': {str(e)}[/red]")

    # Process container variables
    container_vars = {
        name: var
        for name, var in local_vars.items()
        if (hasattr(var, "__dict__") or isinstance(var, (list, tuple, dict)))
        and name not in exclude_vars
    }

    # Process containers (dicts, lists, etc.)
    for name, container in container_vars.items():
        if isinstance(container, dict):
            tensor_keys = [
                k
                for k, v in container.items()
                if isinstance(v, torch.Tensor)
                and v.is_cuda
                and f"{name}[{k}]" not in exclude_vars
            ]
            for k in tensor_keys:
                try:
                    tensor = container[k]
                    size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                    total_memory_freed += size_mb

                    if verbose:
                        size_str = (
                            f"{size_mb:.2f}MB"
                            if size_mb < 1024
                            else f"{(size_mb/1024):.2f}GB"
                        )
                        table.add_row(
                            f"{name}[{k}]",
                            "Dict Tensor",
                            str(list(tensor.shape)),
                            size_str,
                        )

                    del container[k]
                    freed_count += 1
                except Exception as e:
                    if verbose:
                        console.print(
                            f"[red]Error cleaning up tensor in dict '{name}[{k}]': {str(e)}[/red]"
                        )

        elif isinstance(container, (list, tuple)):
            tensor_indices = [
                i
                for i, v in enumerate(container)
                if isinstance(v, torch.Tensor)
                and v.is_cuda
                and f"{name}[{i}]" not in exclude_vars
            ]
            if isinstance(container, list):
                for i in reversed(tensor_indices):
                    try:
                        tensor = container[i]
                        size_mb = (
                            tensor.element_size() * tensor.nelement() / (1024 * 1024)
                        )
                        total_memory_freed += size_mb

                        if verbose:
                            size_str = (
                                f"{size_mb:.2f}MB"
                                if size_mb < 1024
                                else f"{(size_mb/1024):.2f}GB"
                            )
                            table.add_row(
                                f"{name}[{i}]",
                                "List Tensor",
                                str(list(tensor.shape)),
                                size_str,
                            )

                        del container[i]
                        freed_count += 1
                    except Exception as e:
                        if verbose:
                            console.print(
                                f"[red]Error cleaning up tensor in list '{name}[{i}]': {str(e)}[/red]"
                            )

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    if verbose and freed_count > 0:
        # Print table first
        console.print(table)

        # Then print summary in panel
        summary_text = []
        summary_text.append(f"Tensors freed: {freed_count}")

        if total_memory_freed >= 1024:
            summary_text.append(
                f"Total memory freed: [bold green]{total_memory_freed/1024:.2f}GB[/bold green]"
            )
        else:
            summary_text.append(
                f"Total memory freed: [bold green]{total_memory_freed:.2f}MB[/bold green]"
            )

        current_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        current_reserved = torch.cuda.memory_reserved() / 1024 / 1024

        if current_allocated >= 1024:
            summary_text.append(
                f"Current GPU allocated: [bold yellow]{current_allocated/1024:.2f}GB[/bold yellow]"
            )
        else:
            summary_text.append(
                f"Current GPU allocated: [bold yellow]{current_allocated:.2f}MB[/bold yellow]"
            )

        if current_reserved >= 1024:
            summary_text.append(
                f"Current GPU reserved: [bold red]{current_reserved/1024:.2f}GB[/bold red]"
            )
        else:
            summary_text.append(
                f"Current GPU reserved: [bold red]{current_reserved:.2f}MB[/bold red]"
            )

        if exclude_vars:
            summary_text.append(
                f"\nExcluded variables: [dim]{', '.join(exclude_vars)}[/dim]"
            )

        console.print(
            "\n".join(summary_text),
        )
    elif verbose:
        console.print("[yellow]No tensors were cleaned up[/yellow]")

    return freed_count, total_memory_freed
