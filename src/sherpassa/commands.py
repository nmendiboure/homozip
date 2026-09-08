"""Commands for the sherpassa package."""

import os
import shutil
from os.path import join

import numpy as np
import yaml
from docopt import docopt
from mpi4py import MPI

from sherpassa import methods
from sherpassa.modelmaker import export_sbml, generate_gillespie_model
from sherpassa.simulation import run


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


class AbstractCommand:
    """Base class for the commands."""

    def __init__(self, command_args, global_args):
        self.args = docopt(self.__doc__, argv=command_args)
        self.global_args = global_args

    def execute(self):
        raise NotImplementedError


class Run(AbstractCommand):

    """
    Run a Gillespie batch.

    Usage:
        run <parameters> [--cells CELLS] [--output OUTPUT]

    Arguments:
        <parameters>  Path to the parameters .yaml file

    Options:
        -c CELLS, --cells CELLS     Number of cells (replicates) [default: 1]
        -o OUTPUT, --output OUTPUT  Output directory [default: ./output]
    """

    def execute(self):

        yaml_path = self.args["<parameters>"]
        outdir    = self.args["--output"]
        n_cells   = int(self.args["--cells"])

        with open(yaml_path, "r", encoding="utf-8") as fh:
            data_yaml = yaml.safe_load(fh)

        # =============================================================
        # Rank 0 builds the model and writes shared artifacts; all
        # other ranks pick up the broadcast.
        # =============================================================
        if RANK == 0:
            my_model, index_to_species, model_uid = generate_gillespie_model(data_yaml)
            outdir = os.path.join(outdir, str(model_uid))
            os.makedirs(outdir, exist_ok=True)

            # Antimony source
            with open(join(outdir, "model.txt"), "w", encoding="utf-8") as fh:
                fh.write(my_model)

            # SBML XML (consumable by COPASI, libSBML, BioModels, ...)
            try:
                sbml_xml = export_sbml(my_model)
                with open(join(outdir, "model.xml"), "w", encoding="utf-8") as fh:
                    fh.write(sbml_xml)
            except Exception as exc:  # pragma: no cover
                print(f"[rank 0] SBML export skipped: {exc}", flush=True)

            shutil.copy2(yaml_path, os.path.join(outdir, "params.yaml"))
            species_to_index = {v: k for k, v in index_to_species.items()}
        else:
            my_model = None
            model_uid = None
            species_to_index = None
            outdir = None

        my_model         = COMM.bcast(my_model,         root=0)
        model_uid        = COMM.bcast(model_uid,        root=0)
        species_to_index = COMM.bcast(species_to_index, root=0)
        outdir           = COMM.bcast(outdir,           root=0)

        # =============================================================
        # Per-rank simulation slice
        # =============================================================
        sims_per_rank = n_cells // SIZE
        start_idx = RANK * sims_per_rank
        end_idx   = (RANK + 1) * sims_per_rank if RANK != SIZE - 1 else n_cells

        records_dir = join(outdir, "records")
        plots_dir   = join(outdir, "plots")
        if RANK == 0:
            os.makedirs(records_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)
        COMM.Barrier()

        for s in range(start_idx, end_idx):
            run(s, model_uid, species_to_index, my_model, data_yaml, records_dir)

        COMM.Barrier()

        # =============================================================
        # Rank 0 collects, aggregates and plots
        # =============================================================
        if RANK == 0:
            all_results = []
            for s in range(n_cells):
                npz = np.load(join(records_dir, f"simulation_{s}.npz"))
                all_results.append({f: npz[f] for f in npz.files})

            agg = methods.aggregate_groups(all_results)
            methods.plot_trajectories(
                agg,
                os.path.join(plots_dir, "aggregated_trajectories.png"),
            )

            for c in (0, 100, 200):
                methods.plot_dlc(
                    agg["DLC homologous"],
                    c,
                    os.path.join(plots_dir, f"aggregated_homologous_DLC_convo{c}.png"),
                )
