import json
import os
import pandas as pd
import subprocess
import tempfile
from typing import Any

try:
    import wandb
    _WANDB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _WANDB_AVAILABLE = False

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only


class EVARCallback(ModelCheckpoint):
    def __init__(self, *args, script_name: str = "./scripts/quick_eval.sh", **kwargs) -> None:
        super(EVARCallback, self).__init__(*args, **kwargs)

        # attributes
        self.script_name = script_name
        self.tempfiles = dict()
        self.eval_subprocesses = dict()
        self._last_global_step_evaluated = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint and run evaluation in a separate process at the end of the training epoch."""
        super(EVARCallback, self).on_train_epoch_end(trainer, pl_module)
        if (trainer.global_rank == 0
                and self._last_global_step_evaluated != self._last_global_step_saved
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0):
            
            
            tmp_file = tempfile.NamedTemporaryFile(delete=False).name + ".csv"
            eval_subprocess = subprocess.Popen(['/bin/bash',
                                                self.script_name,
                                                self.last_model_path,
                                                tmp_file])
            pid = eval_subprocess.pid
            self.eval_subprocesses[pid] = eval_subprocess
            self.tempfiles[pid] = tmp_file

            self._last_global_step_evaluated = self._last_global_step_saved

    @rank_zero_only
    def on_train_batch_start(
            self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super(EVARCallback, self).on_train_batch_start(trainer, pl_module, batch, batch_idx)

        finished = []
        for pid, subprocess in self.eval_subprocesses.items():
            return_code = subprocess.poll()
            if return_code is None:  # evaluation is not finished, let's try again later
                continue

            try:
                df = pd.read_csv(self.tempfiles[pid])

                # Calculate the average score
                average_score = df['score'].mean()

                # Create a new DataFrame for the average row
                average_row = pd.DataFrame({'task': ['average'], 'score': [average_score]})

                # Concatenate the original DataFrame with the new average row
                df = pd.concat([df, average_row], ignore_index=True)

                accuracies = dict(zip(df["task"], df["score"]))
                pl_module.log_dict({"accuracy/" + k: v for k, v in accuracies.items()})

                # images
                for k in accuracies.keys():
                    pass

                # retrieval
                json_path = self.tempfiles[pid] + ".json"
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        retrieval_metrics = json.load(f)

                    pl_module.log_dict(retrieval_metrics)

            except Exception as e:
                print(f"Couldn't open tmpfile `{self.tempfiles[pid]}`.")
                print(e)
            finally:
                finished.append(pid)
        
        for pid in finished:
            del self.eval_subprocesses[pid]
            del self.tempfiles[pid]

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_end(trainer, pl_module)
        if trainer.global_rank != 0:
            return
        
        tmp_file = tempfile.NamedTemporaryFile(delete=False).name + ".csv"
        eval_subprocess = subprocess.Popen(['/bin/bash',
                                            self.script_name.replace("quick", "all"),
                                            self.last_model_path,
                                            tmp_file])
        self._last_global_step_evaluated = self._last_global_step_saved
    
        eval_subprocess.wait()
        print("process finished")

        df = pd.read_csv(tmp_file).groupby("task", as_index=False)["score"].mean()

        # Calculate the average score
        average_score = df['score'].mean()

        # Create a new DataFrame for the average row
        average_row = pd.DataFrame({'task': ['average'], 'score': [average_score]})

        # Concatenate the original DataFrame with the new average row
        df = pd.concat([df, average_row], ignore_index=True)

        accuracies = dict(zip(df["task"], df["score"]))
        print("logging", accuracies)

        # retrieval
        json_path = tmp_file + ".json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                retrieval_metrics = json.load(f)

            pl_module.logger.experiment.log(retrieval_metrics)

        if _WANDB_AVAILABLE:
            table = wandb.Table(dataframe=df)
            wandb.log({
                "downstream_acc": wandb.plot.bar(table, "task", "score", title="Downstream accuracy")
            })
            print("logged")
        else:
            pl_module.logger.experiment.log({"accuracy/" + k: v for k, v in accuracies.items()})
