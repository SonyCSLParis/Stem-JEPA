import datetime
import logging
import os
import signal

from lightning.fabric.utilities.distributed import _init_dist_connection
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only


log = logging.getLogger(__name__)


class DDPFileInitStrategy(DDPStrategy):
    def __init__(self,
                 shared_file: str,
                 timeout: datetime.timedelta = datetime.timedelta(seconds=3600),
                 *args,
                 **kwargs) -> None:
        super().__init__(timeout=timeout, *args, **kwargs)
        self._shared_file = shared_file

    def setup_distributed(self) -> None:
        log.debug(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()
        self.set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None

        os.makedirs(os.path.dirname(self._shared_file), exist_ok=True)

        _init_dist_connection(self.cluster_environment,
                              self._process_group_backend,
                              init_method=f'file://{self._shared_file}',
                              timeout=self._timeout)


class MPISlurmEnvironment(SLURMEnvironment):
    def __init__(self, auto_requeue: bool = True, requeue_signal: signal.Signals | None = None) -> None:
        super().__init__(auto_requeue, requeue_signal=requeue_signal)
    
    def world_size(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    def local_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    def node_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_NODE_RANK"])
