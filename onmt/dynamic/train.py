#!/usr/bin/env python
"""Train models with dynamic data."""
import os
import signal
import torch

from itertools import cycle

# import onmt.opts as opts
import onmt.utils.distributed

from onmt.utils.misc import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.train_single import main as single_main, _load_fields, \
    _load_checkpoint, get_train_iter

from onmt.dynamic.parse import DynamicArgumentParser
from onmt.dynamic.opts import dynamic_train_opts

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def train(opt):
    DynamicArgumentParser.valid_dynamic_corpus(opt)

    DynamicArgumentParser.validate_train_opts(opt)
    DynamicArgumentParser.update_model_opts(opt)
    DynamicArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    # get_fields also move to batch producer

    """
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            train_iter = build_dataset_iter(shard_base, fields, opt)
    """
    # move to batch producer

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, q, semaphore), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producer = mp.Process(target=batch_producer,
                              args=(queues, semaphore, opt,),
                              daemon=True)
        producer.start()
        error_handler.add_child(producer.pid)

        for p in procs:
            p.join()
        producer.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0, dynamic=True)
    else:   # case only CPU
        single_main(opt, -1, dynamic=True)


def get_fields(opt, dynamic=False):
    """Get fields from opt."""
    checkpoint = _load_checkpoint(opt)
    fields = _load_fields(opt, checkpoint, dynamic=dynamic)
    return fields


def batch_producer(queues, semaphore, opt):
    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)

    fields = get_fields(opt, dynamic=True)

    train_iter = get_train_iter(opt, fields, dynamic=True)

    def next_batch():
        new_batch = next(train_iter)
        semaphore.acquire()
        return new_batch

    b = next_batch()

    for device_id, q in cycle(enumerate(queues)):
        b.dataset = None
        # Move batch to correspond device_id: this will be done in each proc
        # batch_to(b, device_id)
        # hack to dodge unpicklable `dict_keys`
        b.fields = list(b.fields)
        q.put(b)
        b = next_batch()


def run(opt, device_id, error_queue, batch_queue, semaphore):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id, batch_queue, semaphore, dynamic=True)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def _get_parser():
    parser = DynamicArgumentParser(description='dynamic_train.py')
    dynamic_train_opts(parser)
    # opts.config_opts(parser)
    # opts.model_opts(parser)
    # opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    import pdb; pdb.set_trace()
    train(opt)


if __name__ == "__main__":
    main()
