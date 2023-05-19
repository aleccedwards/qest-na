# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import multiprocessing
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import logging
import time
import warnings
from collections import namedtuple
from queue import Empty

import numpy as np
import torch
from tqdm import tqdm

import anal
import neural_abstraction as na
import polyhedrons
import spaceex
from benchmarks import read_benchmark
from cegis.cegis import Cegis
from cli import get_config
from config import Config
from utils import Result, Stats, clean


def record_results(config, result):
    """Record results to csv file

    Args:
        config : Cegis configuration object
        result : result namedtuple
    """
    if "csv" in config.output_type:
        a = anal.Analyser(result.NA)
        csv_file = config.results_file
        if not result.res:
            try:
                timers = [*result.timers, None]
            except TypeError:
                timers = [None, None, None, None]
            a.record_failure(
                csv_file,
                config,
                result.res,
                result.delta_t,
                timers,
                result.seed,
            )
        else:
            timers = [*result.timers, result.verifier_time]
            a.record(
                csv_file,
                config,
                result.res,
                result.delta_t,
                timers,
                result.errors,
                result.transitions,
                result.transitions_pruned,
                result.seed,
            )


def run_cegis(config: Config):
    """Run CEGIS algorithm

    Args:
        config (Config): Cegis configuration object

    Returns:
        tuplle: bool, learner, error, benchmark, cegis object
    """
    benchmark = read_benchmark(config.benchmark)
    c = Cegis(benchmark, config.target_error, config.widths, config)
    T = config.timeout_duration if config.timeout else float("inf")
    res, net, e = c.synthesise_with_timeout(t=T)
    return res, net, e, benchmark, c


def run_safety_verification(config: Config, NA, benchmark):
    """Run safety verification step on neural abstraction

    Args:
        config (Config): cegis configuration object
        NA: neural abstraction
        benchmark : benchmark object

    Returns:
        time (int): time taken to run safety verification (None if failed)
    """
    verifier_time = None
    if "xml" in config.output_type or config.spaceex:
        if config.initial:
            try:
                XI = polyhedrons.vertices2polyhedron(config.initial).to_str(sep=",")
            except TypeError:
                XI = config.initial
        else:
            XI = None
        if config.forbidden:
            XU = polyhedrons.vertices2polyhedron(config.forbidden).to_str(
                ","
            )  # Currently unused
        else:
            XU = None
        NA.to_xml(
            config.output_file,
            bounded_time=config.bounded_time,
            T=config.time_horizon,
            # initial_state=XI,
        )
        if config.spaceex:
            scenario = "phaver" if config.template == "pwc" else "stc"
            sp = spaceex.SpaceExHandler(
                benchmark.dimension,
                config.output_file,
                config.time_horizon,
                XI,
                scenario,
                forbidden=XU,
                output_file=config.output_file,
            )
            verifier_time = sp.run(config.output_file + ".txt")
            if "flowpipe" in config.output_type:
                # 2 dimensions only
                sp.plot()
    if "flowstar" in config.output_type or config.flowstar:
        fc = config.flowstar_config
        NA.to_flowstar(config.output_file, config, fc)
    if config.flowstar:
        f = spaceex.FlowstarHandler()
        verifier_time = f.run(config.output_file + ".model")
        if "flowpipe" in config.output_type:
            root = ""  # "/".join(config.output_file.split("/")[:-1]) + "/"
            output_file = root + "outputs/{}.plt".format(benchmark.short_name)
            f.plot_as_spaceex(output_file, config.output_file)

    return verifier_time


def construct_abstraction(config: Config, cegis, net, e, benchmark):
    """Construct neural abstraction

    Args:
        config: cegis configuration object
        cegis: Cegis objects
        net: learner
        e: error
        benchmark: benchmark object

    Returns:
        neural abstraction, errors, transitions, transitions_pruned
    """

    if config.template == "sig":
        # e = [0, 0]
        NA = na.SigmoidNeuralAbstraction(net, e, benchmark)
        errors = NA.error_analysis()
        transitions = 0
        transitions_pruned = 0

    elif config.template == "tanh":
        NA = na.TanhNeuralAbstraction(net, e, benchmark)
        errors = NA.error_analysis()
        transitions = 0
        transitions_pruned = 0
    else:
        # template is pwc or pwa
        NA = na.NeuralAbstraction(net, e, benchmark, config.template)
        if config.error_check:
            for mode in tqdm(NA.modes.values(), disable=True):
                mode.check_disturbance(benchmark, cegis.S, ver_type=config.verifier)
        errors = NA.error_analysis(verbose=True)
        transitions = len(NA.transitions)
        if config.prune:
            NA.prune_transitions_new(
                f_true=benchmark.f, verbose=config.verbose, ver_type=config.verifier
            )
        transitions_pruned = len(NA.transitions)

    return NA, errors, transitions, transitions_pruned


def run_all(config: Config):
    """Run CEGIS algorithm and safety verification

    Args:
        config: Cegis configuration object

    Returns:
        Result namedtuple
    """
    t0 = time.perf_counter()
    res, net, e, benchmark, cegis = run_cegis(config)
    NA, errors, transitions, transitions_pruned = construct_abstraction(
        config, cegis, net, e, benchmark
    )
    timers = get_timers(cegis, NA)
    verifier_time = run_safety_verification(config, NA, benchmark)
    if verifier_time is None and (config.spaceex or config.flowstar):
        verifier_time = float("inf")
        res = False
    t1 = time.perf_counter()
    delta_t = t1 - t0
    result = Result(
        res,
        NA,
        verifier_time,
        delta_t,
        timers,
        errors,
        transitions,
        transitions_pruned,
        config.seed,
    )
    return result


def _worker_Q(cegis_config, id, queue, run, base_seed=0):
    attempt = 0
    seed = base_seed + id

    torch.manual_seed(seed)
    np.random.seed(seed)
    cegis_config.output_file = (
        cegis_config.output_file + "_" + str(seed)
    )  # This is a hack to ensure the same output file is not used by multiple processes
    cegis_config.seed = (
        seed  # This isn't used by cegis but is for data recording and reproducibility
    )
    while run.is_set():
        result = run_all(cegis_config)
        attempt += 1
        if result.res and run.is_set():
            # Add the id to the label as a sanity check (ensures returned result is from the correct process)
            run.clear()
            logging.debug("Worker", id, "succeeded")
            result = result._replace(seed=seed)
            result_dict = {}
            result_dict["id"] = id
            result_dict["success"] = result.res
            result_dict["result" + str(id)] = result
            result_dict["attempt" + str(id)] = attempt
            queue.put(result_dict)
        elif not result.res:
            result_dict = {}
            result_dict["id"] = id
            result_dict["result" + str(id)] = result
            result_dict["attempt" + str(id)] = attempt
            queue.put(result_dict)
            logging.debug("Worker", id, "failed")
        return result_dict


class CegisSupervisorQ:
    """Runs CEGIS in parallel and returns the first result found. Uses a queue to communicate with the workers."""

    def __init__(self, timeout_sec=1800, max_P=1):
        self.cegis_timeout_sec = timeout_sec
        self.max_processes = max_P

    def run(self, cegis_config):
        stop = False
        procs = []
        queue = multiprocessing.Manager().Queue()
        run = multiprocessing.Manager().Event()
        base_seed = torch.initial_seed()
        run.set()
        id = 0
        n_res = 0
        start = time.perf_counter()
        while not stop:
            while len(procs) < self.max_processes and not stop:
                p = multiprocessing.Process(
                    target=_worker_Q, args=(cegis_config, id, queue, run, base_seed)
                )
                # p.daemon = True
                p.start()
                id += 1
                procs.append(p)
            dead = [not p.is_alive() for p in procs]

            if any(dead) and not run.is_set():
                # If any processes have died and the run event is not set then we have a successful result
                [p.terminate() for p in procs]
                res = queue.get()
                return res
            elif all(dead) and run.is_set():
                # If all processes have died and the run event is set then we have failed
                try:
                    res = queue.get(block=False)
                except:
                    res = None
                return res
            elif any(dead) and sum(dead) > n_res:
                # If any processes have died and the run event is set then a process has failed, but some are still running
                try:
                    res = queue.get(block=False)
                    n_res += 1
                    logging.debug("Got result from worker", res["id"])
                except Empty:
                    logging.debug("Queue is empty")
                    pass
            elif time.perf_counter() - start > self.cegis_timeout_sec:
                # If the timeout has been reached then kill all processes and return
                [p.terminate() for p in procs]
                res = {}
                res["id"] = ""
                delta_t = time.perf_counter() - start
                result = Result(
                    False, None, None, delta_t, None, None, None, None, None
                )
                res["success"] = False
                res["result"] = result
                return res


def main(config: Config):
    result = run_all(config)

    record_results(config, result)

    if "plot" in config.output_type:
        try:
            result.NA.plot(label=True)
        except ValueError:
            warnings.warn("Attempted to plot for n-dimensional system")
    if "pkl" in config.output_type:
        result.NA.to_pkl(config.output_file)


def main_parallel(config: Config, P=1):
    supervisor = CegisSupervisorQ(max_P=P)

    ret_dict = supervisor.run(config)
    proc_id = ret_dict["id"]
    result = ret_dict["result" + str(proc_id)]
    record_results(config, result)

    if "plot" in config.output_type:
        try:
            result.NA.plot(label=False, show=False, fname=config.output_file)
        except ValueError:
            warnings.warn("Attempted to plot for n-dimensional system")
    if "error_plot" in config.output_type:
        print("Plotting error plot")
        try:
            result.NA.error_plot()
        except ValueError:
            warnings.warn("Attempted to plot for n-dimensional system")
    if "pkl" in config.output_type:
        result.NA.to_pkl(config.output_file)
    clean(config.output_file, result.seed)


def get_timers(cegis, NA, verbose=True):
    T_learner = cegis.learner[0].get_timer()
    T_certifier = cegis.verifier.get_timer()
    T_abstraction = NA.get_timer()
    if verbose:
        print("Learner Timers: {} \n".format(T_learner))
        print("Certifier Timers: {} \n".format(T_certifier))
        print("Abstraction Timers: {} \n".format(T_abstraction))
        print("The abstraction consists of {} modes".format(len(NA.locations)))
    return T_learner.sum, T_certifier.sum, T_abstraction.sum


if __name__ == "__main__":
    conf = get_config()
    N_PROCS = int(conf.n_procs)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    for i in range(int(conf.repeat)):
        j = i * N_PROCS
        torch.manual_seed(j + conf.seed)
        np.random.seed(j + conf.seed)
        main_parallel(conf, N_PROCS)
