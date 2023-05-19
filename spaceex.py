import logging
import os
import re
import subprocess
from dataclasses import dataclass

try:
    SPACEEX_PATH = os.environ["SPACEEX_PATH"]
except:
    logging.debug(
        "Defualt SpaceEx path used. Please set the environment variable SPACEEX_PATH to the path of the SpaceEx binary"
    )
    SPACEEX_PATH = "spaceex_exe/spaceex"

try:
    FLOWSTAR_PATH = os.environ["FLOWSTAR_PATH"]
except:
    logging.debug(
        "Defualt Flowstar path used. Please set the environment variable FLOWSTAR_PATH to the path of the Flowstar binary"
    )
    FLOWSTAR_PATH = "flowstar"

TIMEOUT = 500


@dataclass
class SpaceExConfig:
    REL_ERR: str = "1.0E-12"
    ABS_ERR: str = "1.0E-15"
    OUTPUT_ERR: str = "0"
    DIRECTIONS: str = "oct"
    SET_AGG: str = "chull"
    VERBOSITY: str = "l"
    SYSTEM: str = "NA"
    SAMPLING_TIME: str = "0.1"
    INIT_SAMPLING_POINTS: str = "0"
    FLOWPIPE_TOLERANCE: str = "0.01"
    FLOWPIPE_TOLERANCE_REL: str = "0"
    ITER_MAX: str = "-1"
    OUTPUT_FORMAT: str = "GEN"


class SpaceExHandler:
    def __init__(
        self,
        ndim,
        model_file,
        time_horizon,
        initially,
        scenario,
        config: SpaceExConfig = SpaceExConfig(),
        forbidden=None,
        output_file=None,
    ):
        self.model_file = model_file + ".xml"
        self.time_horizon = time_horizon
        self.initially = initially
        self.scenario = scenario
        self.output_file = output_file
        self.forbidden = forbidden
        self.config = config
        self.output_format = config.OUTPUT_FORMAT
        if ndim == 1:
            self.output_vars = "x0, t"
        else:
            self.output_vars = ",".join(["x" + str(i) for i in range(2)])

    def run(self, file):
        args = [
            SPACEEX_PATH,
            "--model-file",
            self.model_file,
            "--rel-err",
            self.config.REL_ERR,
            "--abs-err",
            self.config.ABS_ERR,
            "--output-error",
            self.config.OUTPUT_ERR,
            "--scenario",
            self.scenario,
            "--directions",
            self.config.DIRECTIONS,
            "--set-aggregation",
            self.config.SET_AGG,
            "--verbosity",
            self.config.VERBOSITY,
            # "--sampling-time",
            # SAMPLING_TIME,  # I don't know what this does, but it's in the SpaceEx call on the VM
            # "--simu-init-sampling-points",
            # INIT_SAMPLING_POINTS,
            "--flowpipe-tolerance",
            self.config.FLOWPIPE_TOLERANCE,
            "--flowpipe-tolerance-rel",
            self.config.FLOWPIPE_TOLERANCE_REL,
            "--iter-max",
            self.config.ITER_MAX,
            "--output-format",
            self.config.OUTPUT_FORMAT,
            "--time-horizon",
            self.time_horizon,
            "--initially",
            self.initially,
            "--system",
            self.config.SYSTEM,
            "--output-variables",
            self.output_vars,
        ]
        if self.output_file:
            args.extend(
                [
                    "--output-file",
                    self.output_file + "." + self.config.OUTPUT_FORMAT.lower(),
                ]
            )
        if self.forbidden:
            args.extend(["--forbidden", self.forbidden])
        # output = subprocess.run(
        #     args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        # ).stdout.decode("utf-8")

        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=500)
        except subprocess.TimeoutExpired:
            proc.kill()
            logging.warning("SpaceEx timed out")
            return None
        return self.find_time_taken(out.decode("utf-8"), err.decode("utf-8"))

    def find_time_taken(self, output, err):
        try:
            match = re.search(
                r"Computing reachable states done after (\d+\.\d+)s", output
            )
            time_taken = float(match.group(1))
        except AttributeError:
            time_taken = None
            logging.warning("SpaceEx failed. Error: {}".format(err))

        return time_taken

    def plot(self):
        if self.output_format == "GEN":
            output_file = self.output_file + ".ps"
            args = [
                "graph",
                "-T",
                "ps",
                "-C",
                "-B",
                "-q",
                "-1",
                "-f",
                "0.06",
                "-W",
                "0",
                "--pen-colors",
                "1=#8888FF:2=#BBBBFF:3=red:4=black:5=#FF00FF",
                "--fill-fraction",
                "0.3",
                "-q",
                "0.5",
                "--bitmap-size",
                "1080x1200",
                "-x",
                "-1",
                "1",
                "-y",
                "-1",
                "1",
                "<",
                self.output_file + "." + self.config.OUTPUT_FORMAT.lower(),
                ">",
                output_file,
            ]
            subprocess.run(" ".join(args), shell=True)
        else:
            raise ValueError("Output format is, not GEN")
        # graph -T png -C --fill-fraction 0.3 --pen-colors "1=blue:2=blue:3=blue:4=blue:5=blue" --bitmap-size "1080x1200"< spaceex-test.gen > plot.png
        # graph -T ps -x -1 1 -y -1 1 -C -B -q -1 -f 0.06 -W 0 --pen-colors 1=#8888FF:2=#BBBBFF:3=red:4=black:5=#FF00FF -q 0.5 --bitmap-size "1080x1200" < nl2-copy.plt > nl2-nl.ps


class FlowstarHandler:
    def run(self, model_file: str):
        command = FLOWSTAR_PATH
        with open(model_file, "r") as f:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=f
            )
        try:
            out, err = proc.communicate(timeout=500)
        except subprocess.TimeoutExpired:
            proc.kill()
            logging.warning("Flowstar timed out")
            return None
        return self.find_time_taken(out.decode("utf-8"), err.decode("utf-8"))

    def find_time_taken(self, output, err):
        try:
            match_success = re.search(r"Computation completed:", output)
            match_time = re.search(
                r"Total time cost:\x1b\[1m (\d+\.\d+)\x1b\[0m seconds.",
                output,
            )
            time_taken = float(match_time.group(1))
            if not match_success:
                time_taken = None
                logging.warning(
                    "Flowstar failed. Did not compute full flowpipe".format(err)
                )
            return time_taken
        except AttributeError:
            time_taken = None
            logging.warning("Flowstar failed. Error: {}".format(err))
        return time_taken

    def plot(self, plot_file: str):
        command = "gnuplot " + plot_file
        subprocess.run(command, shell=True)

    def plot_as_spaceex(self, plt_file: str, output_file: str):
        """Plots a flowstar plot file like a spaceex plot."""
        # graph -T ps -x -1 1 -y -1 1 -C -B -q -1 -f 0.06 -W 0 --pen-colors 1=#8888FF:2=#BBBBFF:3=red:4=black:5=#FF00FF -q 0.5 --bitmap-size "1080x1200" < nl2-copy.plt > nl2-nl.ps
        # remove first 10 lines
        subprocess.run("sed -i '1,10d' {}".format(plt_file), shell=True)

        output_file = output_file + ".ps"
        args = [
            "graph",
            "-T",
            "ps",
            "-C",
            "-B",
            "-q",
            "-1",
            "-f",
            "0.06",
            "-W",
            "0",
            "--pen-colors",
            "1=#8888FF:2=#BBBBFF:3=red:4=black:5=#FF00FF",
            "--fill-fraction",
            "0.3",
            "-q",
            "0.5",
            "--bitmap-size",
            "1080x1200",
            "-x",
            "-1",
            "1",
            "-y",
            "-1",
            "1",
            "<",
            plt_file,
            ">",
            output_file,
        ]
        subprocess.run(" ".join(args), shell=True)


if __name__ == "__main__":
    from cli import get_config

    config = get_config()
    model_file = config.output_file
    time_horizon = "6"
    initially = config.initial
    # forbidden = "x0 >= 0.3 & x0 <=0.35 & x1 >=0.5 & x1 <=0.6 "
    output_file = config.output_file
    scenario = "stc" if config.template == "pwa" else "phaver"
    s = SpaceExHandler(
        2, model_file, time_horizon, initially, scenario, output_file=output_file
    )
    s.run()
    s.plot()
    f = FlowstarHandler()
    model = "jet.model"
    f.run(model)
    f.plot("outputs/jet.plt")
