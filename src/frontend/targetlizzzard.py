# Adapted from https://github.com/pycket/pycket
from rpython.config.config import OptionDescription, BoolOption, IntOption, ArbitraryOption, FloatOption
from rpython.config.translationoption import get_combined_translation_config
from rpython.rlib import jit, objectmodel

from debugger import debug, done_success, done_exit
from optimisers import OPTIMISERS

lizzzardoption_descr = OptionDescription(
        "lizzzard", "lizzzard options", [])

SIMPLE_INTERPRETER = False

def get_testing_config(**overrides):
    return get_combined_translation_config(
            lizzzardoption_descr,
            translating=False,
            overrides=overrides)

def make_entry_point(lizzzardconfig=None):
    def entry_point(argv):
        debug(u"Starting entry point...", 1)
        jit.set_param(None, "trace_limit", 1000000)
        jit.set_param(None, "threshold", 131)
        jit.set_param(None, "trace_eagerness", 50)
        # jit.set_param(None, "function_threshold", 20)
        # jit.set_param(None, "vec", 1)
        # jit.set_param(None, "vec_all", 1)
        # jit.set_param(None, "max_unroll_loops", 15)
        debug(u"Importing main...", 1)
        if SIMPLE_INTERPRETER:
            from simple_interpreter import main
        else:
            from interpreter import main
        exit_code = main("../code-examples/example.clizz")
        if exit_code == 0:
            done_success()
        else:
            done_exit(exit_code)
        return 0
    return entry_point

exposed_options = []

def target(driver, args):
    from rpython.jit.metainterp import optimizeopt
    _old_build_opt_chain = optimizeopt.build_opt_chain
    def build_opt_chain(enable_opts):
        opts = _old_build_opt_chain(enable_opts)
        for Class in OPTIMISERS:
            opts.append(Class())
        return opts
    optimizeopt.build_opt_chain = build_opt_chain

    from rpython.config.config import to_optparse
    config = driver.config
    parser = to_optparse(config, useoptions=[])
    parser.parse_args(args)
    driver.exe_name = "lizzzard"
    entry_point = make_entry_point(config)
    return entry_point, None

take_options = True


if __name__ == "__main__":
    import interpreter