from rpython.config.config import OptionDescription, BoolOption, IntOption, ArbitraryOption, FloatOption
from rpython.config.translationoption import get_combined_translation_config
from rpython.rlib import jit, objectmodel

lizzzardoption_descr = OptionDescription(
        "lizzzard", "lizzzard options", [])

def get_testing_config(**overrides):
    return get_combined_translation_config(
            lizzzardoption_descr,
            translating=False,
            overrides=overrides)

def make_entry_point(lizzzardconfig=None):
    def entry_point(argv):
        jit.set_param(None, "trace_limit", 1000000)
        jit.set_param(None, "threshold", 131)
        jit.set_param(None, "trace_eagerness", 50)
        jit.set_param(None, "max_unroll_loops", 15)
        from interpreter import main
        main("../code-examples/example.clizz")
        return 0
    return entry_point

exposed_options = []

def target(driver, args):
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