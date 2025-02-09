from __future__ import annotations
from subprocess import Popen, DEVNULL, PIPE
from sys import executable as py_executable
from tempfile import TemporaryDirectory
from time import perf_counter
from os import path, environ

LIZZ_EXECUTABLE:str = "../frontend/lizzzard"
JIT_LOG_PATH:str = "../jit.log"


BENCHMARKS = {
               "test":
                                 [
                                   # ("sleep", ("3",)), # test timing accuracy
                                 ],
               "benchmarksgame":
                                 [
                                   # ("fasta", ("25000000",)),
                                   ("fasta", ("2500000",)),
                                 ],
             }

MAX_TIME_WAIT:float = 600 # wait 60 sec before canceling benchmark test

_expected:str = {}


def main() -> None:
    for benchmark_name, benchmarks in BENCHMARKS.items():
        if not benchmarks: continue
        print(f" Starting {benchmark_name!r} ".center(80, "="))
        for benchmark in benchmarks:
            print(f"\t{benchmark[0]}:")
            run_py_benchmark(benchmark_name, *benchmark)
            run_lizz_benchmark(benchmark_name, *benchmark)
            run_c_benchmark(benchmark_name, *benchmark)


def run_lizz_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    def _run(_:float) -> None:
        run(LIZZ_EXECUTABLE, compiled_file, *args,
            onsuccess=print_test_success(benchmark_name, test_name, "lizz"),
            onfail=print_fail(benchmark_name, test_name, "lizz", "RUN"),
            chk_expected=(benchmark_name,test_name),
            env=environ|{"PYPYLOG":f"jit-log-opt:{JIT_LOG_PATH}"})

    src_file:str = filename(benchmark_name, test_name, "lizz")
    compiled_file:str = filename(benchmark_name, test_name, "clizz")
    run(py_executable, "../bytecoder.py", src_file, onsuccess=_run,
        onfail=print_fail(benchmark_name, test_name, "lizz", "COMPILE"))

def run_py_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    run(py_executable, filename(benchmark_name, test_name, "py"), *args,
        onsuccess=print_test_success(benchmark_name, test_name, "py"),
        onfail=print_fail(benchmark_name, test_name, "py", "RUN"),
        set_expected=(benchmark_name,test_name))

def run_c_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    ...


THIS:str = path.dirname(path.abspath(__file__))
def filename(benchmark_name:str, test_name:str, extension:str) -> str:
    return path.join(THIS, benchmark_name, test_name) + "." + extension

def print_test_success(benchmark:str, test:str, runner:str) -> Function:
    def inner(time:float) -> None:
        print(f"\t\t{runner}: {time:.2f} sec")
    return inner

def print_fail(benchmark:str, test:str, runner:str, _type:str) -> Function:
    def inner(reason:str, text:str="") -> None:
        print(f"\t\t{runner}: {_type}_FAIL[{reason}]")
        if text:
            print("Do you with to examine the debug_info?")
            while True:
                inp:str = input("Y/N: ").lower()
                if inp == "y":
                    less(text)
                    break
                elif inp == "n":
                    break
    return inner


def run(*args:tuple[str], onsuccess:Function[float,None],
        onfail:Function[None], set_expected:object=None,
        chk_expected:object=None, env:dict[str:str]=environ) -> None:
    start:float = perf_counter()
    skip_error:bool = False
    try:
        proc:Popen = Popen(args, shell=False, stdin=DEVNULL, stdout=PIPE,
                           stderr=PIPE, env=env)
    except FileNotFoundError:
        onfail("FileNotFoundError")
        return None
    try:
        while perf_counter() - start < MAX_TIME_WAIT:
            if proc.poll() is not None:
                time_taken:float = perf_counter()-start
                stdout_text:str = decode_bytes(proc.stdout.read())
                stderr_text:str = decode_bytes(proc.stderr.read())
                error:bool = (proc.poll() != 0) or stderr_text
                if not error:
                    if set_expected is not None:
                        _expected[set_expected] = stdout_text
                    elif chk_expected is not None:
                        if chk_expected not in _expected:
                            raise RuntimeError("chk_expected before " \
                                               "set_expected")
                        if _expected[chk_expected] != stdout_text:
                            debug_info:str = " EXPECTED ".center(80, "=") + "\n"
                            debug_info += _expected[chk_expected]
                            debug_info += "".center(80, "=") + "\n\n"
                            debug_info += " GOT ".center(80, "=") + "\n"
                            debug_info += stdout_text + "".center(80, "=")
                            onfail(f"ExitCode[{proc.poll()}]", debug_info)
                if error:
                    debug_info:str = " STDOUT ".center(80, "=") + "\n"
                    debug_info += stdout_text + "".center(80, "=") + "\n\n"
                    debug_info += " STDERR ".center(80, "=") + "\n"
                    debug_info += stderr_text + "".center(80, "=")
                    onfail(f"ExitCode[{proc.poll()}]", debug_info)
                else:
                    try:
                        onsuccess(time_taken)
                    except KeyboardInterrupt as error:
                        skip_error:bool = True
                        raise error
                return None
        onfail("TimeoutError")
    except KeyboardInterrupt as error:
        if not skip_error:
            onfail("KeyboardInterrupt")
        raise error

def decode_bytes(data:bytes) -> str:
    return data.decode("utf-8", "replace")

def less(text:str) -> None:
    with TemporaryDirectory() as folder:
        filename:str = f"{folder}/error.log"
        with open(filename, "w") as file:
            file.write(text)
        proc:Popen = Popen(["less", "-R", filename], shell=False)
        proc.wait()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass