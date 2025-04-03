from __future__ import annotations
from os import path, environ, getcwd, chdir, remove
from subprocess import Popen, DEVNULL, PIPE
from sys import executable as py_executable
from tempfile import TemporaryDirectory
from time import perf_counter
import resource

LIZZZARD_JIT_LOG_PATH:str = "../../jit-lizzzard.log"
PYPY_JIT_LOG_PATH:str = "../../jit-pypy.log"
LIZZ_EXECUTABLE:tuple[str] = ("../../../frontend/lizzzard",)
pypy_base:str = path.join(path.dirname(__file__), "pypy")
pypy_executable:str = path.join(pypy_base, "bin", "pypy3.11")

PYPY_ADD_ENV = {
                 "LD_LIBRARY_PATH": pypy_base,
                 "PYPYLOG": "jit-log-opt:" + PYPY_JIT_LOG_PATH
               }


class Generate:
    def __init__(self, data_file:str, *args:tuple[object]) -> None:
        self.data_file, self.args = data_file, args
class StopBenchmark(Exception): ...

def set_unlimited_stack() -> None:
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,
                                               resource.RLIM_INFINITY))


TEST_REPEAT:int = 5
BENCHMARKS = {
           "test":
                   [
                     ("sleep", (2,)), # test timing accuracy
                   ],
           "benchmarksgame":
                   [
                     ("fasta", (2500000,)),
                     ("fannkuch", (10,)),
                     ("binary-trees", (15,)),
                     ("n-body", (500000,)),
                     ("mandelbrot", (1000,)),
                     ("spectral-norm", (1000,)),
                     ("reverse-complement", Generate("data.fasta", 500000)),
                   ],
           "simplebenchmarks":
                   [
                     ("fib1", (37,)),
                     ("fib2", (37,)),
                     ("while++", (100000000,)),
                     ("rec++", (1000000,)),
                     ("attr++", (100000000,)),
                     ("raytracer", ()),
                     ("prime-sieve", (8000,)),
                     ("nsieve", ("5000000",)),
                   ],
             }

"""
pycket:
    binary-trees[15] => 0.269
    n-body[500000] => 1.565
    spectral-norm[1000] => 2.452
    fannkuch[10] => 1.096
    fib1[37] => 2.044
    fib2[37] => 2.017
    nsieve[5000000] => 1.881
    prime-sieve[8000] => 2.73
    rec++[1000000] => 1.685
"""

MAX_TIME_WAIT:float = 600 # wait 60 sec before canceling benchmark test
def average(times:list[float]) -> float:
    return sorted(times)[len(times)//2]

_expected:str = {}


def main() -> None:
    # ulimit is a shell built-in... This was obviously not going to work :/
    # def log_fail_ulimit_stack(reason:str, text:str="") -> None:
    #     print(f"[ERROR]: Failed to remove stack limits {reason=}")
    # _run("ulimit", "-s", "unlimited", onsuccess=lambda f:None,
    #      onfail=log_fail_ulimit_stack)
    # Run each benchmark 3 times (make the median)
    for benchmark_name, benchmarks in BENCHMARKS.items():
        if not benchmarks: continue
        print(f" Starting {benchmark_name!r} ".center(80, "="))
        for benchmark, args in benchmarks:
            if isinstance(args, Generate):
                try:
                    generate_data_file(benchmark_name, benchmark, args)
                except StopBenchmark:
                    continue
                args:tuple[str] = (args.data_file,)
            else:
                args:tuple[str] = tuple(map(str, args))
            print(f"\t{benchmark}[{', '.join(map(str,args))}]:")
            run_py_benchmark(benchmark_name, benchmark, args)
            run_pypy_benchmark(benchmark_name, benchmark, args)
            run_lizz_benchmark(benchmark_name, benchmark, args)
            run_ocaml_benchmark(benchmark_name, benchmark, args)
            run_ocamlc_benchmark(benchmark_name, benchmark, args)
            run_c_benchmark(benchmark_name, benchmark, args)

def generate_data_file(benchmark_name:str, test_name:str, gen:Generate) -> None:
    gen_file = filename(benchmark_name, test_name, "py", name="gen")
    _run(py_executable, gen_file, *map(str,gen.args), gen.data_file,
         onfail=raise_stop_benchmark, onsuccess=lambda f:None,
         cwd=path.dirname(gen_file))

def raise_stop_benchmark(*_) -> None:
    raise StopBenchmark()


def run_lizz_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    def _run(_:float) -> None:
        if not path.exists(compiled_file): return
        run(*LIZZ_EXECUTABLE, compiled_file.split("/")[-1], *args,
            onsuccess=print_test_success(benchmark_name, test_name, "lizz"),
            onfail=print_fail(benchmark_name, test_name, "lizz", "RUN"),
            chk_expected=(benchmark_name,test_name),
            add_env={"PYPYLOG":f"jit-log-opt:{LIZZZARD_JIT_LOG_PATH}"},
            cwd=path.dirname(compiled_file))

    src_file:str = filename(benchmark_name, test_name, "lizz")
    if not path.exists(src_file): return
    compiled_file:str = filename(benchmark_name, test_name, "clizz")
    run(py_executable, "../bytecoder.py", src_file, onsuccess=_run,
        onfail=print_fail(benchmark_name, test_name, "lizz", "COMPILE"))
    remove(compiled_file)

def run_py_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    return _run_py_benchmark(py_executable, benchmark_name, test_name, args,
                             exec_name="py")

def run_pypy_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    return _run_py_benchmark(pypy_executable, benchmark_name, test_name, args,
                             exec_name="pypy", add_env=PYPY_ADD_ENV)

def _run_py_benchmark(executable:str, benchmark_name:str, test_name:str,
                      args:tuple[str], exec_name:str,
                      add_env:dict[str:str]={}) -> None:
    src_file:str = filename(benchmark_name, test_name, "py")
    if not path.exists(src_file): return
    run(executable, src_file, *args,
        onsuccess=print_test_success(benchmark_name, test_name, exec_name),
        onfail=print_fail(benchmark_name, test_name, exec_name, "RUN"),
        set_expected=(benchmark_name,test_name), add_env=add_env,
        cwd=path.dirname(src_file))

def run_c_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    ...

def run_ocamlc_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    with TemporaryDirectory() as folder:
        compiled_file:str = f"{folder}/exec"
        src_file:str = filename(benchmark_name, test_name, "ml")
        if not path.exists(src_file): return
        def _run(*_:tuple[object]) -> None:
            if not path.exists(compiled_file):
                print_fail(benchmark_name, test_name, "ocamlc", "COMPILE")()
            onsucc = print_test_success(benchmark_name, test_name, "ocamlc")
            onfail = print_fail(benchmark_name, test_name, "ocamlc", "RUN")
            run(compiled_file, *args, onsuccess=onsucc, onfail=onfail,
                chk_expected=(benchmark_name,test_name),
                cwd=path.dirname(src_file))
        run("ocamlopt", "-o", compiled_file, "-O3", src_file,
            onfail=_run, onsuccess=_run, cwd=folder)
    remove(filename(benchmark_name, test_name, "o"))
    remove(filename(benchmark_name, test_name, "cmi"))
    remove(filename(benchmark_name, test_name, "cmx"))
    try:
        remove(filename(benchmark_name, test_name, "json"))
    except FileNotFoundError: pass

def run_ocaml_benchmark(benchmark_name:str, test_name:str, args:tuple[str]):
    src_file:str = filename(benchmark_name, test_name, "ml")
    if not path.exists(src_file): return
    run("ocaml", src_file, *args,
        onsuccess=print_test_success(benchmark_name, test_name, "ocaml"),
        onfail=print_fail(benchmark_name, test_name, "ocaml", "RUN"),
        chk_expected=(benchmark_name,test_name), cwd=path.dirname(src_file))


THIS:str = path.dirname(path.abspath(__file__))
def filename(benchmark_name:str, test_name:str, ext:str, name:str="") -> str:
    return path.join(THIS, benchmark_name, test_name, name or test_name) + \
           "." + ext

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
        raise_stop_benchmark()
    return inner


def _resolve_env(base:dict[str:str], add:dict[str:str]) -> dict[str:str]:
    new:dict[str:str] = base | add
    for merge in ("PATH", "LD_LIBRARY_PATH"):
        if merge in add:
            new[merge] = (base.get(merge, "") + ":" + add[merge]).strip(":")
    return new

def run(*args:tuple[str], **kwargs:dict) -> None:
    old_success = kwargs.pop("onsuccess", lambda t: None)
    times:list[float] = []
    def success(time:float):
        times.append(time)
    try:
        for _ in range(TEST_REPEAT):
            _run(*args, **kwargs, onsuccess=success)
    except StopBenchmark:
        pass
    if len(times) == TEST_REPEAT:
        old_success(average(times))

CWD:str = getcwd()
def _run(*args:tuple[str], onsuccess:Function[float,None],
        onfail:Function[None], set_expected:object=None, cwd:str="",
        chk_expected:object=None, add_env:dict[str:str]={}) -> None:
    env:dict[str:str] = _resolve_env(environ, add_env)
    start:float = perf_counter()
    skip_error:bool = False
    try:
        proc:Popen = Popen(args, shell=False, stdin=DEVNULL, stdout=PIPE,
                           stderr=PIPE, env=environ|add_env, cwd=cwd or CWD,
                           preexec_fn=set_unlimited_stack)
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
                    if stdout_text or stderr_text:
                        debug_info:str = " STDOUT ".center(80, "=") + "\n"
                        debug_info += stdout_text + "".center(80, "=") + "\n\n"
                        debug_info += " STDERR ".center(80, "=") + "\n"
                        debug_info += stderr_text + "".center(80, "=")
                    else:
                        debug_info:str = ""
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
            stdout_text:str = decode_bytes(proc.stdout.read())
            stderr_text:str = decode_bytes(proc.stderr.read())
            debug_info:str = " STDOUT ".center(80, "=") + "\n"
            debug_info += stdout_text + "".center(80, "=") + "\n\n"
            debug_info += " STDERR ".center(80, "=") + "\n"
            debug_info += stderr_text + "".center(80, "=")
            onfail("KeyboardInterrupt", debug_info)
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