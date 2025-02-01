from rpython.jit.metainterp.optimizeopt.optimizer import Optimization


class OptimiseAccesses(Optimization):
    def emit(self, op, *args):
        return Optimization.emit(self, op, *args)

    def opt_instr(self, op, *args):
        # print(op)
        return self.emit(op, *args)


OptimiseAccesses.propagate_forward = OptimiseAccesses.opt_instr
OPTIMISERS = [OptimiseAccesses]