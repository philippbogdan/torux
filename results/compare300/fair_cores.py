"""Fair cores: did the solver's threads have their CPUs while it was timed?

The comparison's timing rule: a timed run counts only if its threads waited for a CPU at most 5% of the time they
ran, and other work used the process's CPUs for at most 5% of that time. On shared cores a run can lose most of its
4 cores to other work with no sign in its own output (one identical 4-thread VMEC++ warm-up solve took 17 s once
and 877 s another time). This meter reads the Linux scheduler's own accounting, /proc/<pid>/task/<tid>/schedstat,
for every thread of this process and of its child processes: how long each thread ran, and how long it was ready to
run but waited for a CPU. Every solver's harness uses this same file (its sha256 goes into each answer), so every
run is judged by the same measurement.

    import fair_cores
    meter = fair_cores.Meter().start()
    ...                                # the untimed warm-up, or the timed solve
    cores = meter.stop()               # a dict for the answer (host.cores_warmup, host.cores)
    fair_cores.fair(cores)             # True when the threads waited for a CPU at most LIMIT of the time they ran,
                                       # and other work used the process's cpus at most LIMIT of that time

    python fair_cores.py               # self-test: 4 busy processes on 4 cpus, alone and with 8 more beside them

wait_fraction is the threads' total wait for a CPU divided by their total running time. The meter reads every
`interval` seconds, so a thread that ends before stop() keeps its last reading, and a thread id the kernel reuses is
counted as a new thread. The sampler's own thread is left out.

Second measure: another process can share a physical core with the solver's thread on its sibling hardware thread,
slowing it with no wait to see (QFM ran 0.269 s per iteration beside other work on the sibling threads, against
0.19 s without it, at a wait fraction of 0.030). So the meter also reads /proc/stat for every cpu in the process's
affinity mask: the time those cpus were busy beyond the solver's own threads is foreign work on its cores. The
solver's own time there is taken exactly: getrusage for this process (every thread, ended ones too) and for its
reaped children, plus the schedstat time of descendants still alive. foreign_fraction is the rest divided by the
threads' running time; fair() needs both fractions at most LIMIT (the foreign time may also be up to FLOOR_S in
all, the counters' 10 ms granularity on a short run).

Calibration (471 runs of one fixed 4-thread VMEC++ solve on shared cores): the wait fraction explains the slow
runs (correlation 0.78 with log time). Runs at or under LIMIT took a median 1.17 times the fastest run (at most
1.65); runs over it, up to 100 times. Passive OpenMP waiting (OMP_WAIT_POLICY=PASSIVE) doubled the median time on
shared cores and multiplied the wait tenfold, so each solver keeps its own default waiting unless passive waiting
measures better for it; the answer records the waiting variables in force.

Off Linux there is no schedstat: Meter.start() raises RuntimeError, and the harnesses record that nothing was
metered.
"""
import hashlib
import os
import resource
import threading
import time

LIMIT = 0.05
FLOOR_S = 0.2
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def _threads(pid):
    try:
        return os.listdir(f"/proc/{pid}/task")
    except OSError:
        return []


def _read(root, skip):
    """{(pid, tid): (run_ns, wait_ns)} for every thread of root and of its descendants, less the thread `skip`."""
    out, todo, seen = {}, [root], set()
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for t in _threads(pid):
            tid = int(t)
            if tid == skip:
                continue
            try:
                with open(f"/proc/{pid}/task/{t}/schedstat") as f:
                    run, wait = f.read().split()[:2]
                out[(pid, tid)] = (int(run), int(wait))
            except (OSError, ValueError):
                continue
            try:
                with open(f"/proc/{pid}/task/{t}/children") as f:
                    todo += [int(c) for c in f.read().split()]
            except OSError:
                pass
    return out


def _rusage():
    """CPU seconds of this process (all its threads, ended ones too) and of its reaped children."""
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def _cpu_busy(cpus):
    """{cpu: busy ticks} for the given cpus from /proc/stat: user, nice, system, irq, softirq and steal."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3:4].isdigit():
                parts = line.split()
                c = int(parts[0][3:])
                if c in cpus:
                    v = [int(x) for x in parts[1:9]]
                    out[c] = v[0] + v[1] + v[2] + v[5] + v[6] + v[7]
    return out


class Meter:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._skip = None
        self.samples = 0

    def _update(self, now):
        with self._lock:
            for k, v in now.items():
                if k not in self._cur:
                    self._base[k], self._cur[k], self._acc[k] = (0, 0), v, (0, 0)
                elif v[0] < self._cur[k][0]:  # the kernel reused this thread id: bank the old thread, start anew
                    b, c, a = self._base[k], self._cur[k], self._acc[k]
                    self._acc[k] = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
                    self._base[k], self._cur[k] = (0, 0), v
                else:
                    self._cur[k] = v
            self.samples += 1

    def _run(self):
        self._skip = threading.get_native_id()
        self._ready.set()
        while not self._stop.wait(self.interval):
            self._update(_read(self.pid, self._skip))

    def start(self):
        if not os.path.exists(f"/proc/{self.pid}/task/{self.pid}/schedstat"):
            raise RuntimeError("no /proc schedstat on this machine: fair_cores cannot judge the run")
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fair_cores", daemon=True)
        self._thread.start()
        self._ready.wait()
        first = _read(self.pid, self._skip)
        self._base, self._cur, self._acc = dict(first), dict(first), {k: (0, 0) for k in first}
        self.load_start = os.getloadavg()
        self.cpus = set(os.sched_getaffinity(0))
        self.busy0 = _cpu_busy(self.cpus)
        self.ru0 = _rusage()
        self.t0 = time.perf_counter()
        return self

    def stop(self):
        wall = time.perf_counter() - self.t0
        busy1 = _cpu_busy(self.cpus)
        ru1 = _rusage()
        alive = _read(self.pid, self._skip)
        self._update(alive)
        self._stop.set()
        self._thread.join()
        with self._lock:
            run = sum(self._acc[k][0] + self._cur[k][0] - self._base[k][0] for k in self._cur) / 1e9
            wait = sum(self._acc[k][1] + self._cur[k][1] - self._base[k][1] for k in self._cur) / 1e9
            threads = len(self._cur)
            procs = len({k[0] for k in self._cur})
            alive_desc = sum(self._acc[k][0] + self._cur[k][0] - self._base[k][0] for k in alive if k[0] != self.pid) / 1e9
        own = ru1 - self.ru0 + alive_desc
        cpus_busy = sum(busy1.get(c, 0) - self.busy0.get(c, 0) for c in self.cpus) / CLK_TCK
        foreign = max(0.0, cpus_busy - own)
        return dict(meter=f"fair_cores {SHA256[:12]}", meter_sha256=SHA256, limit=LIMIT, floor_s=FLOOR_S,
                    wait_fraction=wait / run if run > 0 else 0.0, run_s=run, wait_s=wait, wall_s=wall,
                    cpus_busy_s=cpus_busy, own_cpu_s=own, foreign_s=foreign, foreign_fraction=foreign / own if own > 0 else 0.0,
                    cpus=sorted(self.cpus),
                    busy_cores=run / wall if wall > 0 else 0.0, threads=threads, processes=procs,
                    samples=self.samples, interval_s=self.interval, cores_in_affinity_mask=len(os.sched_getaffinity(0)),
                    load_average_start=list(self.load_start), load_average_stop=list(os.getloadavg()),
                    omp_wait_policy=os.environ.get("OMP_WAIT_POLICY"), gomp_spincount=os.environ.get("GOMP_SPINCOUNT"))


def fair(cores, limit=LIMIT):
    return cores["wait_fraction"] <= limit and (cores["foreign_fraction"] <= limit or cores["foreign_s"] <= FLOOR_S)


def _busy(seconds):
    t = time.time()
    while time.time() - t < seconds:
        pass


def _measured(q):
    """4 busy child processes under the meter (the self-test's stand-in for a solver's 4 threads)."""
    import multiprocessing as mp
    m = Meter(0.5).start()
    kids = [mp.Process(target=_busy, args=(4,)) for _ in range(4)]
    [p.start() for p in kids]
    [p.join() for p in kids]
    q.put(m.stop())


if __name__ == "__main__":
    import multiprocessing as mp
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])   # everything below shares the same 4 cpus
    for extra in (0, 8):
        others = [mp.Process(target=_busy, args=(7,)) for _ in range(extra)]   # siblings, so not measured
        [p.start() for p in others]
        q = mp.Queue()
        p = mp.Process(target=_measured, args=(q,))
        p.start()
        c = q.get()
        p.join()
        [o.join() for o in others]
        print(f"4 busy processes on 4 cpus, {extra} more beside them: wait_fraction {c['wait_fraction']:.3f}, "
              f"foreign_fraction {c['foreign_fraction']:.3f}, busy_cores {c['busy_cores']:.2f}, "
              f"threads {c['threads']}, processes {c['processes']}, fair {fair(c)}")
