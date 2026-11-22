"""Run lab arms from a file, keeping at most four lab processes on the machine (one core each),
whoever started them. An arm is skipped when its result exists or its log is being written."""
import os, shlex, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
jobs = [ln.strip() for ln in open(sys.argv[1]) if ln.strip() and not ln.startswith("#")]
common = shlex.split(sys.argv[2])
script = None


def busy_cores():
    out = subprocess.run(["pgrep", "-f", r"lab[0-9]*\.py --name"], capture_output=True, text=True).stdout.split()
    cores = set()
    for pid in out:
        r = subprocess.run(["taskset", "-cp", pid], capture_output=True, text=True).stdout.strip()
        if ":" in r:
            aff = r.split(":")[1].strip()
            if aff.isdigit():
                cores.add(int(aff))
    return cores


mine = {}
while jobs or mine:
    for name, proc in list(mine.items()):
        if proc.poll() is not None:
            print(time.strftime("%H:%M:%S"), name, "exit", proc.returncode, flush=True)
            del mine[name]
    ncores = int(open(os.path.join(HERE, "ncores.txt")).read().strip()) if os.path.exists(os.path.join(HERE, "ncores.txt")) else 4
    free = [c for c in range(ncores) if c not in busy_cores()]
    while free and jobs:
        spec = shlex.split(jobs.pop(0)); script = spec.pop(0)
        name = spec[0]
        log = os.path.join(HERE, "runs", name + ".log")
        if os.path.exists(os.path.join(HERE, "runs", name + ".json")) or (os.path.exists(log) and time.time() - os.path.getmtime(log) < 300):
            continue
        core = free.pop(0)
        mine[name] = subprocess.Popen(["taskset", "-c", str(core), sys.executable, os.path.join(HERE, script), "--name", name]
                                      + common + spec[1:], stdout=open(log, "w"), stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), name, "started on core", core, flush=True)
    time.sleep(5)
print("queue done")
