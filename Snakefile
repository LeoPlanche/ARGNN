"""
Snakefile: the argGNN pipeline in 5 stages. `snakemake STAGE` runs that stage plus whatever it
still needs and skips what is already on disk, and with no stage named, all 5 run.

  training_data  src/simulate_data.py training_data               -> training_data/{model}/
  testing_data   src/simulate_data.py testing_data                -> testing_data/{model}/
  external_data  src/run_external_tools.py, on the testing data   -> external_data/{tool}/testing_data/{model}/
  train          src/train.py                                     -> checkpoints/{prefix}/
  benchmark      src/score_benchmark.py and src/score_arggnn.py   -> benchmarking/{name}/scores/, benchmarking/{name}/{prefix}/

Every GNN_* setting comes from the config file and reaches the scripts unchanged, as an environment
variable. The Snakefile itself sets only what joins the stages: the folders, the model and size of
a job, and the checkpoint name. The config key external_tools_testing_data, a comma list, says which
tools run (default relate,tsinfer,true_topology,singer). The tools run on the testing data only,
since run_external_tools.py reads the tool seed of their tables, so external_tools_training_data
must be empty. The settings of the tools are constants of run_external_tools.py, and a relate,
tsinfer, or singer block of the config is not read.

Config blocks and the scripts that get them:
  training_data, testing_data   simulate_data.py
  tool_paths                    run_external_tools.py: the tools, as paths relative to the repo root
  shared                        train.py, and score_arggnn.py, which reads it from the config file
  train                         train.py
  benchmark                     score_arggnn.py, which reads it from the config file

The train and score_arggnn jobs first build a folder of links to the files they need, and nothing
is copied.

Names:
  prefix  {training models}__seq{GNN_SEQ_LEN}_knn{GNN_KNN_K}[w]_rank{GNN_RANK_LEVELS}[_{label}]
  name    {testing models}__{tools run on the testing data}
The 5 folders are in the repo root, or in the folder the config names as output_folder. 2 configs
that write the same folder share its files, so they must agree on the settings behind those files,
and the small check run has its own output_folder for that reason. The first job a config runs in a
folder leaves a copy of the config file there.

A job runs when one of its outputs is missing or older than one of its inputs (the profile sets
rerun-triggers to mtime). A changed setting never reruns a finished job: give the new run another
name, or delete the folders it should replace. To list the finished jobs whose settings differ
from the config:
  snakemake -n --rerun-triggers mtime params input --configfile CONFIG

There is no default config: --configfile is required, so 2 configs never merge.
Run with (from the repo root):
  snakemake -n --configfile config/pipeline_check.yaml                       (dry run)
  sbatch hpc/scripts/snakemake_driver.sh config/pipeline_check.yaml          (all stages)
  sbatch hpc/scripts/snakemake_driver.sh config/pipeline_check.yaml train    (1 stage)
"""

import os
import shlex

if not workflow.configfiles:
    raise ValueError("no config file: add --configfile config/NAME.yaml")
CONFIGFILE = str(workflow.configfiles[0])
CONFIGNAME = os.path.basename(CONFIGFILE)
PY         = ".venv/bin/python"               # the repo venv on the Mac and on Oscar
DATASETS   = ["training_data", "testing_data"]
PER_JOB    = ("GNN_VS_SCENARIOS", "GNN_VS_SIZES")     # set by each simulation job from its wildcards
OUT        = str(config.get("output_folder") or "")   # "" = the repo root
OUT        = OUT.rstrip("/") + "/" if OUT else ""


def words(text):
    """Entries of a comma list, as strings ("" or an empty value gives no entries)."""
    return [w for w in str("" if text is None else text).split(",") if w]


def env(*blocks, skip=()):
    """KEY=value words for `env`, from every variable of the named config blocks, unchanged."""
    return " ".join(f"{k}={shlex.quote(str(v))}" for b in blocks for k, v in config.get(b, {}).items()
                    if k not in skip)


MODELS  = {d: words(config[d]["GNN_VS_SCENARIOS"]) for d in DATASETS}
SIZES   = {d: words(config[d]["GNN_VS_SIZES"]) for d in DATASETS}
N_FILES = {d: int(config[d]["GNN_VS_FILES"]) for d in DATASETS}
# The files that run_external_tools.py writes into 1 folder per testing file, its OUTPUT_FILES. SINGER keeps
# samples 500 to 599 of its chain of 600.
TOOL_OUTPUTS = {
    "relate": ["relate.trees", "relate.coal", "relate_sample_branch_lengths.newick.gz",
               "relate_sample_branch_lengths.sites", "run_times.csv"],
    "tsinfer": ["tsinfer.trees", "tsinfer_node_posteriors.npz", "run_times.csv"],
    "true_topology": ["true_topology.trees", "true_topology_node_posteriors.npz", "true_topology_polegon.npz",
                      "run_times.csv"],
    "singer": [f"singer_{k}.trees" for k in range(500, 600)]
              + ["singer.log", "singer_trace.csv", "singer_polegon.npz", "run_times.csv"],
}
TOOLS = words(config.get("external_tools_testing_data", "relate,tsinfer,true_topology,singer"))
for t in TOOLS:
    if t not in TOOL_OUTPUTS:
        raise ValueError(f"external_tools_testing_data: unknown tool '{t}' (known: {list(TOOL_OUTPUTS)})")
if words(config.get("external_tools_training_data", "")):
    raise ValueError("external_tools_training_data must be empty: the external tools run on the testing data only")
# Relate runs from a temporary folder, so the scripts need absolute tool paths.
# The config holds paths relative to the repo root.
TOOL_ENV = " ".join(f"{k}={shlex.quote(os.path.abspath(str(v)))}" for k, v in config["tool_paths"].items())

TRAIN_MODELS = words(config["train"]["GNN_VS_SCENARIOS"])
PREFIX = ("_".join(TRAIN_MODELS)
          + f"__seq{config['shared']['GNN_SEQ_LEN']}_knn{config['shared']['GNN_KNN_K']}"
          + ("w" if int(config["shared"]["GNN_KNN_K"]) > 0
             and str(config["shared"].get("GNN_KNN_WEIGHTED", "1")) == "1" else "")
          + "_rank" + (str(config["train"].get("GNN_RANK_LEVELS", "1"))
                       if float(config["train"].get("GNN_RANK_WEIGHT", "0")) > 0 else "0")
          + (f"_{config['label']}" if config.get("label") else ""))
NAME       = "_".join(MODELS["testing_data"]) + "__" + ("_".join(TOOLS) or "no_tools")
CKPT_DIR   = f"{OUT}checkpoints/{PREFIX}"
BENCH_DIR  = f"{OUT}benchmarking/{NAME}/{PREFIX}"
SCORES_DIR = f"{OUT}benchmarking/{NAME}/scores"


def true_files(dataset, model, sizes=None):
    """The true tree sequences simulate_data.py writes for 1 model."""
    return [f"{OUT}{dataset}/{model}/{model}_S{size}_{i}.trees"
            for size in (sizes or SIZES[dataset]) for i in range(N_FILES[dataset])]


def simulation_tables(dataset):
    """The tables simulate_data.py writes, one per model and size. They are targets of the
    simulation stages, so a folder of true files without its tables is simulated again."""
    return [f"{OUT}{dataset}/{model}/{model}_S{size}_simulations.csv"
            for model in MODELS[dataset] for size in SIZES[dataset]]


def tool_files(model, sizes=None):
    """The files of every tool listed for the testing data, for 1 model."""
    return [f"{OUT}external_data/{tool}/testing_data/{model}/{model}_S{size}_{i}/{name}"
            for tool in TOOLS for size in (sizes or SIZES["testing_data"])
            for i in range(N_FILES["testing_data"]) for name in TOOL_OUTPUTS[tool]]


def tool_tables(model):
    """The tables score_benchmark.py writes for every tool listed for the testing data, for one model."""
    return [f"{SCORES_DIR}/{tool}/{model}/{model}_S{size}_{i}.csv.gz"
            for tool in TOOLS for size in SIZES["testing_data"] for i in range(N_FILES["testing_data"])]


def arggnn_tables(model):
    """The tables score_arggnn.py writes from the checkpoint, one per true testing file."""
    return [f"{BENCH_DIR}/scores/{model}/{model}_S{size}_{i}.csv.gz"
            for size in SIZES["testing_data"] for i in range(N_FILES["testing_data"])]


def copy_config(folder):
    """Shell words for the end of a job: leave a copy of the config file in the folder the job
    wrote, unless a copy is already there. The copy is not an input or output of any rule, so it
    never makes a job run."""
    return f" && ( [ -e {folder}/{CONFIGNAME} ] || cp {shlex.quote(CONFIGFILE)} {folder}/{CONFIGNAME} )"


STAGES = {
    "training_data": [f for m in MODELS["training_data"] for f in true_files("training_data", m)]
                     + simulation_tables("training_data"),
    "testing_data":  [f for m in MODELS["testing_data"] for f in true_files("testing_data", m)]
                     + simulation_tables("testing_data"),
    "external_data": [f for m in MODELS["testing_data"] for f in tool_files(m)],
    "train":         [f"{CKPT_DIR}/{PREFIX}.pt"],
    "benchmark":     [f for m in MODELS["testing_data"] for f in tool_tables(m) + arggnn_tables(m)],
}
# links to the files a job needs, in 1 temporary folder that stands in for Léo's data folder
LINK = 'LINKS=$(mktemp -d) && for f in {input.data}; do ln -s "$PWD/$f" "$LINKS/"; done && '

wildcard_constraints:
    model="[^/]+",
    size=r"\d+",
    replicate=r"\d+",

localrules: all, training_data, testing_data, external_data, train, benchmark


# ── the 5 stages (targets only) ──
rule all:
    input:
        [f for files in STAGES.values() for f in files],


rule training_data:
    input:
        STAGES["training_data"],


rule testing_data:
    input:
        STAGES["testing_data"],


rule external_data:
    input:
        STAGES["external_data"],


rule train:
    input:
        STAGES["train"],


rule benchmark:
    input:
        STAGES["benchmark"],


# ── simulation: 1 job per model and size ──
# The seeds of simulate_data.py depend only on the data set, the model, the size, and the replicate,
# so a job per size writes the same tree sequences as one run over all sizes.
# The table of each job has one row per replicate: its seeds, its diversity and coalescence times,
# and the md5 of its file.
rule simulate_training_data:
    output:
        [OUT + "training_data/{model}/{model}_S{size}_" + f"{i}.trees"
         for i in range(N_FILES["training_data"])],
        table=OUT + "training_data/{model}/{model}_S{size}_simulations.csv",
    params:
        env=env("training_data", skip=PER_JOB),
        dir=OUT + "training_data/{model}",
    shell:
        "env {params.env} GNN_VS_SCENARIOS={wildcards.model} GNN_VS_SIZES={wildcards.size} "
        "GNN_VS_DIR={params.dir} " + PY + " -u src/simulate_data.py training_data" + copy_config("{params.dir}")


rule simulate_testing_data:
    output:
        [OUT + "testing_data/{model}/{model}_S{size}_" + f"{i}.trees"
         for i in range(N_FILES["testing_data"])],
        table=OUT + "testing_data/{model}/{model}_S{size}_simulations.csv",
    params:
        env=env("testing_data", skip=PER_JOB),
        dir=OUT + "testing_data/{model}",
    shell:
        "env {params.env} GNN_VS_SCENARIOS={wildcards.model} GNN_VS_SIZES={wildcards.size} "
        "GNN_VS_DIR={params.dir} " + PY + " -u src/simulate_data.py testing_data" + copy_config("{params.dir}")


# ── external tools on the testing data ──
# run_external_tools.py writes TOOL_OUTPUTS[tool] into 1 folder per testing file, and its printed lines
# go to the log. Relate and SINGER run 1 job per file. tsinfer and the true topology (parsimony,
# tsdate, and POLEGON) take seconds per file, so they run 1 job per model and size.
rule relate:
    input:
        trees=OUT + "testing_data/{model}/{model}_S{size}_{replicate}.trees",
        table=OUT + "testing_data/{model}/{model}_S{size}_simulations.csv",
    output:
        [OUT + "external_data/relate/testing_data/{model}/{model}_S{size}_{replicate}/" + name
         for name in TOOL_OUTPUTS["relate"]],
    log:
        OUT + "external_data/relate/testing_data/{model}/logs/{model}_S{size}_{replicate}.log",
    params:
        env=TOOL_ENV,
        dir=OUT + "external_data/relate/testing_data/{model}",
    shell:
        "env {params.env} " + PY + " -u src/run_external_tools.py relate {params.dir} {input.trees} > {log} 2>&1"
        + copy_config("{params.dir}")


rule singer:
    input:
        trees=OUT + "testing_data/{model}/{model}_S{size}_{replicate}.trees",
        table=OUT + "testing_data/{model}/{model}_S{size}_simulations.csv",
    output:
        [OUT + "external_data/singer/testing_data/{model}/{model}_S{size}_{replicate}/" + name
         for name in TOOL_OUTPUTS["singer"]],
    log:
        OUT + "external_data/singer/testing_data/{model}/logs/{model}_S{size}_{replicate}.log",
    params:
        env=TOOL_ENV,
        dir=OUT + "external_data/singer/testing_data/{model}",
    shell:
        "env {params.env} " + PY + " -u src/run_external_tools.py singer {params.dir} {input.trees} > {log} 2>&1"
        + copy_config("{params.dir}")


rule tsinfer:
    input:
        trees=lambda w: true_files("testing_data", w.model, [w.size]),
        table=OUT + "testing_data/{model}/{model}_S{size}_simulations.csv",
    output:
        [OUT + "external_data/tsinfer/testing_data/{model}/{model}_S{size}_" + f"{i}/" + name
         for i in range(N_FILES["testing_data"]) for name in TOOL_OUTPUTS["tsinfer"]],
    log:
        OUT + "external_data/tsinfer/testing_data/{model}/logs/{model}_S{size}.log",
    params:
        env=TOOL_ENV,
        dir=OUT + "external_data/tsinfer/testing_data/{model}",
    shell:
        "env {params.env} " + PY + " -u src/run_external_tools.py tsinfer {params.dir} {input.trees} > {log} 2>&1"
        + copy_config("{params.dir}")


rule true_topology:
    input:
        trees=lambda w: true_files("testing_data", w.model, [w.size]),
        table=OUT + "testing_data/{model}/{model}_S{size}_simulations.csv",
    output:
        [OUT + "external_data/true_topology/testing_data/{model}/{model}_S{size}_" + f"{i}/" + name
         for i in range(N_FILES["testing_data"]) for name in TOOL_OUTPUTS["true_topology"]],
    log:
        OUT + "external_data/true_topology/testing_data/{model}/logs/{model}_S{size}.log",
    params:
        env=TOOL_ENV,
        dir=OUT + "external_data/true_topology/testing_data/{model}",
    shell:
        "env {params.env} " + PY + " -u src/run_external_tools.py true_topology {params.dir} {input.trees} "
        "> {log} 2>&1" + copy_config("{params.dir}")


# ── train ──
# train.py saves next to itself at train.SAVE_PATH.
# GNN_TAG makes that name unique per prefix, and the second call imports train with the same environment to find the file.
# -P: the current folder is not searched, so a stray train.py there cannot replace src/train.py.
rule train_model:
    input:
        data=[f for m in TRAIN_MODELS for f in true_files("training_data", m)],
    output:
        f"{CKPT_DIR}/{PREFIX}.pt",
    log:
        f"{CKPT_DIR}/train.log",
    params:
        env=env("shared", "train") + f" GNN_VS_FILES={N_FILES['training_data']} GNN_TAG={PREFIX}",
    shell:
        LINK + "env {params.env} GNN_VS_DIR=$LINKS " + PY + " -u src/train.py 2>&1 | tee {log} && "
        "mv \"$(env {params.env} GNN_VS_DIR=$LINKS PYTHONPATH=src " + PY
        + " -P -c 'import train; print(train.SAVE_PATH)')\" {output} && rm -rf \"$LINKS\""
        + copy_config(CKPT_DIR)


# ── benchmark: the score tables of the external tools and of the checkpoint ──
# score_benchmark.py scores the files of one tool and writes one table per testing file. As the tools ran,
# relate and singer run one job per file, and tsinfer and the true topology one job per model and size.
rule score_tool_file:
    wildcard_constraints:
        tool="relate|singer",
    input:
        trees=OUT + "testing_data/{model}/{model}_S{size}_{replicate}.trees",
        table=OUT + "testing_data/{model}/{model}_S{size}_simulations.csv",
        tool=lambda w: [f"{OUT}external_data/{w.tool}/testing_data/{w.model}/{w.model}_S{w.size}_{w.replicate}/{name}"
                        for name in TOOL_OUTPUTS[w.tool]],
    output:
        SCORES_DIR + "/{tool}/{model}/{model}_S{size}_{replicate}.csv.gz",
    log:
        SCORES_DIR + "/{tool}/logs/{model}_S{size}_{replicate}.log",
    params:
        tool_dir=OUT + "external_data/{tool}/testing_data/{model}",
        dir=SCORES_DIR + "/{tool}/{model}",
    shell:
        PY + " -u src/score_benchmark.py {wildcards.tool} {params.tool_dir} {params.dir} {input.trees} > {log} 2>&1"
        + copy_config("{params.dir}")


rule score_tool_size:
    wildcard_constraints:
        tool="tsinfer|true_topology",
    input:
        trees=lambda w: true_files("testing_data", w.model, [w.size]),
        table=OUT + "testing_data/{model}/{model}_S{size}_simulations.csv",
        tool=lambda w: [f"{OUT}external_data/{w.tool}/testing_data/{w.model}/{w.model}_S{w.size}_{i}/{name}"
                        for i in range(N_FILES["testing_data"]) for name in TOOL_OUTPUTS[w.tool]],
    output:
        [SCORES_DIR + "/{tool}/{model}/{model}_S{size}_" + f"{i}.csv.gz" for i in range(N_FILES["testing_data"])],
    log:
        SCORES_DIR + "/{tool}/logs/{model}_S{size}.log",
    params:
        tool_dir=OUT + "external_data/{tool}/testing_data/{model}",
        dir=SCORES_DIR + "/{tool}/{model}",
    shell:
        PY + " -u src/score_benchmark.py {wildcards.tool} {params.tool_dir} {params.dir} {input.trees} > {log} 2>&1"
        + copy_config("{params.dir}")


# score_arggnn.py reads {PREFIX}.yaml next to the checkpoint, so the job links the checkpoint and the config into
# BENCH_DIR/checkpoint_links/{model}_S{size}/{PREFIX}/ with relative links. The script reads the shared and
# benchmark blocks of that config itself, since the shell would pass POLEGON's relative path. It keeps the dated
# trees of replicate 0 in BENCH_DIR/arggnn_trees/. Torch uses the threads of the job, 1, as for the paper's predictions.
# The folder at the depth of the links, for computing their relative targets.
LINK_DEPTH = f"{BENCH_DIR}/checkpoint_links/model_S0/{PREFIX}"

rule score_arggnn:
    input:
        ckpt=f"{CKPT_DIR}/{PREFIX}.pt",
        data=lambda w: true_files("testing_data", w.model, [w.size]),
    output:
        [BENCH_DIR + "/scores/{model}/{model}_S{size}_" + f"{i}.csv.gz" for i in range(N_FILES["testing_data"])],
    log:
        BENCH_DIR + "/logs/score_arggnn_{model}_S{size}.log",
    params:
        dir=BENCH_DIR + "/checkpoint_links/{model}_S{size}",
        links=BENCH_DIR + "/checkpoint_links/{model}_S{size}/" + PREFIX,
    threads: 1
    shell:
        "rm -rf {params.dir} && mkdir -p {params.links} && "
        "ln -s " + shlex.quote(os.path.relpath(f"{CKPT_DIR}/{PREFIX}.pt", LINK_DEPTH)) + " {params.links}/"
        + PREFIX + ".pt && "
        "ln -s " + shlex.quote(os.path.relpath(CONFIGFILE, LINK_DEPTH)) + " {params.links}/" + PREFIX + ".yaml && "
        "OMP_NUM_THREADS={threads} " + PY + " -u src/score_arggnn.py {params.links} " + BENCH_DIR + " {input.data} > {log} 2>&1 && "
        "rm -rf {params.dir}" + copy_config(BENCH_DIR)
