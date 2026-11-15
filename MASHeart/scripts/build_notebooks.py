"""Build English notebooks that carry their own verified source/data bootstrap."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def bundle_assets():
    """Bundle only executable source, configuration and attributed input data."""
    paths = sorted(list((ROOT / "masheart").glob("*.py")) + list((ROOT / "configs").glob("*.json"))
                   + [p for p in (ROOT / "data").rglob("*") if p.is_file()]
                   + [ROOT / "requirements.txt", ROOT / "requirements-t4.txt", ROOT / "LICENSE_DATA.md"])
    checksums = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 11, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    raw = stream.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest(), checksums


def make_helpers(cells):
    counter = [0]

    def markdown(text):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def step(title, description, code):
        counter[0] += 1
        markdown(f"## Step {counter[0]:02d}: {title}\n\n" + textwrap.dedent(description).strip())
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(code).strip() + "\n"))

    return markdown, step


def common_start(cells, exploration=False):
    md, step = make_helpers(cells)
    md("""
    # MASHeart: data exploration
    An English, development-only companion notebook. Explore the real scientific-text data before training;
    preserve the original official test and keep source groups together. Run top to bottom with Shift+Enter.
    """ if exploration else """
    # MASHeart: a step-by-step research workflow
    **Question + scientific statement → stance → multi-agent review → evidence gate → evaluated output.**

    This English notebook covers data setup, visualization, group splitting, actual training, validation,
    frozen test evaluation, errors, uncertainty, saved models and new-input inference. Its reusable code is
    the same `masheart/` package used by the command-line pipeline.

    **Task boundary:** `YS` means the supplied statement affirms the supplied research question.
    `NO` includes negation and lack of affirmation. These are text-stance labels, not clinical truth labels.
    Task B is introduced only as a lexical illustration; its benchmark is not trained or evaluated here.

    **Run on Colab:** upload this notebook, select a T4 GPU for QLoRA/MAS, and run in order. The source code,
    attributed data, and configuration are embedded, so no parent repository, API key, or Drive mount is
    required. The base model needs an Internet download on first use. CPU baseline steps run without a GPU.
    T4 training/full MAS can take longer than a lesson: prepare the measured run before presenting it.

    The embedded data originate with Alamri & Stevenson (2016), with official split alignment from
    Kim, Rahimi & Allan's BioClaim (2023). Preserve the supplied provenance and data license notices.
    """)
    encoded, sha, files = bundle_assets()
    bootstrap = "#@title Restore the verified MASHeart source/data bundle (collapse this technical cell)\n"
    bootstrap += f"ASSET_BASE64 = {encoded!r}\nASSET_SHA256 = {sha!r}\nASSET_FILES = {files!r}\n"
    bootstrap += textwrap.dedent('''
        import base64, hashlib, io, json, sys, zipfile
        from pathlib import Path

        candidates = [Path.cwd(), Path.cwd().parent]
        PROJECT_ROOT = next((p for p in candidates if (p / 'masheart' / 'data.py').is_file()
                             and (p / 'data' / 'manifest.json').is_file()), None)
        if PROJECT_ROOT is None:
            raw_archive = base64.b64decode(ASSET_BASE64)
            assert hashlib.sha256(raw_archive).hexdigest() == ASSET_SHA256
            PROJECT_ROOT = Path.cwd() / ('MASHeart_runtime_' + ASSET_SHA256[:10])
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
                for member in archive.infolist():
                    target = (PROJECT_ROOT / member.filename).resolve()
                    if not target.is_relative_to(PROJECT_ROOT.resolve()) or member.filename not in ASSET_FILES:
                        raise ValueError('Unexpected archive path; extraction stopped.')
                    content = archive.read(member)
                    assert hashlib.sha256(content).hexdigest() == ASSET_FILES[member.filename]
                    if target.exists() and target.read_bytes() != content:
                        raise RuntimeError('An embedded source file was changed. Use a fresh runtime directory.')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
        PROJECT_ROOT = PROJECT_ROOT.resolve()
        sys.path.insert(0, str(PROJECT_ROOT))
        print('Project:', PROJECT_ROOT)
        print('No parent HeartMAS checkout is required.')
    ''')
    step("Restore the standalone project", """
    **Input:** an embedded ZIP containing English Python modules, configuration, source data and license notices.
    **Output:** `PROJECT_ROOT`, independent of the original research repository.
    SHA-256 verifies byte identity; safe path checks keep every extracted file inside the runtime folder.
    The long Base64 string is transport, not a learned model. Existing project files are used when opened
    inside a complete MASHeart folder; the embedded copy makes a single uploaded notebook sufficient.
    """, bootstrap)
    cells[-1].metadata["cellView"] = "form"
    step("Install the measured CPU environment", """
    `importlib.metadata.version()` reads actual installed versions. The supplied requirements record the
    reference environment. Only missing/different packages are installed. If an older module is already in
    memory, restart after installation; pip cannot replace objects already imported into a kernel.
    The local verification host already has these packages and performs no installation or network request.
    """, '''
        import importlib.metadata, importlib.util, subprocess

        required = {}
        for line in (PROJECT_ROOT / 'requirements.txt').read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                name, version = line.strip().split('==')
                required[name] = version
        needed, loaded_changes = [], []
        for name, version in required.items():
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed = None
            if installed != version:
                needed.append(name + '==' + version)
                module = 'sklearn' if name == 'scikit-learn' else name
                if module in sys.modules:
                    loaded_changes.append(module)
        if needed:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *needed])
        if loaded_changes:
            raise RuntimeError('Packages were updated after import: restart the kernel and run from the top.')
        print({name: importlib.metadata.version(name) for name in required})
    ''')
    step("Import shared code and fix the research configuration", """
    `SEED=17` controls group randomization; it is not a guarantee of identical GPU arithmetic.
    `RUN_DIR` creates a new experiment directory and protects existing results. The code records the
    environment rather than assuming that equal seeds prove identical outputs on every machine.
    """, '''
        import os, time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import joblib
        import masheart
        from masheart.data import (verify_sources, load_frame, split_development, assert_disjoint,
                                   public_payload, compose_text)
        from masheart.baseline import make_classifier, make_dummy_classifier
        from masheart.metrics import evaluate, paired_cluster_bootstrap
        from IPython.display import display, Image

        assert Path(masheart.__file__).resolve().is_relative_to(PROJECT_ROOT)
        DATA_DIR = PROJECT_ROOT / 'data'
        CPU_CONFIG = json.loads((PROJECT_ROOT / 'configs' / 'cpu.json').read_text())
        SEED = CPU_CONFIG['seed']
        LABELS = ['NO', 'YS']
        RUN_DIR = PROJECT_ROOT / 'runs' / ('notebook_' + str(time.time_ns()))
        RUN_DIR.mkdir(parents=True, exist_ok=False)
        np.random.seed(SEED)
        print('New run:', RUN_DIR)
    ''')
    step("Verify data and read development only", """
    `verify_sources()` checks manifest paths, byte counts and SHA-256 without parsing test labels.
    `load_frame(..., 'dev')` reads UTF-8 JSONL and validates schema, ID uniqueness and labels.
    Preserve question/text source strings for quoting. Do not remove negation, numbers or conditions.
    The original release has no independent train set: this project uses educational development fitting.
    """, '''
        source_audit = verify_sources(DATA_DIR)
        dev = load_frame(DATA_DIR / 'raw' / 'a_dev.jsonl', 'dev')
        assert len(dev) == 125 and dev.group_id.nunique() == 11
        display(dev[['id', 'question', 'text', 'label', 'group_id']].head(3))
        display(dev.label.value_counts().reindex(LABELS).rename('examples').to_frame())
        print('Verified source files:', source_audit['files_verified'])
    ''')
    step("Visualize label balance and statement lengths", """
    **Scientific display:** export the measured table as well as PNG/SVG. The label chart exposes class
    imbalance; the histogram describes whitespace length, which differs from the model's subword tokens.
    These plots summarize development only and do not select examples based on test scores.
    """, '''
        development_profile = dev[['id', 'group_id', 'label']].copy()
        development_profile['whitespace_words'] = dev.text.map(lambda value: len(value.split()))
        development_profile.to_csv(RUN_DIR / 'development_profile.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
        dev.label.value_counts().reindex(LABELS).plot.bar(ax=axes[0], color=['#c56c39', '#287da8'])
        axes[0].set(title='Development label balance', xlabel='Stance', ylabel='Examples')
        axes[0].tick_params(axis='x', rotation=0)
        axes[1].hist(development_profile.whitespace_words, bins=10, color='#287da8', edgecolor='white')
        axes[1].set(title='Statement length', xlabel='Whitespace words', ylabel='Examples')
        fig.tight_layout()
        fig.savefig(RUN_DIR / 'development_overview.png', dpi=200, bbox_inches='tight')
        fig.savefig(RUN_DIR / 'development_overview.svg', bbox_inches='tight')
        plt.show()
    ''')
    step("Create group-disjoint training and validation", """
    `validation_fraction=0.25` means a fraction of source groups, not exactly 25% of rows.
    Whole linked groups remain together. For seed 17, development becomes 95 examples/8 groups for
    training and 30 examples/3 groups for validation. `assert_disjoint()` also checks IDs, PMIDs and
    normalized statements. The official test is not an input to this splitting function.
    """, '''
        train, validation = split_development(dev, SEED, CPU_CONFIG['validation_fraction'])
        split_audit = assert_disjoint(train, validation)
        split_summary = pd.DataFrame([
            {'split': name, 'n': len(frame), 'groups': frame.group_id.nunique(),
             'NO': int(frame.label.eq('NO').sum()), 'YS': int(frame.label.eq('YS').sum())}
            for name, frame in [('train', train), ('validation', validation)]
        ])
        display(split_summary)
        split_summary.to_csv(RUN_DIR / 'development_split.csv', index=False)
        assert len(train) == 95 and len(validation) == 30
    ''')
    step("Protect the inference feature boundary", """
    `public_payload()` returns question and text only. ID/group/PMID may be used to audit or join results,
    while gold labels are used only for fitting or scoring. `compose_text()` prepares the same two fields
    for the CPU classifier. Changing gold or metadata must not change the public model input.
    """, '''
        example = train.iloc[0].to_dict()
        modified = dict(example, label='NO' if example['label'] == 'YS' else 'YS', pmid='EXCLUDED')
        assert public_payload(example) == public_payload(modified)
        display(public_payload(example))
        print(compose_text(example))
    ''')
    return md, step


def build_main():
    cells = []
    md, step = common_start(cells)
    step("Understand TF-IDF before training", """
    `fit_transform()` learns vocabulary and inverse-document frequency and transforms its input.
    Later validation/test use only `transform()` inside the Pipeline. Word n-grams (1,2) retain features
    such as `not reduce`; `stop_words=None` preserves negation. This small synthetic example is never
    inserted into the real dataset. Smoothed IDF is `log((1+n)/(1+df))+1`, followed by vector normalization.
    """, '''
        from sklearn.feature_extraction.text import TfidfVectorizer
        toy = ['the trial reduced outcome', 'the trial did not reduce outcome',
               'the study reported another outcome']
        toy_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
        toy_values = toy_vectorizer.fit_transform(toy)
        toy_table = pd.DataFrame(toy_values.toarray(), columns=toy_vectorizer.get_feature_names_out())
        display(toy_table[['trial', 'outcome', 'not', 'not reduce']].round(3))
    ''')
    step("Train two CPU controls on permitted training rows", """
    `make_classifier()` builds two TF-IDF feature branches and **one** balanced logistic regression.
    Word(1,2) and char_wb(3,5) are representations, not independent agents. C=1 controls inverse
    regularization strength; max_iter=2000 limits optimizer iterations. `fit()` is actual training.
    `make_dummy_classifier()` learns the training-majority class; it never inspects test label frequencies.
    """, '''
        train_texts = [compose_text(row) for row in train.to_dict('records')]
        validation_texts = [compose_text(row) for row in validation.to_dict('records')]
        cpu_model = make_classifier({key: CPU_CONFIG[key] for key in ['C', 'max_iter', 'seed']})
        majority_model = make_dummy_classifier()
        started = time.perf_counter()
        cpu_model.fit(train_texts, train.label)
        majority_model.fit(np.zeros((len(train), 1)), train.label)
        print('Training seconds:', round(time.perf_counter() - started, 3))
        print('Classes:', cpu_model.classes_)
        print('Features:', len(cpu_model['features'].get_feature_names_out()))
    ''')
    step("Inspect learned weights and score validation", """
    A coefficient describes a fitted association, not causality. `predict()` applies the trained model;
    it does not refit TF-IDF. `evaluate()` keeps NO and YS in macro-F1 and retains INVALID as errors.
    Precision asks how often a predicted class is correct; recall asks how much of a gold class is found.
    Macro-F1 averages the two class F1 values equally. Probability-like classifier scores are uncalibrated.
    """, '''
        names = cpu_model['features'].get_feature_names_out()
        weights = cpu_model['classifier'].coef_[0]
        order = np.argsort(weights)
        display(pd.DataFrame({'feature': names[np.r_[order[:5], order[-5:]]],
                              'weight_toward_YS': weights[np.r_[order[:5], order[-5:]]]}))
        validation_cpu = cpu_model.predict(validation_texts).tolist()
        validation_majority = majority_model.predict(np.zeros((len(validation), 1))).tolist()
        validation_metrics = pd.DataFrame([
            evaluate(validation.label, validation_majority, 'majority'),
            evaluate(validation.label, validation_cpu, 'tfidf_logreg')])
        display(validation_metrics.round(4))
        validation_metrics.to_csv(RUN_DIR / 'validation_metrics.csv', index=False)
    ''')
    md("""
    ### Optional T4 experiment: pretraining, QLoRA and role-based MAS
    The base Qwen model already knows language from pretraining. QLoRA freezes its quantized base and
    updates small rank-8 adapters on q/v projections. NF4 reduces weight storage; FP16 is the compute
    type chosen for T4. Not all activation/optimizer memory is four-bit. The training target contains
    **only** the supplied JSON stance label: no invented gold explanations are added.

    All GPU steps below are guarded. In CPU mode, they describe the real implemented method and report
    that no GPU run occurred. Parser/gate examples are explicitly synthetic logic checks, not scores.
    """)
    step("Select GPU mode and install the pinned T4 stack", """
    Colab users select Runtime → Change runtime type → T4 GPU. CUDA availability is checked rather than
    assuming a Python import provides hardware. PyTorch/CUDA are kept as supplied by Colab.
    The project pins Transformers/PEFT/accelerate/bitsandbytes. Restart when changing an imported package.
    """, '''
        USE_T4 = False
        if importlib.util.find_spec('torch') is not None:
            import torch
            USE_T4 = torch.cuda.is_available()
        USE_T4 = USE_T4 and os.environ.get('MASHEART_CPU_ONLY') != '1'
        if USE_T4:
            pins = {'transformers': '4.57.1', 'peft': '0.17.1',
                    'accelerate': '1.10.1', 'bitsandbytes': '0.48.1'}
            for name, version in pins.items():
                if name in sys.modules and getattr(sys.modules[name], '__version__', None) != version:
                    raise RuntimeError('Restart before changing an imported GPU library: ' + name)
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                                   *[name + '==' + value for name, value in pins.items()]])
            print('GPU:', torch.cuda.get_device_name(0))
        else:
            print('CPU mode: no QLoRA training or LLM inference is claimed.')
        from masheart.gpu import T4Runtime, encode_supervised, parse_packet, gate_packets
        t4_config = json.loads((PROJECT_ROOT / 'configs' / 't4.json').read_text())
        gpu_runtime = None
    ''')
    step("Load the quantized model and inspect assistant-only loss masking", """
    `load_model()` uses the pinned model revision, NF4, double quantization, FP16 and gradient checkpointing.
    `encode_supervised()` returns token IDs, attention mask, and loss labels. Prompt positions get -100;
    answer tokens and EOS contribute loss. Padding also gets -100, while attention padding gets 0.
    Training context 512, inference input 1536 and new-output 192 are three different limits.
    """, '''
        if USE_T4:
            gpu_runtime = T4Runtime(output_dir=RUN_DIR / 't4', config=t4_config)
            gpu_runtime.load_model()
            encoded = encode_supervised(train.iloc[0].to_dict(), gpu_runtime.tokenizer,
                                        max_tokens=t4_config['max_train_tokens'])
            targets = [token for token in encoded['labels'] if token != -100]
            print('Input tokens:', len(encoded['input_ids']))
            print('Loss target:', gpu_runtime.tokenizer.decode(targets))
        else:
            print('Read masheart/gpu.py: encode_supervised() and collate_examples().')
    ''')
    step("Train QLoRA with a validation-selected checkpoint", """
    Batch1 × accumulation 8 gives a nominal effective batch 8 on one GPU. Train 3 epochs with lr1e-4,
    warmup 5%, gradient clipping 0.3 and checkpointing. The best checkpoint is selected by validation loss,
    not by test. Prediction-loss-only evaluation avoids retaining enormous vocabulary logits.
    Re-running this stage can be expensive; prepare the run before teaching. Loss is not 1−accuracy.
    """, '''
        if USE_T4:
            gpu_runtime.train(train, validation)
            print('Adapter saved:', gpu_runtime.adapter_dir)
            print('Training seconds:', gpu_runtime.training_seconds)
        else:
            print('Training not executed: no CUDA device.')
    ''')
    step("Compare pretrained and fine-tuned modes on validation", """
    The adapter checkpoint is chosen by validation loss. Then compare generated labels from base and
    adapter on the same 30 validation rows: select the adapter only if its macro-F1 is strictly higher;
    ties retain base. Fine-tuning can hurt on 95 examples. Reviewer/Verifier always use the base instruct
    model via disable_adapter, so the label-only adapter does not suppress their structured review.
    """, '''
        if USE_T4:
            gpu_validation = gpu_runtime.validate_and_select(validation)
            display(pd.DataFrame(gpu_validation).round(4))
            print('Analyst uses adapter:', gpu_runtime.analyst_use_adapter)
        else:
            print('No pretrained/fine-tuned performance measurements exist for this CPU notebook run.')
    ''')
    step("Understand the exact-source evidence gate", """
    `parse_packet()` separates schema validity from quote validity. Reviewer/Verifier quotes must occur
    exactly once and unchanged in the source, with a nonempty rationale. A sourced quote is a provenance
    check, not proof of correct meaning. `gate_packets()` accepts a changed valid draft only when both
    later agents agree on the new label and have valid source quotes; otherwise it keeps the draft and
    marks review. An invalid draft may be repaired by a sourced final, still with a review flag.
    """, '''
        synthetic_source = {'question': 'Does X reduce outcome Y?',
                            'text': 'The synthetic study did not reduce outcome Y.'}
        synthetic_generation = {'status': 'ok', 'raw': json.dumps(
            {'label': 'NO', 'quote': 'an invented quote', 'rationale': 'A synthetic test.'})}
        checked = parse_packet(synthetic_generation, synthetic_source, require_quote=True)
        assert not checked['quote_valid']
        initial = {'label': 'YS', 'schema_valid': True, 'quote_valid': False}
        review = {'label': 'NO', 'schema_valid': True, 'quote_valid': False}
        final = {'label': 'NO', 'schema_valid': True, 'quote_valid': True}
        blocked = gate_packets(initial, review, final)
        assert blocked['label'] == 'YS'
        display({'synthetic_logic_check_only': True, 'gate': blocked})
    ''')
    step("Freeze the method before loading test labels", """
    Save configuration, training/validation IDs, source audit and trained CPU hash. The T4 protocol also
    freezes prompts, revision, mode, adapter and implementation. MAS uses Analyst → Reviewer → Verifier;
    these are role prompts using one base checkpoint, not independently trained specialist doctors.
    """, '''
        frozen_cpu_hash = joblib.hash(cpu_model)
        frozen = {'config': CPU_CONFIG, 'train_ids': train.id.tolist(),
                  'validation_ids': validation.id.tolist(), 'cpu_hash': frozen_cpu_hash,
                  'source_audit': source_audit, 'gpu_executed': USE_T4}
        (RUN_DIR / 'notebook_frozen_protocol.json').write_text(json.dumps(frozen, indent=2))
        if USE_T4:
            gpu_runtime.freeze_protocol(source_audit)
            validation_traces = gpu_runtime.run_split(validation, 'notebook_validation')
            print(json.dumps(validation_traces[0], indent=2))
        print('CPU hash:', frozen_cpu_hash)
    ''')
    step("Load the preserved official test and predict every row", """
    Test is loaded after freezing. Test input always passes through public_payload/compose_text; gold is
    attached only for scoring. The protocol preserves all 134 official examples and10 source groups.
    The original project has prior development/test exposure: this is not a historically pristine holdout.
    """, '''
        test = load_frame(DATA_DIR / 'raw' / 'a_test.jsonl', 'test')
        assert_disjoint(train, test)
        assert_disjoint(validation, test)
        assert joblib.hash(cpu_model) == frozen_cpu_hash
        test_texts = [compose_text(row) for row in test.to_dict('records')]
        cpu_predictions = cpu_model.predict(test_texts).tolist()
        majority_predictions = majority_model.predict(np.zeros((len(test), 1))).tolist()
        predictions = test[['id', 'group_id', 'question', 'text', 'label']].rename(columns={'label': 'gold'}).copy()
        predictions['majority'] = majority_predictions
        predictions['tfidf_logreg'] = cpu_predictions
        metrics = [evaluate(test.label, majority_predictions, 'majority'),
                   evaluate(test.label, cpu_predictions, 'tfidf_logreg')]
        print('Complete test rows:', len(predictions))
        assert len(predictions) == 134 and predictions.id.is_unique
    ''')
    step("Execute the full three-stage MAS test when T4 is available", """
    `run_split()` records every raw response, parsed packet, decision, failure, token count and elapsed time.
    Protocol/input-bound JSONL checkpoints support continuation without silently retrying failed examples.
    `direct_label` is the draft from the same trace: Direct costs one model step, MAS normally three.
    Keep INVALID in the denominator. Compare complete 134-row outputs, not a few selected examples.
    """, '''
        llm_records = []
        if USE_T4:
            llm_records = gpu_runtime.run_split(test, 'notebook_test')
            assert len(llm_records) == len(test)
            predictions['direct'] = [record['direct_label'] for record in llm_records]
            predictions['mas'] = [record['label'] for record in llm_records]
            predictions['review'] = [record['review'] for record in llm_records]
            metrics += [evaluate(test.label, predictions.direct, 'qwen_direct'),
                        evaluate(test.label, predictions.mas, 'qwen_mas')]
        else:
            print('LLM test not run; no GPU scores are filled in.')
        metric_table = pd.DataFrame(metrics)
        display(metric_table.round(4))
    ''')
    step("Read confusion matrices and per-class recall", """
    Rows are gold; columns are predictions. Add INVALID to the matrix when needed so failures remain
    visible. A majority predictor can have high accuracy and zero NO recall. Always read macro-F1,
    class support and per-class results alongside accuracy.
    """, '''
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
        column = 'mas' if USE_T4 else 'tfidf_logreg'
        order = LABELS + (['INVALID'] if 'INVALID' in predictions[column].tolist() else [])
        matrix = confusion_matrix(test.label, predictions[column], labels=order)
        ConfusionMatrixDisplay(matrix, display_labels=order).plot(cmap='Blues', colorbar=False)
        plt.title('Complete official test: ' + column)
        plt.tight_layout(); plt.show()
        print(classification_report(test.label, predictions[column], labels=LABELS, zero_division=0))
    ''')
    step("Measure selective coverage and computational cost", """
    A review flag does not erase a prediction from complete-test metrics. Coverage is the fraction not
    flagged for review. Compare Direct and MAS on exactly the same accepted subset, and show its size.
    Token/time measurements come from the stored generations, not from a fixed assumed call count.
    """, '''
        if USE_T4:
            mask = ~predictions.review.astype(bool)
            print('Coverage:', float(mask.mean()), '| accepted:', int(mask.sum()), '/', len(mask))
            if mask.any():
                display(pd.DataFrame([evaluate(test.label[mask], predictions[name][mask], name)
                                      for name in ['direct', 'mas']]).round(4))
            cost_rows = []
            for record in llm_records:
                for call in record['trace']:
                    generation = call['generation']
                    cost_rows.append({'role': call['role'], 'seconds': generation['elapsed_seconds'],
                                      'calls': generation['model_calls'],
                                      'input_tokens': generation['input_tokens'],
                                      'output_tokens': generation['output_tokens']})
            display(pd.DataFrame(cost_rows).groupby('role').sum().round(3))
        else:
            print('GPU coverage/cost cannot be measured from CPU controls.')
    ''')
    step("Export actual GPU training and review figures", """
    When a real T4 run exists, export its assistant-token train/validation loss, full-test metrics,
    confusion matrix, measured cost and beneficial/harmful review outcomes. Every figure includes PNG,
    SVG, a source CSV and a hash manifest. CPU logic checks cannot supply any of these GPU measurements.
    """, '''
        if USE_T4:
            from masheart.gpu import generate_gpu_figures
            gpu_test_metrics = [row for row in metrics if row['system'] in ['qwen_direct', 'qwen_mas']]
            gpu_figures = generate_gpu_figures(gpu_runtime.output_dir, test, llm_records, gpu_test_metrics)
            print('Measured GPU figure types:', len(gpu_figures['figures']))
            for path in sorted((gpu_runtime.output_dir / 'figures').glob('*.png')):
                display(Image(filename=str(path)))
        else:
            print('No GPU figures are fabricated for a CPU-only run.')
    ''')
    step("Inspect errors and estimate uncertainty by source group", """
    Read the full question, statement and conditions before interpreting an error. Do not tune on test
    after seeing these errors. Paired cluster bootstrap resamples complete source groups, applies the
    same selected rows to both systems, and reports macro-F1(B)−macro-F1(A). With only 10 groups, the
    percentile interval is uncertain and is not a clinical guarantee or automatic evidence of superiority.
    """, '''
        column_a, column_b = ('direct', 'mas') if USE_T4 else ('majority', 'tfidf_logreg')
        errors = predictions[predictions.gold.ne(predictions[column_b])]
        with pd.option_context('display.max_colwidth', 180):
            display(errors[['question', 'text', 'gold', column_b]].head(5))
        bootstrap = paired_cluster_bootstrap(test.label, predictions[column_a], predictions[column_b],
                                              test.group_id, rounds=CPU_CONFIG['bootstrap_rounds'], seed=SEED)
        print(column_b, '-', column_a, '| CI95:', bootstrap['ci_lower'], bootstrap['ci_upper'])
        plt.hist(bootstrap['differences'], bins=25, color='#287da8', edgecolor='white')
        plt.axvline(0, color='black', linestyle='--')
        plt.xlabel('Paired cluster bootstrap macro-F1 difference'); plt.ylabel('Draws')
        plt.tight_layout(); plt.show()
    ''')
    step("Save and reload the model and measured outputs", """
    Persist the Pipeline, CSV predictions, metrics, split and environment. Check a validation roundtrip;
    serialization is not another training stage. Adapters alone do not include all 4B base weights.
    """, '''
        model_path = RUN_DIR / 'notebook_tfidf_logreg.joblib'
        joblib.dump(cpu_model, model_path)
        restored = joblib.load(model_path)
        assert restored.predict(validation_texts).tolist() == validation_cpu
        assert joblib.hash(cpu_model) == frozen_cpu_hash
        predictions.to_csv(RUN_DIR / 'notebook_test_predictions.csv', index=False)
        metric_table.to_csv(RUN_DIR / 'notebook_test_metrics.csv', index=False)
        print('Saved model:', model_path)
    ''')
    step("Generate the canonical reproducible research artifact set", """
    The command-line experiment is the same fixed CPU method implemented in reusable functions. Run it
    once here to export the full scientific artifact set: eight PNG/SVG figure pairs, their exact CSV
    inputs, validation/test tables, saved models, source/split audits, frozen hashes, environment,
    complete manifest and a research report. This is a repeatability exercise with unchanged settings,
    not a fresh held-out assessment or a new round of test-driven model selection.
    """, '''
        from masheart.experiment import run_cpu_experiment
        research_run = RUN_DIR / 'cpu_research'
        research_summary = run_cpu_experiment(DATA_DIR, research_run, CPU_CONFIG)
        canonical_predictions = pd.read_csv(research_run / 'tables' / 'test_predictions.csv')
        assert canonical_predictions.tfidf_logreg.tolist() == cpu_predictions
        display(pd.read_csv(research_run / 'tables' / 'test_metrics.csv').round(4))
        for path in sorted((research_run / 'figures').glob('*.png')):
            display(Image(filename=str(path)))
        print((research_run / 'reports' / 'CPU_RESEARCH_REPORT.md').read_text())
    ''')
    step("Test a new synthetic statement and export your run", """
    A new interactive input has no independently supplied gold, so do not report accuracy from one demo.
    The English synthetic example is for language behavior, not treatment advice. Task B can be explored
    by comparing surface tokens, but lexical difference is not a semantic contradiction annotation.
    Download the result ZIP before a temporary Colab runtime is reclaimed.
    """, '''
        new_source = {'question': 'Does treatment X reduce blood pressure compared with placebo?',
                      'text': 'The synthetic trial did not find a reduction after eight weeks compared with placebo.'}
        if USE_T4:
            print(json.dumps(gpu_runtime.run_mas(new_source), indent=2))
        else:
            print({'system': 'CPU control', 'label': restored.predict([compose_text(new_source)])[0]})
        import shutil
        export_path = shutil.make_archive(str(RUN_DIR), 'zip', root_dir=RUN_DIR)
        print('Result ZIP:', export_path)
        DOWNLOAD_RESULTS = False
        if DOWNLOAD_RESULTS:
            from google.colab import files
            files.download(export_path)
    ''')
    md("""
    ## Research questions and reporting checklist
    1. Why does a statement about a different outcome receive NO even without the word `not`?
    2. Which operations fit parameters, and which merely transform or validate data?
    3. Why can a majority classifier look accurate while missing every minority-class example?
    4. Did fine-tuning improve validation? Did the extra MAS calls improve complete-test results?
    5. How often did the evidence gate block or allow a change? Can sourced quotes still be misinterpreted?
    6. What are the number of source groups, confidence interval, failures, coverage and computational cost?

    Report measured results, source history and limitations. The packaged CPU reference is a real run;
    the GPU status file is `not_run` until an actual T4 experiment is performed.

    Primary references: [PCC (2016)](https://doi.org/10.1186/s13326-016-0083-z),
    [Cond-NLI/BioClaim (2023)](https://aclanthology.org/2023.findings-emnlp.456/),
    [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
    [Trainer 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer),
    [PEFT quantization](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization).
    """)
    return cells


def build_exploration():
    cells = []
    md, step = common_start(cells, exploration=True)
    step("Inspect source groups without opening test", """
    Several statements can arise from a related source group. Count observations and classes by group;
    this motivates group splitting and later cluster bootstrap instead of assuming rows are independent.
    """, '''
        grouped = dev.groupby(['group_id', 'label']).size().unstack(fill_value=0).reindex(columns=LABELS)
        grouped.to_csv(RUN_DIR / 'development_groups.csv')
        grouped.plot.barh(stacked=True, figsize=(8, 5), color=['#c56c39', '#287da8'])
        plt.xlabel('Statements'); plt.ylabel('Linked source group')
        plt.title('Development-only group structure'); plt.tight_layout(); plt.show()
    ''')
    step("Compare original text with an audit key", """
    Audit normalization changes case/whitespace only; do not overwrite the source. Scope, negation,
    population, time and numbers carry meaning. A token count is not a clinical evidence-quality score.
    """, '''
        from masheart.data import normalize_for_audit
        synthetic = '  Treatment did NOT change outcome after 2 days.  '
        print('Source:', repr(synthetic))
        print('Audit key:', normalize_for_audit(synthetic))
        print('Development normalized duplicate rows:', dev.text.map(normalize_for_audit).duplicated().sum())
        print('Prepared files:', sorted(p.name for p in RUN_DIR.iterdir()))
    ''')
    md("""
    ## Continue to the full experiment
    Open **MASHeart_Step_by_Step.ipynb** for actual CPU training and optional T4 QLoRA/MAS, frozen evaluation,
    figures, uncertainty and exported artifacts. This exploratory notebook deliberately does not train
    a model, read official test labels or present a performance estimate.
    """)
    return cells


def write_notebook(name, cells, gpu):
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": name, "provenance": []},
        "masheart": {"version": "1.0.0", "standalone_assets": True, "language": "English"},
    })
    if gpu:
        notebook.metadata.update({"accelerator": "GPU", "gpuType": "T4"})
    for index, cell in enumerate(notebook.cells):
        cell.id = f"masheart-{index:03d}"
    nbformat.validate(notebook)
    destination = ROOT / "notebooks" / name
    destination.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, destination)
    print(f"Built {name}: {len(cells)} cells")


if __name__ == "__main__":
    write_notebook("01_Data_Exploration.ipynb", build_exploration(), False)
    write_notebook("MASHeart_Step_by_Step.ipynb", build_main(), True)
