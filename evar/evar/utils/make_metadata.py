"""Metadata maker for downstream tasks.

This program will create metadata files for tasks.
A metadata file is a .csv which contains file_name, label, split columns, and rows for all the audio samples
that belong to the task dataset.

Usage:
    python -m utils.preprocess_ds nsynth /path/to/nsynth
    python -m utils.preprocess_ds spcv2 /path/to/speech_commands_v0.02
    python -m utils.preprocess_ds spcv1 /path/to/speech_commands_v0.01
    python -m utils.preprocess_ds surge /path/to/surge
    python -m utils.preprocess_ds us8k /path/to/UrbanSound8K
    python -m utils.preprocess_ds vc1 /path/to/VoxCeleb1
"""

import logging
import os
import tempfile
import zipfile
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import requests
import fire


log = logging.getLogger(__name__)

MTT_SYNONYMS = [['beat', 'beats'],
            ['chant', 'chanting'],
            ['choir', 'choral'],
            ['classical', 'clasical', 'classic'],
            ['drums', 'drum'],
            ['electronic', 'electro', 'electronica', 'electric'],
            ['fast', 'fast beat', 'quick'],
            ['female', 'female singer', 'female singing', 'female vocals', 'female voice', 'woman', 'woman singing',
             'women'],
            ['flute', 'flutes'],
            ['guitar', 'guitars'],
            ['hard', 'hard rock'],
            ['harpsichord', 'harpsicord'],
            ['heavy', 'heavy metal', 'metal'],
            ['horn', 'horns'],
            ['indian', 'india'],
            ['jazz', 'jazzy'],
            ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
            ['no beat', 'no drums'],
            ['no vocals', 'no singer', 'no singing', 'no vocal', 'no voice', 'no voices', 'instrumental'],
            ['opera', 'operatic'],
            ['orchestra', 'orchestral'],
            ['quiet', 'silence'],
            ['singing', 'singer'],
            ['space', 'spacey'],
            ['strings', 'string'],
            ['synth', 'synthesizer'],
            ['violin', 'violins'],
            ['vocal', 'vocals', 'voice', 'voices'],
            ['weird', 'strange']]


def flatten_list(lists):
    return list(chain.from_iterable(lists))


def download_and_extract_zip_archive(url: str, tempdir: str):
    # Ensure tempdir exists or create it
    os.makedirs(tempdir, exist_ok=True)

    # Download the archive
    response = requests.get(url)
    response.raise_for_status()

    # Write the downloaded content to a temporary file
    with tempfile.NamedTemporaryFile(dir=tempdir, suffix=".zip") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

        # Extract the archive
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(tempdir)

    # Return the directory where the contents are extracted
    return tempdir


def key_to_integer(key_str):
    # Define mapping of notes to integers
    note_to_int = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
                   'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}

    keys = []

    for key in key_str.split(' / '):
        # Extract note and mode from key string
        elems = key.split(' ')
        if len(elems) < 2:
            log.error(f'Invalid key string: {key_str}')
            continue
        note, mode = elems[:2]

        # Convert note to integer
        note_int = note_to_int.get(note.capitalize())
        if note_int is None:
            log.error(f'Note {note} not recognized')
            continue

        # Assign different offsets for major and minor modes
        if mode not in ('major', 'minor'):
            log.error(f'Invalid mode: {mode}')
            continue

        keys.append(note_int + 12 if mode == "minor" else note_int)

    return keys


def merge_synonyms_mtt(row):
    merged_values = {}
    for synonym_group in MTT_SYNONYMS:
        values = [row[col] for col in synonym_group if col in row.index]
        merged_values[synonym_group[0]] = max(values)  # Choose max value if multiple synonyms are present
    return merged_values



BASE = Path('evar/metadata')


# UrbanSound8K https://urbansounddataset.weebly.com/urbansound8k.html

def convert_us8k_metadata(root):
    US8K = Path(root)
    df = pd.read_csv(US8K/f'metadata/UrbanSound8K.csv')
    df['file_name'] = df.fold.map(lambda x: f'audio/fold{x}/') + df.slice_file_name

    re_df = pd.DataFrame(df['class'].values, index=df.file_name, columns=['label'])
    re_df['fold'] = df.fold.values
    re_df.to_csv(f'{BASE}/us8k.csv')

    # test
    df = pd.read_csv(f'{BASE}/us8k.csv').set_index('file_name')
    labels = df.label.values
    classes = sorted(set(list(labels)))
    assert len(classes) == 10
    assert len(df) == 8732
    assert np.all([fold in [1,2,3,4,5,6,7,8,9,10] for fold in df.fold.values])
    print(f'Created {BASE}/us8k.csv - test passed')


def us8k(root):
    convert_us8k_metadata(root)


# ESC-50 https://github.com/karolpiczak/ESC-50

def convert_esc50_metadata(root):
    root = Path(root)
    df = pd.read_csv(root / 'meta/esc50.csv')
    repl_map = {'filename': 'file_name', 'category': 'label'}
    df.columns = [repl_map[c] if c in repl_map else c for c in df.columns]
    df.file_name = 'audio/' + df.file_name
    df.to_csv(f'{BASE}/esc50.csv', index=None)

    # test
    df = pd.read_csv(f'{BASE}/esc50.csv').set_index('file_name')
    labels = df.label.values
    classes = sorted(set(list(labels)))
    assert len(classes) == 50
    assert len(df) == 2000
    assert np.all([fold in [1,2,3,4,5] for fold in df.fold.values])
    print(f'{BASE}/esc50.csv - test passed')


def esc50(root):
    convert_esc50_metadata(root)


def convert_giantsteps_stems_metadata(root):
    metadata_file = BASE / 'giantsteps_stems.csv'
    downloads_dir = Path("/home/alain/code/evar/downloads/giantsteps")
    print(downloads_dir.resolve())

    # training set is Giansteps MTG, test set is Giantsteps
    metadata = []

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        annot_dir = downloads_dir / subdir.name / "annotations" / "key"

        for file_path in sorted(subdir.glob("**/*.wav")):
            file_name = file_path.relative_to(root)
            annotation = (annot_dir / file_name.name).with_suffix(".key")

            with annotation.open('r') as f:
                elems = f.read().split('\t')
                key = elems[0]
                if len(elems) > 1:
                    split = "train" if elems[1] == "2" else "val"
                else:
                    split = "test"

            key_id = key_to_integer(key)
            if len(key_id) == 0:
                continue

            metadata.append((str(file_name), ','.join(str(k) for k in key_id), key, split))

    df = pd.DataFrame(metadata, columns=['file_name', 'key', "detailed_key", "split"])
    df.to_csv(metadata_file, index=False)



def giantsteps_stems(root: str):
    convert_giantsteps_stems_metadata(Path(root))


# GTZAN
# Thanks to https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/gtzan/gtzan.py
# Thanks to https://github.com/xavierfav/coala
# Splits follow https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split
def convert_gtzan_metadata(root):
    # Make a list of files
    file_labels = [[str(f).replace(root + '/', ''), f.parent.name] for f in sorted(Path(root).glob('*/*.wav'))]
    df = pd.DataFrame(file_labels, columns=['file_name', 'label'])
    # Set splits
    contents = requests.get('https://raw.githubusercontent.com/jongpillee/music_dataset_split/master/GTZAN_split/test_filtered.txt')
    _files = contents.text.splitlines()
    df.loc[df.file_name.isin(_files), 'split'] = 'test'
    contents = requests.get('https://raw.githubusercontent.com/jongpillee/music_dataset_split/master/GTZAN_split/valid_filtered.txt')
    _files = contents.text.splitlines()
    df.loc[df.file_name.isin(_files), 'split'] = 'valid'
    contents = requests.get('https://raw.githubusercontent.com/jongpillee/music_dataset_split/master/GTZAN_split/train_filtered.txt')
    _files = contents.text.splitlines()
    df.loc[df.file_name.isin(_files), 'split'] = 'train'
    np.all(df.isna().values) == False
    df.to_csv(f'{BASE}/gtzan.csv', index=None)

    # test
    df = pd.read_csv(f'{BASE}/gtzan.csv').set_index('file_name')
    labels = df.label.values
    classes = sorted(set(list(labels)))
    assert len(classes) == 10
    assert len(df) == 1000
    print(f'{BASE}/gtzan.csv - test passed')


def convert_gtzan_stems_metadata(root: Path):
    metadata_file = BASE / 'gtzan_stems.csv'

    # globals
    split_url_fmt = "https://raw.githubusercontent.com/jongpillee/music_dataset_split/master/GTZAN_split/{}_filtered.txt"
    tempo_url_fmt = "https://raw.githubusercontent.com/superbock/ISMIR2019/master/gtzan/annotations/tempo/gtzan_{}_{}.bpm"

    metadata = []
    pbar = tqdm(sorted(root.glob('**/*.wav')), leave=False)

    for f in pbar:
        pbar.set_description(f.name)

        # retrieve fname and genre
        fname = f.relative_to(root)
        genre, track_id, ext = fname.name.split('.')

        # retrieve tempo annotation
        tempo_url = tempo_url_fmt.format(genre, track_id)
        contents = requests.get(tempo_url)
        if contents.status_code != 200:
            log.error(f"Received status code {contents.status_code} for the following URL: {tempo_url}")
            continue
        tempo = float(contents.text.strip())

        metadata.append((str(fname), genre, tempo))

    df = pd.DataFrame(metadata, columns=['file_name', 'genre', 'tempo'])

    for split in ['train', 'valid', 'test']:
        contents = requests.get(split_url_fmt.format(split))
        _files = contents.text.splitlines()
        df.loc[df.file_name.isin(_files), 'split'] = split

    print(df)
    df.to_csv(metadata_file, index=False)

    # test
    df = pd.read_csv(metadata_file).set_index('file_name')
    labels = df.genre.values
    classes = sorted(set(list(labels)))
    assert len(classes) == 10, f"There should be 10 classes but got {classes}."
    print(metadata_file, ' - test passed')


def gtzan(root):
    convert_gtzan_metadata(root)


def gtzan_stems(root):
    convert_gtzan_stems_metadata(Path(root))


# Magna Tag a Tune: https://github.com/keunwoochoi/magnatagatune-list?tab=readme-ov-file
def convert_mtt_metadata(root: Path):
    metadata_file = BASE / 'mtt.csv'

    # constants
    annotations_url = "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"
    top50_tags = [
        "guitar", "classical", "slow", "techno", "strings", "drums", "electronic", "rock", "fast", "piano", "ambient",
        "beat", "violin", "vocal", "synth", "female", "indian", "opera", "male", "singing", "vocals", "no vocals",
        "harpsichord", "loud", "quiet", "flute", "woman", "male vocal", "no vocal", "pop", "soft", "sitar", "solo",
        "man", "classic", "choir", "voice", "new age", "dance", "male voice", "female vocal", "beats", "harp", "cello",
        "no voice", "weird", "country", "metal", "female voice", "choral"
    ]
    broken_files = [
        "norine_braun-now_and_zen-08-gently-117-146",
        "jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233",
        "american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117"
    ]

    annotations_file = root / "annotations.csv"

    contents = requests.get(annotations_url)

    with open(annotations_file, 'w') as f:
        f.write(contents.text)

    annot_df = pd.read_csv(annotations_file, sep='\t')
    merged_df = annot_df.apply(merge_synonyms_mtt, axis=1, result_type="expand")

    # Add columns without synonyms back to the DataFrame
    cols_without_synonyms = [col for col in annot_df.columns if col not in [w for group in MTT_SYNONYMS for w in group]]
    for col in cols_without_synonyms:
        merged_df[col] = annot_df[col]

    df = merged_df[["mp3_path"] + [col for col in top50_tags if col in merged_df.columns]]

    df['file_name'] = df['mp3_path'].str.replace('.mp3', '.wav')

    # Combine label columns into one list column
    df['label'] = (df[df.columns.difference(['mp3_path', 'file_name'])]
                   .apply(list, axis=1)
                   .apply(lambda l: ','.join([str(i) for i, c in enumerate(l) if c])))

    # Drop individual label columns
    df.drop(["mp3_path"], axis=1, inplace=True)

    # Create 'split' column based on filename
    df['split'] = df['file_name'].apply(
        lambda x: 'val' if x.startswith('c') else ('test' if x.startswith(('d', 'e', 'f')) else 'train')
    )

    # reorder columns
    cols = df.columns.tolist()
    new_cols = ["file_name", "label", "split"] + sorted([c for c in cols if c not in ["file_name", "label", "split"]])

    df = df[new_cols]

    # remove broken files
    for broken in broken_files:
        df = df[~df['file_name'].str.contains(broken)]

    print(df)
    print(df.iloc[0])

    df.to_csv(metadata_file, index=False)


def mtt(root: str):
    convert_mtt_metadata(Path(root))


# NSynth https://magenta.tensorflow.org/datasets/nsynth

def convert_nsynth_metadata(root, nsynth='nsynth', label_column='instrument_family_str',
    n_samples=305979, files=None, filter_=lambda x: x, label_fn=None):

    def read_meta(root, mode):
        j = json.load(open(f'{root}/nsynth-{mode}/examples.json'))
        loop_indexes = files if files and mode == 'train' else j
        file_names = [f'nsynth-{mode}/audio/{file_id}.wav' for file_id in loop_indexes]
        labels = [j[x][label_column] if label_fn is None else label_fn(j[x]) for x in loop_indexes]
        return pd.DataFrame({'file_name': file_names, 'label': labels, 'split': mode})

    df = pd.concat([read_meta(root, mode) for mode in ['train', 'valid', 'test']], ignore_index=True)
    df = filter_(df)
    df.to_csv(f'{BASE}/{nsynth}.csv')

    df = pd.read_csv(f'{BASE}/{nsynth}.csv')
    assert len(df) == n_samples, f'{len(df)}'
    print(f'Created {nsynth}.csv - test passed')
    print(f'train:valid:test = {sum(df.split == "train")}:{sum(df.split == "valid")}:{sum(df.split == "test")}')


def nsynth(root):
    convert_nsynth_metadata(root)


# FSDnoisy18k http://www.eduardofonseca.net/FSDnoisy18k/

def convert_fsdnoisy18k_metadata(root):
    FSD = Path(root)
    train_df = pd.read_csv(FSD/f'FSDnoisy18k.meta/train.csv')
    # train_df = train_df[train_df.manually_verified != 0]
    # train_df = train_df[train_df.noisy_small == 0]
    test_df = pd.read_csv(FSD/f'FSDnoisy18k.meta/test.csv')
    # fname := split/fname
    train_df['fname'] = 'FSDnoisy18k.audio_train/' + train_df.fname
    test_df['fname'] = 'FSDnoisy18k.audio_test/' + test_df.fname
    # split. train -> train + val
    train_df['split'] = 'train'
    valid_index = np.random.choice(train_df.index.values, int(len(train_df) * 0.1), replace=False)
    train_df.loc[valid_index, 'split'] = 'valid'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df], ignore_index=True)
    # filename -> file_name
    df.columns = [c if c != 'fname' else 'file_name' for c in df.columns]
    df.to_csv(f'{BASE}/fsdnoisy18k.csv', index=False)
    n_samples = len(df)

    df = pd.read_csv(f'{BASE}/fsdnoisy18k.csv')
    assert len(df) == n_samples, f'{len(df)}'
    print(f'Created fsdnoisy18k.csv - test passed')


def fsdnoisy18k(root):
    convert_fsdnoisy18k_metadata(root)


# FSD50K https://arxiv.org/abs/2010.00475

def convert_fsd50k_multilabel(FSD50K_root):
    FSD = Path(FSD50K_root)
    df = pd.read_csv(FSD/f'FSD50K.ground_truth/dev.csv')
    df['split'] = df['split'].map({'train': 'train', 'val': 'valid'})
    df['file_name'] = df.fname.apply(lambda s: f'FSD50K.dev_audio/{s}.wav')
    dftest = pd.read_csv(FSD/f'FSD50K.ground_truth/eval.csv')
    dftest['split'] = 'test'
    dftest['file_name'] = dftest.fname.apply(lambda s: f'FSD50K.eval_audio/{s}.wav')
    df = pd.concat([df, dftest], ignore_index=True)
    df['label'] = df.labels
    
    df = df[['file_name', 'label', 'split']]
    df.to_csv(f'{BASE}/fsd50k.csv')
    return df


def fsd50k(root):
    convert_fsd50k_multilabel(root)


# Speech Command https://arxiv.org/abs/1804.03209

def convert_spc_metadata(root, version=2):
    ROOT = Path(root)
    files = sorted(ROOT.glob('[a-z]*/*.wav'))
    
    labels = [f.parent.name for f in files]
    file_names = [f'{f.parent.name}/{f.name}' for f in files]
    df = pd.DataFrame({'file_name': file_names, 'label': labels})
    assert len(df) == [64721, 105829][version - 1] # v1, v2
    assert len(set(labels)) == [30, 35][version - 1] # v1, v2
    
    with open(ROOT/'validation_list.txt') as f:
        vals = [l.strip() for l in f.readlines()]
    with open(ROOT/'testing_list.txt') as f:
        tests = [l.strip() for l in f.readlines()]
    assert len(vals) == [6798, 9981][version - 1] # v1, v2
    assert len(tests) == [6835, 11005][version - 1] # v1, v2
    
    df['split'] = 'train'
    df.loc[df.file_name.isin(vals), 'split'] = 'valid'
    df.loc[df.file_name.isin(tests), 'split'] = 'test'
    assert len(df[df.split == 'valid']) == [6798, 9981][version - 1] # v1, v2
    assert len(df[df.split == 'test']) == [6835, 11005][version - 1] # v1, v2
    df.to_csv(f'{BASE}/spcv{version}.csv', index=False)

    # test
    df = pd.read_csv(f'{BASE}/spcv{version}.csv').set_index('file_name')
    assert len(df) == [64721, 105829][version - 1] # v1, v2
    print(f'Created spcv{version}.csv - test passed')


def spcv1(root):
    convert_spc_metadata(root, version=1)


def spcv2(root):
    convert_spc_metadata(root, version=2)


def vc1():
    contents = requests.get('https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt')
    texts = contents.text.splitlines()
    num_files = [text.split(' ') for text in texts]
    splits = [{'1': 'train', '2': 'valid', '3': 'test'}[nf[0]] for nf in num_files]
    files = ['wav/'+nf[1].strip() for nf in num_files]
    labels = [f.split('/')[1] for f in files]
    df = pd.DataFrame({'file_name': files, 'label': labels, 'split': splits})
    df.to_csv(f'{BASE}/vc1.csv', index=None)

    df = pd.read_csv(f'{BASE}/vc1.csv')
    assert len(df) == 153516, f'153516 != {len(df)}'
    assert len(df.label.unique()) == 1251, f'# of labels is not correct.'
    print(f'{BASE}/vc1.csv - test passed')


def surge(root):
    files = sorted(Path(root).glob('*/*.ogg'))
    tone_split_map = {tone: split for tone, split in pd.read_csv('evar/predefined/dataset_surge_tone_splits.csv').values}
    df = pd.DataFrame({'file_name': [str(f).replace(root+'/', '') for f in files],
                    'label': [f.stem for f in files],
                    'split': [tone_split_map[f.parent.name] for f in files]})
    df.to_csv(f'{BASE}/surge.csv')
    print(f'Created surge.csv, # of samples in train/valid/test splits are:')
    print(f' {sum(df.split == "train")} {sum(df.split == "valid")} {sum(df.split == "test")}')


def __making_dataset_surge_tone_splits(root):
    tones = sorted([d.name for d in Path(root).glob('*')])
    N_tones = len(tones)
    test_tone_indexes = np.random.randint(0, N_tones, size=N_tones // 10) # 10%
    rest_indexes = [i for i in range(N_tones) if i not in test_tone_indexes]
    valid_tone_indexes = np.random.choice(rest_indexes, size=N_tones // 10) # 10%
    train_tone_indexes = [i for i in rest_indexes if i not in valid_tone_indexes]
    print(len(train_tone_indexes), len(valid_tone_indexes), len(test_tone_indexes))
    df = pd.DataFrame({'tone': tones})
    df.loc[train_tone_indexes, 'split'] = 'train'
    df.loc[valid_tone_indexes, 'split'] = 'valid'
    df.loc[test_tone_indexes, 'split'] = 'test'
    df.to_csv('evar/predefined/dataset_surge_tone_splits.csv', index=False)
    # test -> fine, nothing printed.
    for i in range(N_tones):
        if i in train_tone_indexes: continue
        if i in valid_tone_indexes: continue
        if i in test_tone_indexes: continue
        print(i, 'missing')


def __making_voxforge_metadata(url_folders):
    ## CAUTION: following will not work, leaving here for providing the detail.
    N = len(url_folders)
    folders_train = list(np.random.choice(url_folders, size=int(N * 0.7), replace=False))
    rest = [folder for folder in url_folders if folder not in folders_train]
    folders_valid = list(np.random.choice(rest, size=int(N * 0.15), replace=False))
    folders_test = [folder for folder in rest if folder not in folders_valid]

    Ltrn, Lval, Ltest = len(folders_train), len(folders_valid), len(folders_test)
    print(Ltrn, Lval, Ltest, Ltrn + Lval + Ltest)
    # 9685 2075 2077 13837

    for folder in file_folders:
        if folder in folders_train:
            split = 'train'
        elif folder in folders_valid:
            split = 'valid'
        elif folder in folders_test:
            split = 'test'
        else:
            assert False
        splits.append(split)

    ns = np.array(splits)
    Ltrn, Lval, Ltest, L = sum(ns == 'train'), sum(ns == 'valid'), sum(ns == 'test'), len(ns)
    print(f'Train:valid:test = {Ltrn/L:.2f}:{Lval/L:.2f}:{Ltest/L:.2f}, total={Ltrn + Lval + Ltest}')
    # Train:valid:test = 0.69:0.15:0.16, total=176428


def __making_cremad_metadata(not_working_just_a_note):
    ## CAUTION: following will not work, leaving here for providing the detail.
    TFDS_URL = 'https://storage.googleapis.com/tfds-data/manual_checksums/crema_d.txt'

    contents = requests.get(TFDS_URL)
    urls = [line.strip().split()[0] for line in contents.text.splitlines()]
    urls = [url for url in urls if url[-4:] == '.wav'] # wav only, excluding summaryTable.csv

    filenames = [url.split('/')[-1] for url in urls]
    speaker_ids = [file_name.split('_')[0] for file_name in filenames]
    labels = [file_name.split('_')[2] for file_name in filenames]

    print(len(filenames))
    # 7438

    uniq_speakers = list(set(speaker_ids))
    N = len(uniq_speakers)
    speakers_train = list(np.random.choice(uniq_speakers, size=int(N * 0.7), replace=False))
    rest = [sp for sp in uniq_speakers if sp not in speakers_train]
    speakers_valid = list(np.random.choice(rest, size=int(N * 0.1), replace=False))
    speakers_test = [sp for sp in rest if sp not in speakers_valid]

    Ltrn, Lval, Ltest = len(speakers_train), len(speakers_valid), len(speakers_test)
    print(Ltrn, Lval, Ltest, Ltrn + Lval + Ltest)
    # 63 9 19 91

    splits = []
    for sp in speaker_ids:
        if sp in speakers_train:
            split = 'train'
        elif sp in speakers_valid:
            split = 'valid'
        elif sp in speakers_test:
            split = 'test'
        else:
            assert False
        splits.append(split)

    ns = np.array(splits)
    Ltrn, Lval, Ltest, L = sum(ns == 'train'), sum(ns == 'valid'), sum(ns == 'test'), len(ns)
    print(f'Train:valid:test = {Ltrn/L:.2f}:{Lval/L:.2f}:{Ltest/L:.2f}, total={Ltrn + Lval + Ltest}')
    # Train:valid:test = 0.69:0.10:0.21, total=7438


if __name__ == "__main__":
    Path(BASE).mkdir(parents=True, exist_ok=True)
    fire.Fire()
