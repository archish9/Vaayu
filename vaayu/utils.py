import gc
import json
import os
import ctypes
import shutil
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time

from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .persist import ModelPersister


try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
except ImportError:
    bitsandbytes_installed = False


import huggingface_hub


# replacement for bnb quantstat.as_dict(True), until the bug is fixed....
def save_quant_state_to_dict(self, packed=True):
    """
    returns dict of tensors and strings to use in serialization via _save_to_state_dict()
    param: packed -- returns dict[str, torch.Tensor] for state_dict
    """
    qs_dict = {
        'quant_type': self.quant_type,
        'absmax': self.absmax,
        'blocksize': self.blocksize,
        'quant_map': self.code,
        'dtype': str(self.dtype).strip('torch.'),
        'shape': tuple(self.shape),
    }
    if self.nested:
        qs_dict.update({
            'nested_absmax': self.state2.absmax,
            'nested_blocksize': self.state2.blocksize,
            'nested_quant_map': self.state2.code,
            'nested_dtype': str(self.state2.dtype).strip('torch.'),
            'nested_offset': self.offset.item(),
        })
    if not packed:
        return qs_dict

    qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = bnb.utils.pack_dict_to_tensor(non_tensor_dict)
    return qs_packed_dict



class NotEnoughSpaceException(Exception):
    pass

# Function to clean RAM & vRAM
def clean_memory(cuda: bool = True):
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as ex:
        # maybe platform
        pass
    # cuda=False: host-only hygiene. empty_cache() returns blocks to the driver,
    # forcing a cudaMalloc storm on the next token's expert H2D — skip it on the
    # steady-state decode path (callers pass cuda=False there).
    if cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()


def uncompress_layer_state_dict(layer_state_dict):
    uncompressed_layer_state_dict = None
    if any(['4bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '4bit' not in k:
                quant_state_dict = {kk[len(k):]: kv for kk, kv in layer_state_dict.items() if kk.startswith(k) and k != kk}
                quant_state = bnb.functional.QuantState.from_dict(qs_dict=quant_state_dict, device="cuda")

                dqv = bnb.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict
    elif any(['8bit' in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if '8bit' not in k:

                absmax = layer_state_dict[k + ".8bit.absmax"]
                code = layer_state_dict[k + ".8bit.code"]

                dqv = bnb.functional.dequantize_blockwise(v.cuda(),
                                                          bnb.functional.QuantState(absmax=absmax.cuda(),
                                                                                    code=code.cuda(),
                                                                                    blocksize=2048,
                                                                                    dtype=torch.float16))
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict

    return layer_state_dict if uncompressed_layer_state_dict is None else uncompressed_layer_state_dict

def load_layer(local_path, layer_name, profiling=False):
    #layer_state_dict = load_file(Path(local_path) / (layer_name + ".safetensors"), device="cpu")
    layer_state_dict = ModelPersister.get_model_persister().load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict)

    #clean_memory()

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    else:
        return to_return



def check_space(checkpoint_path, layer_shards_saving_path=None, compression=None, splitted_model_dir_name='splitted_model'):
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    total, used, free = shutil.disk_usage(checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(f"Not enough space. Free space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path}:"  \
                                      f" {free / 1024 / 1024 / 1024:.02f}GB. Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB. " \
                                      f"existing space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path} assuming can reuse: {total_saved_split_files_size_bytes/ 1024 / 1024 / 1024:.02f}GB. "
                                      )

def compress_layer_state_dict(layer_state_dict, compression=None):
    compressed_layer_state_dict = None
    if compression == '4bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for quant_state_k, quant_state_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[k + ".4bit." + quant_state_k] = quant_state_v
    elif compression == '8bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            absmax = quant_state.absmax.clone().contiguous()
            code = quant_state.code.clone().contiguous()
            compressed_layer_state_dict[k] = v_quant
            compressed_layer_state_dict[k + ".8bit.absmax"] = absmax
            compressed_layer_state_dict[k + ".8bit.code"] = code

    return compressed_layer_state_dict if compressed_layer_state_dict is not None else layer_state_dict

def remove_real_and_linked_file(to_delete):
    # os.fspath + abspath before comparing: realpath() returns a str, and every
    # caller passes a Path, so `realpath(p) != p` used to be True even for a
    # plain file -- the same path was then unlinked twice and the second unlink
    # raised ENOENT AFTER the delete had already succeeded.
    to_delete = os.path.abspath(os.fspath(to_delete))
    realpath = os.path.realpath(to_delete)
    targetpath = realpath if realpath != to_delete else None

    os.remove(to_delete)
    if targetpath:
        os.remove(targetpath)



def delete_original_shards(checkpoint_path, index, safetensors_format):
    """Delete the source checkpoint shards a completed split has replaced.

    Only the weight files named by `index` are removed -- config, tokenizer and
    the index itself stay, since every subsequent load still reads them. Missing
    files are skipped (already reclaimed by an earlier run).

    Callers MUST have verified the split is complete first: this is
    irreversible, and `remove_real_and_linked_file` follows symlinks, so a shard
    linked into the shared HF cache takes the cached blob with it.
    """
    checkpoint_path = Path(checkpoint_path)
    freed = 0
    for name in sorted(set(index.values())):
        f = checkpoint_path / name
        if not f.exists():
            continue
        try:
            size = os.path.getsize(f)
            remove_real_and_linked_file(f)
            freed += size
            print(f"deleting original file: {f}")
        except OSError as e:
            print(f"could not delete {f}: {e}")
    if freed:
        print(f"delete_original: reclaimed {freed / 1024 ** 3:.1f} GB from "
              f"{checkpoint_path} (the split copy is now the only copy)")
    return freed


def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """

    if compression is not None:
        assert bitsandbytes_installed, f"when using compression bitsandbytes has to be installed."
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)


    saving_path = checkpoint_path / splitted_model_dir_name

    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name


    safetensors_format = False
    index = None
    
    # Check for sharded models first
    if os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json'):
        with open(checkpoint_path / 'pytorch_model.bin.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    elif os.path.exists(checkpoint_path / 'model.safetensors.index.json'):
        safetensors_format = True
        with open(checkpoint_path / 'model.safetensors.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    # Check for single file models
    elif os.path.exists(checkpoint_path / 'model.safetensors'):
        print("Found single file model: model.safetensors")
        safetensors_format = True
        # Load the single file to create an index
        state_dict = load_file(checkpoint_path / 'model.safetensors', device='cpu')
        index = {k: 'model.safetensors' for k in state_dict.keys()}
    elif os.path.exists(checkpoint_path / 'pytorch_model.bin'):
        print("Found single file model: pytorch_model.bin")
        # Load the single file to create an index
        state_dict = torch.load(checkpoint_path / 'pytorch_model.bin', map_location='cpu')
        index = {k: 'pytorch_model.bin' for k in state_dict.keys()}
    else:
        raise FileNotFoundError(f"No model files found in {checkpoint_path}. Expected one of: "
                              f"pytorch_model.bin.index.json, model.safetensors.index.json, "
                              f"model.safetensors, or pytorch_model.bin")

    if layer_names is None:
        prefix = 'model.layers'
    else:
        prefix = layer_names['layer_prefix']

    # Safely parse layer indices regardless of where the prefix appears in the key
    n_layers = len(set([int(k.split(prefix + '.')[1].split('.')[0]) for k in index.keys() if prefix + '.' in k]))
    if n_layers == 0:
        raise ValueError(
            f"No decoder layers found under prefix '{prefix}' in model index. "
            f"Check layer_names['layer_prefix'] matches the model's actual weight key structure."
        )

    if layer_names is None:
        layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    else:
        layers = [layer_names['embed']] + [f'{layer_names["layer_prefix"]}.{i}' for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]

        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        layers = [l + "." for l in layers]

    # A tied-weight checkpoint has no lm_head of its own (D4). Stop expecting
    # one BEFORE the completeness check below, so a split that legitimately has
    # no head still verifies -- and so the write loop below does not mint an
    # empty shard for it.
    _lm_head = 'lm_head.' if layer_names is None else layer_names['lm_head'] + '.'
    _with_head = layers
    layers = drop_absent_lm_head(layers, index, _lm_head)
    if layers is not _with_head:
        print(f"no {_lm_head.rstrip('.')} weight in the checkpoint index "
              f"(tied embeddings) -- not expecting a shard for it")

    # check if splitting exists and all files are there
    found_layers = None
    #print(f"checking exists: {saving_path}")
    if os.path.exists(saving_path):
        # dir already exists, check if all layer files are there

        found_layers = {}
        for layer in layers:
            found_layers[layer] = ModelPersister.get_model_persister().model_persist_exist(layer.rstrip('.'), saving_path)

        print(f"found_layers:{found_layers}")
        if all(found_layers.values()):
            # already downloaded, return saving path...
            print(f"saved layers already found in {saving_path}")
            if delete_original:
                # The mid-split deletion below never runs for an already-split
                # model, so without this the originals sit next to the split
                # copy forever -- the exact duplication the flag exists to
                # avoid (measured 60-66 GB per model on the dev box).
                # Gated on the split being VERIFIED COMPLETE: every expected
                # layer has both its .safetensors and its .done marker, which
                # is what all(found_layers.values()) above means. IRREVERSIBLE
                # -- recovery is a re-download.
                delete_original_shards(checkpoint_path, index, safetensors_format)
            return str(saving_path)
        else:
            print(f"some layer splits found, some are not, re-save all layers in case there's some corruptions.")

    # Checked in BOTH modes. This used to be skipped when delete_original was on,
    # because the split freed shards as it went and so never needed 2x the model
    # on disk. It no longer deletes mid-split (see the end of this function), so
    # the pre-flight is what it always should have been: the mode that can lose
    # data is not the mode that gets no free-space check.
    check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_model_dir_name=splitted_model_dir_name)


    shard = -1
    n_shards = len(set(index.values()))
    state_dict = {}
    single_file_loaded = False

    # For single file models, check if we already loaded the state_dict
    if n_shards == 1 and list(set(index.values()))[0] in ['model.safetensors', 'pytorch_model.bin']:
        single_file_name = list(set(index.values()))[0]
        if os.path.exists(checkpoint_path / single_file_name):
            print(f"Loading single model file: {single_file_name}")
            if single_file_name.endswith('.safetensors'):
                state_dict = load_file(checkpoint_path / single_file_name, device='cpu')
            else:
                state_dict = torch.load(checkpoint_path / single_file_name, map_location='cpu')
            single_file_loaded = True


    if not os.path.exists(saving_path):
        #os.makedirs(saving_path)
        saving_path.mkdir(parents=True, exist_ok=True)


    for layer in tqdm(layers):

        # Skip loading if we already have the full state_dict from a single file
        if not single_file_loaded:
            # Optionnally load next shard
            # checking whether after spliting from '-', if second element exists. otherwise it throws errors for single 'model.safetensor' files
            shards = [int(v.split('-')[1]) for k, v in index.items() if layer in k and '-' in v and len(v.split('-')) > 1]
            if len(shards) > 0:
                target_shard = max(shards)
                if target_shard != shard:
                    # No deletion here. Deleting shard N-1 on the way to shard N
                    # destroyed sources while the split was still unfinished: a
                    # crash at shard 5 of 9 left 1-4 gone AND the split unusable,
                    # and it never reclaimed the last shard, which nothing
                    # advances past. Deletion now happens once, at the end, gated
                    # on the same completeness check the already-split path uses.
                    shard = target_shard
                    print(f'Loading shard {shard}/{n_shards}')

                    # Dynamically find the real filename from the index map instead of rigidly string formatting
                    matching_files = [val for val in set(index.values()) if '-' in val and len(val.split('-')) > 1 and int(val.split('-')[1]) == shard]
                    
                    if len(matching_files) > 0:
                        to_load = checkpoint_path / matching_files[0]
                    else:
                        # Defensive fallback
                        if not safetensors_format:
                            to_load = checkpoint_path / f'pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin'
                        else:
                            to_load = checkpoint_path / f'model-000{shard:02d}-of-000{n_shards:02d}.safetensors'

                    # check if to_load exist, if not downloaad it...
                    if not os.path.exists(to_load):
                        assert repo_id is not None
                        huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                        token=hf_token)

                    if not safetensors_format:
                        state_dict.update(torch.load(to_load, map_location='cpu'))
                    else:
                        state_dict.update(load_file(to_load, device='cpu'))

            else:
                shards = [v for k, v in index.items() if layer in k]
                if len(shards) == 0:
                    # No weights for this layer in index, skip it
                    continue
                to_load = checkpoint_path / shards[0]
                # check if to_load exist, if not downloaad it...
                if not os.path.exists(to_load):
                    assert repo_id is not None
                    huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                    token=hf_token)
                if not safetensors_format:
                    state_dict.update(torch.load(to_load, map_location='cpu'))
                else:
                    state_dict.update(load_file(to_load, device='cpu'))

        # Get layer state dict and strip unknown architectural prefixes to standardize map for Vaayu Base
        layer_state_dict = {}
        for k, v in state_dict.items():
            if layer in k:
                # Strip everything preceding the canonical layer ID string map
                idx = k.find(layer)
                normalized_k = k[idx:]
                layer_state_dict[normalized_k] = v

        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

        # Save layer state dict as using safetensors

        marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer.rstrip('.'), saving_path)
        if not marker_exists:
            ModelPersister.get_model_persister().persist_model(layer_state_dict, layer.rstrip('.'), saving_path)

        # Free memory
        for k in layer_state_dict.keys():
            if k in state_dict:
                del state_dict[k]
        del layer_state_dict
        clean_memory()

    # The ONE place a fresh split deletes anything, and it runs after every layer
    # is written. Gated on the same completeness check the already-split path at
    # the top of this function uses: every expected layer has both its
    # .safetensors and its .done marker. Covers sharded and single-file sources
    # alike -- `index` names the source file for every weight either way.
    # IRREVERSIBLE, and remove_real_and_linked_file follows symlinks, so a shard
    # linked into the shared HF cache takes the cached blob with it.
    if delete_original:
        persister = ModelPersister.get_model_persister()
        missing = [l for l in layers
                   if not persister.model_persist_exist(l.rstrip('.'), saving_path)]
        if missing:
            # A real limit, not something to hide (CLAUDE.md §5): the disk stays
            # full, and the next load re-splits from sources that still exist.
            print(f"WARNING: keeping the original shards -- the split under "
                  f"{saving_path} is INCOMPLETE, missing: {', '.join(missing)}")
        else:
            delete_original_shards(checkpoint_path, index, safetensors_format)

    return str(saving_path)

def drop_absent_lm_head(layers, index, lm_head):
    """Remove the lm_head entry from `layers` when the index has no key for it.

    D4: a checkpoint with `tie_word_embeddings: true` stores no lm_head weight,
    but `layers` listed one unconditionally. The splitter then either wrote a
    valid EMPTY shard plus its `.done` marker -- so `all(found_layers.values())`
    called the split COMPLETE when it was not -- or, on sharded sources, skipped
    the layer and left found_layers[lm_head] False forever, re-splitting the
    whole model on every load.

    Decided from the INDEX, not from the config's tie_word_embeddings flag:
    detection over assumption (CLAUDE.md §5), and it covers any reason the
    weight is absent rather than the one reason we thought of. The consumer side
    already handles a missing head -- an untied checkpoint pins a real lm_head,
    a tied one falls back to the embedding.

    ONLY the head is ever dropped. A decoder layer missing from the index is a
    real problem and must stay loud.
    """
    if any(lm_head in key for key in index):
        return layers
    return [layer for layer in layers if layer != lm_head]


def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None, compression=None,
                                       layer_names=None, hf_token=None, delete_original=False):
    """
    find the model's local cache path, download the cache if not exists, then split and save the model.

    Parameters
    ----------
    model_local_path_or_repo_id : str
        model local path or hf repo id
    layer_shards_saving_path : str, optional
        optional path to save the splitted model, by default directly under the model local path

    Returns
    -------
    model_local_path : str
        local model path
    saved_layer_shards_path : str
        the path saved layer shards
    compression: str, optinal
        setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
    hf_token: str, optional
        huggingface api token could be provided, by default None
    """

    # try local model path, if the model exist split and save there
    if os.path.exists(model_local_path_or_repo_id):
        if os.path.exists(Path(model_local_path_or_repo_id) / 'pytorch_model.bin.index.json') or \
           os.path.exists(Path(model_local_path_or_repo_id) / 'model.safetensors.index.json') or \
           os.path.exists(Path(model_local_path_or_repo_id) / 'model.safetensors') or \
           os.path.exists(Path(model_local_path_or_repo_id) / 'pytorch_model.bin'):
            print(f"found index file...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(model_local_path_or_repo_id, layer_shards_saving_path,
                                                                            compression=compression, layer_names=layer_names, delete_original=delete_original)
        else:
            print(
                f"Found local directory in {model_local_path_or_repo_id}, but didn't find downloaded model. Try using {model_local_path_or_repo_id} as a HF repo...")

    # it should be a repo id at this point...
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
        #allow_patterns= ["model.safetensors.index.json", 'pytorch_model.bin.index.json'],
        ignore_patterns=['*.safetensors', '*.bin'])
        
    if not (os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json') or 
            os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json')):
        print(f"No index file found in cache, possibly a single file model. Attempting to download single file model weights...")
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
             allow_patterns=['*.safetensors', '*.bin'])


    # check if there's safetensors saved, if so, exclude torch saves
    # delay download now...
    '''
    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token, allow_patterns="model.safetensors.index.json")
    if len(glob(str(Path(hf_cache_path) / "model.safetensors.index.json"))) > 0:
        # there's safe tensor version, exclude torch version
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
                                                          ignore_patterns=['pytorch_model.bin.index.json', '*.bin'])

    else:
        hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id,
                                                          token=hf_token)
    '''

    #assert os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json') or \
    #       os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json'), \
    #       f"{hf_cache_path}/pytorch_model.bin.index.json or {hf_cache_path}/model.safetensors.index.json should exists."

    # if splitted_model subdir exists under cache use it, otherwise split and save
    return Path(hf_cache_path), split_and_save_layers(hf_cache_path, layer_shards_saving_path,
                                                      compression=compression, layer_names=layer_names,
                                                      delete_original=delete_original, repo_id=model_local_path_or_repo_id, hf_token=hf_token)
