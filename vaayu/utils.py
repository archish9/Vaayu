import gc
import contextlib
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
from safetensors import safe_open
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



def freeze_after_load():
    """Move the loaded model's object graph out of the GC's reach.

    A loaded model is a large, PERMANENT graph -- module tree, expert-cache
    views, pinned-slab bookkeeping -- that every gen-2 collection re-traverses
    for objects that will never be freed. `gc.freeze()` moves everything that
    already exists into a permanent generation the collector skips.

    Collect first: whatever is frozen is retained for the life of the process,
    so freezing over load-time garbage leaks it permanently. For the same
    reason this belongs to a process that loads one model and then serves --
    call it again after a second load and the first model's garbage is pinned.
    """
    gc.collect()
    gc.freeze()
    print(f"[Vaayu] gc.freeze(): {gc.get_freeze_count()} objects moved out of "
          f"the collector's reach")


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



def _leave_an_index_behind(checkpoint_path, index, safetensors_format):
    """Synthesise the index a SINGLE-FILE checkpoint never had, before deleting.

    A sharded checkpoint survives its own delete because
    `model.safetensors.index.json` is not a weight file: it stays, and
    `find_or_create_local_splitted_path` recognises the directory by it. A
    single-file checkpoint has no index -- the `model.safetensors` about to be
    deleted is the ONLY one of the four names that discovery looks for -- so the
    next load called the path undiscoverable, treated it as a HF repo id and
    died, with a complete `splitted_model/` sitting right there (B14).

    Writing the map the splitter already built puts the single-file case on the
    sharded case's proven path: discovery finds the index, `split_and_save_layers`
    reads its weight_map, `all(found_layers.values())` holds, and the split is
    returned. Only config/tokenizer/generation-config are ever read from this
    directory afterwards -- no weight loader touches it -- so an index naming a
    file that is about to go is exactly as true as the sharded one.

    A checkpoint that already has an index keeps it untouched. Returns whether
    the delete may proceed.
    """
    if any((checkpoint_path / n).exists() for n in
           ('model.safetensors.index.json', 'pytorch_model.bin.index.json')):
        return True
    name = 'model.safetensors.index.json' if safetensors_format else 'pytorch_model.bin.index.json'
    try:
        (checkpoint_path / name).write_text(json.dumps({"weight_map": index}))
    except OSError as e:
        # A real limit, not something to hide (CLAUDE.md §5): without the index
        # the delete strands the checkpoint, so give up the reclaim instead --
        # the split is already written and usable, and the disk stays full.
        print(f"WARNING: keeping the original files -- could not write {name} ({e}), "
              f"and deleting the only weight file would leave {checkpoint_path} undiscoverable")
        return False
    print(f"delete_original: wrote {name} so this checkpoint stays discoverable")
    return True


def delete_original_shards(checkpoint_path, index, safetensors_format):
    """Delete the source checkpoint shards a completed split has replaced.

    Only the weight files named by `index` are removed -- config, tokenizer and
    the index stay, since every subsequent load still reads them. A single-file
    checkpoint has no index of its own, so one is synthesised first
    (`_leave_an_index_behind`); without it the delete leaves the directory
    undiscoverable. Missing files are skipped (already reclaimed by an earlier
    run).

    Callers MUST have verified the split is complete first: this is
    irreversible, and `remove_real_and_linked_file` follows symlinks, so a shard
    linked into the shared HF cache takes the cached blob with it.
    """
    checkpoint_path = Path(checkpoint_path)
    if not _leave_an_index_behind(checkpoint_path, index, safetensors_format):
        return 0
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
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None,
                          key_remap=None):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """

    if compression is not None:
        assert bitsandbytes_installed, f"when using compression bitsandbytes has to be installed."
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)

    # G0: refuse an incomplete checkpoint BEFORE splitting it. This is the only
    # seam that sees the original shard set -- the splitted output carries neither
    # -of-NNNNN names nor an index, so a guard downstream of here sees nothing.
    from .persist.checkpoint_integrity import assert_checkpoint_complete
    assert_checkpoint_complete(str(checkpoint_path))

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

    # G1: ask the same layout question ExpertOffsetIndex asks, but on the
    # ORIGINAL keys, before anything is written or (under delete_original)
    # deleted. The equivalent check in ExpertOffsetIndex.build() runs only
    # after the split -- and after delete_original_shards() under the
    # server's DELETE_ORIGINAL default -- so a checkpoint Vaayu cannot index
    # would otherwise cost the user their only copy before anything said so.
    from .persist.checkpoint_integrity import assert_expert_layout_indexable
    assert_expert_layout_indexable(str(checkpoint_path), index.keys())

    # A checkpoint whose keys do not already name the module tree gets ONE pass
    # of renaming here, before anything below reads a key: `n_layers`, the
    # `layers` list, `keys_by_layer` and the written tensor names all follow
    # from it. `_fetch` keeps reading by the ORIGINAL key, so a remap only ever
    # has to describe the naming -- never the file layout.
    #
    # Deliberately AFTER G1, which asks its layout question of the original
    # keys, and deliberately before everything else, because the splitter
    # matches keys by SUBSTRING: DeepSeek-V4's own names would put 248
    # layer-internal `*norm` keys in the `norm` shard.
    #
    # A key the callable maps to None is DROPPED (DeepSeek-V4's `mtp.*` draft
    # module: 10.117 GiB, and 768 experts that `detect_moe_blocks` would read
    # as a second layer 0). `original_index` survives for the delete path,
    # which names FILES and must keep naming the checkpoint's own keys.
    original_index = index
    source_key = {}
    if key_remap is not None:
        remapped = {}
        for orig_key, fname in index.items():
            new_key = key_remap(orig_key)
            if new_key is None:
                continue
            if new_key in remapped:
                raise ValueError(
                    f"key_remap maps two checkpoint keys to {new_key!r} "
                    f"({source_key[new_key]!r} and {orig_key!r}). The split "
                    f"would write one and silently lose the other.")
            remapped[new_key] = fname
            source_key[new_key] = orig_key
        print(f"key_remap: {len(remapped)} tensors renamed, "
              f"{len(index) - len(remapped)} dropped")
        index = remapped

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
                delete_original_shards(checkpoint_path, original_index, safetensors_format)
            return str(saving_path)
        else:
            print(f"some layer splits found, some are not, re-save all layers in case there's some corruptions.")

    # Checked in BOTH modes. This used to be skipped when delete_original was on,
    # because the split freed shards as it went and so never needed 2x the model
    # on disk. It no longer deletes mid-split (see the end of this function), so
    # the pre-flight is what it always should have been: the mode that can lose
    # data is not the mode that gets no free-space check.
    check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_model_dir_name=splitted_model_dir_name)


    # A layer's tensors may live in ANY set of shards, in ANY order. This used to
    # load, per layer, only `max(shard number)`, and only when that number differed
    # from the shard already held -- which assumes the index is layer-major and
    # monotonic, so that every lower-numbered shard was pulled in by an earlier
    # layer. Modern quantized MoE checkpoints break that: one layer spreads over
    # several NON-ADJACENT shards and the numbers do not rise with the layer index.
    # The layer was then written anyway, minus the tensors from the shards that were
    # never loaded, with its .done marker stamped beside it -- so the completeness
    # check above called the split good. Measured on a 48-layer / 512-expert int4
    # checkpoint: 24356 of 24576 experts, two layers short, and no error anywhere.
    # See tests/test_split_shard_spanning.py.
    #
    # Group the index by layer once, then read each layer's keys out of whichever
    # files hold them. safetensors is read ONE TENSOR AT A TIME, so the resident set
    # is a single layer rather than a growing pile of whole shards.
    keys_by_layer = {layer: [] for layer in layers}
    for k, fname in index.items():
        for layer in layers:
            if layer in k:
                keys_by_layer[layer].append((k, fname))

    shard_readers = contextlib.ExitStack()
    open_shards = {}          # safetensors: fname -> safe_open handle
    bin_state_dicts = {}      # .bin: fname -> the whole loaded state dict

    def _fetch(key, fname):
        key = source_key.get(key, key)
        path = checkpoint_path / fname
        if not os.path.exists(path):
            assert repo_id is not None
            huggingface_hub.snapshot_download(repo_id, allow_patterns=fname, token=hf_token)
        if safetensors_format:
            if fname not in open_shards:
                print(f'Opening shard {fname}')
                open_shards[fname] = shard_readers.enter_context(
                    safe_open(path, framework='pt', device='cpu'))
            return open_shards[fname].get_tensor(key)
        if fname not in bin_state_dicts:
            print(f'Loading shard {fname}')
            bin_state_dicts[fname] = torch.load(path, map_location='cpu')
        return bin_state_dicts[fname][key]

    if not os.path.exists(saving_path):
        #os.makedirs(saving_path)
        saving_path.mkdir(parents=True, exist_ok=True)


    for layer in tqdm(layers):

        # Get layer state dict and strip unknown architectural prefixes to standardize map for Vaayu Base
        layer_state_dict = {}
        for k, fname in keys_by_layer[layer]:
            # Strip everything preceding the canonical layer ID string map
            idx = k.find(layer)
            layer_state_dict[k[idx:]] = _fetch(k, fname)

        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

        # Save layer state dict as using safetensors

        marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer.rstrip('.'), saving_path)
        if not marker_exists:
            ModelPersister.get_model_persister().persist_model(layer_state_dict, layer.rstrip('.'), saving_path)

        # Free memory
        for k, fname in keys_by_layer[layer]:
            if fname in bin_state_dicts:
                bin_state_dicts[fname].pop(source_key.get(k, k), None)
        del layer_state_dict
        clean_memory()

    shard_readers.close()

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
            delete_original_shards(checkpoint_path, original_index, safetensors_format)

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
                                       layer_names=None, hf_token=None, delete_original=False, key_remap=None):
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
                                                                            compression=compression, layer_names=layer_names,
                                                                            delete_original=delete_original, key_remap=key_remap)
        else:
            # G0: a directory holding shards but no index is an interrupted
            # download (the 46-of-48-with-no-index case G0 was written for),
            # not a genuine repo id -- raise G0's message here rather than
            # falling through to snapshot_download(<this local path>), which
            # raises an opaque HFValidationError instead. A directory with
            # zero *.safetensors returns silently, so a real repo id is
            # unaffected.
            from .persist.checkpoint_integrity import assert_checkpoint_complete
            assert_checkpoint_complete(str(model_local_path_or_repo_id))
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
                                                      delete_original=delete_original, repo_id=model_local_path_or_repo_id,
                                                      hf_token=hf_token, key_remap=key_remap)
