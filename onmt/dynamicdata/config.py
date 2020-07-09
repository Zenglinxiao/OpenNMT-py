import collections
import itertools
import os
import yaml

from .utils import external_linecount


def _group_input_by_task_(data_config):
    """In-place group all input to its task as `_inputs`."""
    for input in data_config['inputs']:
        task = data_config['inputs'][input]['task']
        if '_inputs' not in data_config['tasks'][task]:
            data_config['tasks'][task]['_inputs'] = []
        data_config['tasks'][task]['_inputs'].append(input)
    for task in data_config['tasks']:
        if '_inputs' in data_config['tasks'][task]:
            data_config['tasks'][task]['_inputs'].sort()


def _share_inputs(data_config):
    for task in data_config['tasks']:
        if 'share_inputs' in data_config['tasks'][task]:
            share_from = data_config['tasks'][task]['share_inputs']
            inputs = data_config['tasks'][share_from]['_inputs']
            data_config['tasks'][task]['_inputs'] = inputs


def _remove_shared_inputs(shard_config):
    to_remove = []
    for task in shard_config['tasks']:
        if 'share_inputs' in shard_config['tasks'][task]:
            to_remove.append(task)
    for task in to_remove:
        del shard_config['tasks'][task]


def _an_input(d, name):
    candidates = ('src', 'mono')
    for candidate in candidates:
        if candidate in d:
            return d[candidate]
    raise Exception('Could not find an input for "{}"'.format(name))


def normalize_sizes(data_config):
    size_per_task = collections.Counter()
    for input in data_config['inputs']:
        task = data_config['inputs'][input]['task']
        size = data_config['inputs'][input].get('size', 'auto')
        if size == 'auto':
            an_input = _an_input(data_config['inputs'][input], input)
            size = external_linecount(an_input)
            data_config['inputs'][input]['size'] = size
        size_per_task[task] += size
    for input in data_config['inputs']:
        task = data_config['inputs'][input]['task']
        size = data_config['inputs'][input]['size']

        data_config['inputs'][input]['size'] = int(
            100 * size / size_per_task[task])
        data_config['tasks'][task]['_size'] = size_per_task[task]


def _get_all_transforms(data_config):
    """Return all transforms that specified in each task."""
    all_transforms = set()
    for task in data_config['tasks']:
        all_transforms.update(data_config['tasks'][task].get('transforms', []))
    return list(sorted(all_transforms))


def _task_defaults_(data_config):
    for task in data_config['tasks']:
        if 'split' not in data_config['tasks'][task]:
            data_config['tasks'][task]['split'] = 'train'
        if 'weight' not in data_config['tasks'][task]:
            data_config['tasks'][task]['weight'] = 1


def read_data_config(data_config_file):
    with open(data_config_file, 'r') as fobj:
        data_config = yaml.safe_load(fobj)
    return process_config(data_config)


def data_config_from_string(string):
    data_config = yaml.safe_load(string)
    return process_config(data_config)


def process_config(data_config, template=False):
    _group_input_by_task_(data_config)
    if not template:
        data_config['_transforms'] = _get_all_transforms(data_config)
    _task_defaults_(data_config)
    # validate_schema(data_config)
    return data_config


def _filter_config(config, sub_config, path, rules):
    decision = 'continue'
    current = path[-1]
    for rule in rules:
        rule_path, rule_decision = rule
        if all(b is None or a == b
               for (a, b)
               in itertools.zip_longest(path, rule_path)):
            decision = rule_decision
            break
    if decision == 'keep':
        sub_config[current] = config[current]
    elif decision == 'drop':
        return
    else:
        if isinstance(config[current], dict):
            sub_config[current] = {}
            for key in config[current]:
                extended_path = path + [key]
                _filter_config(
                    config[current], sub_config[current], extended_path, rules)
        else:
            # default keep
            sub_config[current] = config[current]


def sharding_only(data_config):
    """ retains only config used in sharding step """
    rules = (
             (('meta', 'shard'), 'keep'),
             (('meta', 'train'), 'drop'),
             (('tasks', None, 'transforms'), 'drop'),
             (('tasks', None, 'weight'), 'drop'),
             (('tasks', None, 'meta'), 'drop'),
             (('inputs',), 'keep'),
             (('_transforms',), 'drop'),
            )
    sub_config = {}
    for key in data_config:
        _filter_config(data_config, sub_config, [key], rules)
    _remove_shared_inputs(sub_config)
    return sub_config


def remove_generated(data_config):
    """ removes generated keys from config """
    rules = (
             (('meta',), 'keep'),
             (('inputs',), 'keep'),
             (('tasks', None, '_inputs',), 'drop'),
             (('_transforms',), 'drop'),
            )
    sub_config = {}
    for key in data_config:
        _filter_config(data_config, sub_config, [key], rules)
    return sub_config


def save_shard_config(data_config):
    stored_shard_config_file = os.path.join(
        data_config['meta']['shard']['rootdir'],
        'stored_shard_config.yaml')
    if os.path.exists(stored_shard_config_file):
        raise Exception(
            'stored shard config "{}"'
            ' already exists, not overwriting'.format(
            stored_shard_config_file))
    os.makedirs(
        data_config['meta']['shard']['rootdir'],
        exist_ok=True)
    with open(stored_shard_config_file, 'w') as fobj:
        yaml.safe_dump(data_config, fobj)


def dict_diff(a, b):
    keys = list(sorted(set(a.keys()).union(b.keys())))
    a_pruned = {}
    b_pruned = {}
    for key in keys:
        if a.get(key, None) == b.get(key, None):
            # equal can be pruned
            continue
        if (key in a
                and key in b
                and isinstance(a[key], dict)
                and isinstance(b[key], dict)):
            a_sub, b_sub = dict_diff(a[key],  b[key])
            a_pruned[key] = a_sub
            b_pruned[key] = b_sub
        else:
            a_pruned[key] = a.get(key, '**** MISSING ****')
            b_pruned[key] = b.get(key, '**** MISSING ****')
    return a_pruned, b_pruned


def verify_shard_config(data_config):
    stored_shard_config_file = os.path.join(
        data_config['meta']['shard']['rootdir'],
        'stored_shard_config.yaml')
    shard_config = sharding_only(data_config)
    with open(stored_shard_config_file, 'r') as fobj:
        stored_shard_config = yaml.safe_load(fobj)
    for task in stored_shard_config['tasks']:
        if '_inputs' in stored_shard_config['tasks'][task]:
            stored_shard_config['tasks'][task]['_inputs'].sort()
        if '_size' in stored_shard_config['tasks'][task]:
            del stored_shard_config['tasks'][task]['_size']
    for inp in stored_shard_config['inputs']:
        if 'size' in stored_shard_config['inputs'][inp]:
            del stored_shard_config['inputs'][inp]['size']
    for inp in shard_config['inputs']:
        if 'size' in shard_config['inputs'][inp]:
            del shard_config['inputs'][inp]['size']
    if not shard_config == stored_shard_config:
        old, new = dict_diff(stored_shard_config, shard_config)
        raise Exception(
            'data_config not compatible with stored shard config.\n'
            'old {}\nnew {}\n'.format(old, new))
