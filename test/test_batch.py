import json
from argparse import ArgumentParser
from pathlib import PurePath
from unittest.mock import mock_open, patch

import pytest
import unified_path as up
from unified_path import PureUnifiedPath, UnifiedPath

from batch_processing import Batch


def _mock_glob(path, recursive=False):
    return [path]


@pytest.fixture
def parser():
    parser = ArgumentParser()
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--dir', type=UnifiedPath)
    parser.add_argument('--var')

    return parser


def test_argument_parsing_cli(parser):
    """Check argument parsing when arguments are given on the command line.
    """
    batch = Batch(['-i', 'input1', 'input2', '-o', 'output', '--alpha', '0.5'])
    args = batch.parse_args(parser)

    assert args.inputs == [PurePath('input1'), PurePath('input2')]
    assert args.output == PurePath('output')
    assert args.alpha == 0.5


def test_argument_parsing_cli_tags(parser):
    """Check argument parsing when arguments are given on the command line, and inputs are given with tags.
    """
    batch = Batch(['-i', '+tag1', 'input1', 'input2', '+tag2', 'input3', '+tag3', '-o', 'output', '--alpha', '0.5'])
    args = batch.parse_args(parser)

    assert args.inputs == {'tag1': [PurePath('input1'), PurePath('input2')], 'tag2': [PurePath('input3')], 'tag3': []}
    assert args.output == PurePath('output')
    assert args.alpha == 0.5


def test_argument_parsing_deserialization(parser):
    """Check argument parsing when arguments are given in one command line serialized file.
    """
    batch = Batch(['-a', '1.json'])

    content = json.dumps({'args': {'inputs': ['input1', 'input2'], 'output': 'output', 'alpha': 0.5}})
    with patch('builtins.open', mock_open(read_data=content)):
        args = batch.parse_args(parser)

    assert args.inputs == [up.absolute(PurePath('input1')), up.absolute(PurePath('input2'))]
    assert args.output == up.absolute(PurePath('output'))
    assert args.alpha == 0.5


def test_argument_parsing_deserialization_merge_1(parser):
    """Check argument parsing when arguments are given in two command line serialized files.
    """
    batch = Batch(['-a', 'dir1/1.json', 'dir2/2.json', '--alpha', '0.7'])

    content_1 = json.dumps({'args': {'inputs': ['input1', 'input2'], 'output': 'output', 'alpha': 0.5}})
    content_2 = json.dumps({'args': {'inputs': ['input3'], 'output': 'output2'}})

    mock = mock_open()
    mock.side_effect = [mock_open(read_data=content_1).return_value, mock_open(read_data=content_2).return_value]

    with patch('builtins.open', mock):
        args = batch.parse_args(parser)

    assert args.inputs == [
        up.absolute(PurePath('dir1/input1')),
        up.absolute(PurePath('dir1/input2')),
        up.absolute(PurePath('dir2/input3'))
    ]
    assert args.output == up.absolute(PurePath('dir2/output2'))
    assert args.alpha == 0.7


def test_argument_parsing_deserialization_merge_2(parser):
    """Check argument parsing when arguments are given in one command line serialized file, that itself refers to
    another serialized file.
    """
    batch = Batch(['-a', 'dir1/1.json', '--alpha', '0.7'])

    content_1 = json.dumps({'args': {'inputs': ['input3'], 'output': 'output2', 'arg_files': ['dir2/2.json']}})
    content_2 = json.dumps({'args': {'inputs': ['input1', 'input2'], 'output': 'output', 'alpha': 0.5}})

    mock = mock_open()
    mock.side_effect = [mock_open(read_data=content_1).return_value, mock_open(read_data=content_2).return_value]

    with patch('builtins.open', mock):
        args = batch.parse_args(parser)

    assert args.inputs == [
        up.absolute(PurePath('dir1/dir2/input1')),
        up.absolute(PurePath('dir1/dir2/input2')),
        up.absolute(PurePath('dir1/input3'))
    ]
    assert args.output == up.absolute(PurePath('dir1/output2'))
    assert args.alpha == 0.7


def test_argument_parsing_deserialization_merge_3(parser):
    """Check argument parsing when arguments are given with set_args().
    """
    batch = Batch()
    batch.set_args(alpha=0.7, arg_files=['dir1/1.json'])

    content_1 = json.dumps({'args': {'inputs': ['input3'], 'output': 'output2', 'arg_files': ['dir2/2.json']}})
    content_2 = json.dumps({'args': {'inputs': ['input1', 'input2'], 'output': 'output', 'alpha': 0.5}})
    content_3 = json.dumps({'args': {'inputs': ['input4', 'input5'], 'output': 'output', 'alpha': 0.5}})

    mock = mock_open()
    mock.side_effect = [
        mock_open(read_data=content_1).return_value,
        mock_open(read_data=content_2).return_value,
        mock_open(read_data=content_3).return_value
    ]

    with patch('builtins.open', mock):
        args = batch.parse_args(parser)

    assert args.inputs == [
        up.absolute(PurePath('dir1/dir2/input1')),
        up.absolute(PurePath('dir1/dir2/input2')),
        up.absolute(PurePath('dir1/input3'))
    ]
    assert args.output == up.absolute(PurePath('dir1/output2'))
    assert args.alpha == 0.7


def test_argument_parsing_deserialization_tags(parser):
    """Check argument parsing when arguments are given in a serialized file, and inputs are given with tags.
    """
    batch = Batch(['-a', '1.json'])

    content = json.dumps({
        'args': {
            'inputs': {
                'tag1': ['input1', 'input2'],
                'tag2': ['input3'],
                'tag3': []
            },
            'output': 'output',
            'alpha': 0.5
        }
    })

    with patch('builtins.open', mock_open(read_data=content)):
        args = batch.parse_args(parser)

    assert args.inputs == {
        'tag1': [up.absolute(PurePath('input1')), up.absolute(PurePath('input2'))],
        'tag2': [up.absolute(PurePath('input3'))],
        'tag3': []
    }
    assert args.output == up.absolute(PurePath('output'))
    assert args.alpha == 0.5


def test_argument_parsing_deserialization_groups(parser):
    """Check argument parsing when arguments are given in a serialized file, and inputs are given with groups.
    """
    batch = Batch(['-a', '1.json'])

    content = json.dumps(
        {'args': {
            'inputs': [['input1', 'input2'], ['input3'], []],
            'output': 'output',
            'alpha': 0.5
        }})

    with patch('builtins.open', mock_open(read_data=content)):
        args = batch.parse_args(parser)

    assert args.inputs == [[up.absolute(PurePath('input1')),
                            up.absolute(PurePath('input2'))], [up.absolute(PurePath('input3'))], []]
    assert args.output == up.absolute(PurePath('output'))
    assert args.alpha == 0.5


@patch('glob.glob', _mock_glob)
def test_collect_serialization(parser):
    """Check serialization after collecting inputs.
    """
    batch = Batch(['-i', 'input1', 'input2', '-o', 'output', '--alpha', '0.5'])
    batch.parse_args(parser)

    with patch('builtins.open', mock_open()) as mock:
        inputs, _ = batch.collect()
        data = json.loads(''.join([arg_list.args[0] for arg_list in mock().write.call_args_list]))

    assert inputs == [up.absolute(PurePath('input1')), up.absolute(PurePath('input2'))]

    assert data['args']['inputs'] == [
        up.absolute(PureUnifiedPath('input1')).as_unified(),
        up.absolute(PureUnifiedPath('input2')).as_unified()
    ]
    assert data['args']['output'] == up.absolute(PureUnifiedPath('output')).as_unified()
    assert data['args']['alpha'] == 0.5
    assert data['inputs'] == [
        up.absolute(PureUnifiedPath('input1')).as_unified(),
        up.absolute(PureUnifiedPath('input2')).as_unified()
    ]

    with pytest.raises(AssertionError):
        inputs, _ = batch.collect('tag1')


@patch('glob.glob', _mock_glob)
def test_collect_serialization_tags(parser):
    """Check serialization after collecting inputs with tags.
    """
    batch = Batch(['-i', '+tag1', 'input1', '+tag2', 'input2', '-o', 'output', '--alpha', '0.5'])
    batch.parse_args(parser)

    # First collect all inputs
    with patch('builtins.open', mock_open()) as mock:
        inputs, _ = batch.collect()
        data = json.loads(''.join([arg_list.args[0] for arg_list in mock().write.call_args_list]))

    assert inputs == [up.absolute(PurePath('input1')), up.absolute(PurePath('input2'))]

    assert data['args']['inputs'] == {
        'tag1': [up.absolute(PureUnifiedPath('input1')).as_unified()],
        'tag2': [up.absolute(PureUnifiedPath('input2')).as_unified()]
    }
    assert data['inputs'] == [
        up.absolute(PureUnifiedPath('input1')).as_unified(),
        up.absolute(PureUnifiedPath('input2')).as_unified()
    ]

    # Then only one tag
    with patch('builtins.open', mock_open()) as mock:
        inputs, _ = batch.collect('tag1')
        data = json.loads(''.join([arg_list.args[0] for arg_list in mock().write.call_args_list]))

    assert inputs == [up.absolute(PurePath('input1'))]

    assert data['args']['inputs'] == {
        'tag1': [up.absolute(PureUnifiedPath('input1')).as_unified()],
        'tag2': [up.absolute(PureUnifiedPath('input2')).as_unified()]
    }
    assert data['inputs'] == {'tag1': [up.absolute(PureUnifiedPath('input1')).as_unified()]}

    # Then another tag
    with patch('builtins.open', mock_open(read_data=json.dumps(data))) as mock:
        inputs, _ = batch.collect('tag2')
        data = json.loads(''.join([arg_list.args[0] for arg_list in mock().write.call_args_list]))

    assert inputs == [up.absolute(PurePath('input2'))]

    assert data['args']['inputs'] == {
        'tag1': [up.absolute(PureUnifiedPath('input1')).as_unified()],
        'tag2': [up.absolute(PureUnifiedPath('input2')).as_unified()]
    }
    assert data['inputs'] == {
        'tag1': [up.absolute(PureUnifiedPath('input1')).as_unified()],
        'tag2': [up.absolute(PureUnifiedPath('input2')).as_unified()]
    }

    # Non-existent tag
    with pytest.raises(AssertionError):
        inputs, _ = batch.collect('tag3')


@patch('glob.glob', _mock_glob)
def test_collect_serialization_groups_regex_1(parser):
    """Check serialization after collecting inputs with groups.
    """
    batch = Batch([
        '-i', '01~time#1~frame#1', '01~time#2~frame#2', '02~time#3~frame#1', '02~time#4~frame#2', '-o', 'output',
        '--alpha', '0.5', '--group-inputs', r'(.+?)~'
    ])
    batch.parse_args(parser)

    # First collect all inputs
    with patch('builtins.open', mock_open()) as mock:
        inputs, _ = batch.collect()
        data = json.loads(''.join([arg_list.args[0] for arg_list in mock().write.call_args_list]))

    assert inputs == [[up.absolute(PurePath('01~time#1~frame#1')),
                       up.absolute(PurePath('01~time#2~frame#2'))],
                      [up.absolute(PurePath('02~time#3~frame#1')),
                       up.absolute(PurePath('02~time#4~frame#2'))]]

    assert data['args']['inputs'] == [
        up.absolute(PureUnifiedPath('01~time#1~frame#1')).as_unified(),
        up.absolute(PureUnifiedPath('01~time#2~frame#2')).as_unified(),
        up.absolute(PureUnifiedPath('02~time#3~frame#1')).as_unified(),
        up.absolute(PureUnifiedPath('02~time#4~frame#2')).as_unified()
    ]

    assert data['inputs'] == [[
        up.absolute(PureUnifiedPath('01~time#1~frame#1')).as_unified(),
        up.absolute(PureUnifiedPath('01~time#2~frame#2')).as_unified()
    ],
                              [
                                  up.absolute(PureUnifiedPath('02~time#3~frame#1')).as_unified(),
                                  up.absolute(PureUnifiedPath('02~time#4~frame#2')).as_unified()
                              ]]


@patch('glob.glob', _mock_glob)
def test_collect_serialization_groups_regex_2(parser):
    """Check serialization after collecting inputs with groups.
    """
    batch = Batch([
        '-i', 'scene1/input1', 'scene1/input2', 'scene2/input1', 'scene2/input2', '-o', 'output', '--alpha', '0.5',
        '--group-inputs', r'(.+)/'
    ])
    batch.parse_args(parser)

    # First collect all inputs
    with patch('builtins.open', mock_open()) as mock:
        inputs, _ = batch.collect()
        data = json.loads(''.join([arg_list.args[0] for arg_list in mock().write.call_args_list]))

    assert inputs == [[up.absolute(PurePath('scene1/input1')),
                       up.absolute(PurePath('scene1/input2'))],
                      [up.absolute(PurePath('scene2/input1')),
                       up.absolute(PurePath('scene2/input2'))]]

    assert data['args']['inputs'] == [
        up.absolute(PureUnifiedPath('scene1/input1')).as_unified(),
        up.absolute(PureUnifiedPath('scene1/input2')).as_unified(),
        up.absolute(PureUnifiedPath('scene2/input1')).as_unified(),
        up.absolute(PureUnifiedPath('scene2/input2')).as_unified()
    ]

    assert data['inputs'] == [[
        up.absolute(PureUnifiedPath('scene1/input1')).as_unified(),
        up.absolute(PureUnifiedPath('scene1/input2')).as_unified()
    ],
                              [
                                  up.absolute(PureUnifiedPath('scene2/input1')).as_unified(),
                                  up.absolute(PureUnifiedPath('scene2/input2')).as_unified()
                              ]]


@patch('glob.glob', _mock_glob)
def test_collect_serialization_groups_inputs(parser):
    """Check serialization after collecting inputs with groups.
    """
    batch = Batch(['-a', '1.json'])

    content = json.dumps(
        {'args': {
            'inputs': [['input1', 'input2'], ['input3'], []],
            'output': 'output',
            'alpha': 0.5
        }})

    with patch('builtins.open', mock_open(read_data=content)):
        batch.parse_args(parser)

    with patch('builtins.open', mock_open()) as mock:
        inputs, _ = batch.collect()
        data = json.loads(''.join([arg_list.args[0] for arg_list in mock().write.call_args_list]))

    assert inputs == [[up.absolute(PurePath('input1')),
                       up.absolute(PurePath('input2'))], [up.absolute(PurePath('input3'))]]

    assert data['args']['inputs'] == [[
        up.absolute(PureUnifiedPath('input1')).as_unified(),
        up.absolute(PureUnifiedPath('input2')).as_unified()
    ], [up.absolute(PureUnifiedPath('input3')).as_unified()], []]

    assert data['inputs'] == [[
        up.absolute(PureUnifiedPath('input1')).as_unified(),
        up.absolute(PureUnifiedPath('input2')).as_unified()
    ], [up.absolute(PureUnifiedPath('input3')).as_unified()]]


@patch('glob.glob', _mock_glob)
def test_run_serialization_root(parser):
    """Check serialization after collecting inputs, inside a given root.
    """
    batch = Batch([
        '--root-dir', 'root', '-i', 'dir1/sub/input', 'dir2/sub/input', '-o', 'output', '--dir', ':dir', '--var',
        ':dir', '--alpha', '0.5'
    ])
    batch.parse_args(parser)

    with patch('builtins.open', mock_open()) as mock:
        inputs, parsed_args = batch.collect()
        data = json.loads(''.join([arg_list.args[0] for arg_list in mock().write.call_args_list]))

    assert inputs == [up.absolute(PurePath('root/dir1/sub/input')), up.absolute(PurePath('root/dir2/sub/input'))]

    assert parsed_args[0].dir == PurePath('dir1/sub')
    assert parsed_args[0].var == 'dir1_sub'

    assert parsed_args[1].dir == PurePath('dir2/sub')
    assert parsed_args[1].var == 'dir2_sub'

    assert data['args']['root_dir'] == up.absolute(PureUnifiedPath('root')).as_unified()
    # input path must be relative to root
    assert data['args']['inputs'] == ['dir1/sub/input', 'dir2/sub/input']
    assert data['args']['output'] == up.absolute(PureUnifiedPath('output')).as_unified()
    assert data['args']['alpha'] == 0.5
    assert data['args']['dir'] == up.absolute(PureUnifiedPath(':dir')).as_unified()
    assert data['args']['var'] == ':dir'
    assert data['inputs'] == [
        up.absolute(PureUnifiedPath('root/dir1/sub/input')).as_unified(),
        up.absolute(PureUnifiedPath('root/dir2/sub/input')).as_unified()
    ]
