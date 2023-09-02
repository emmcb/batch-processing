# Copyright 2023 Emmanuel Chaboud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import glob
import itertools
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path, PurePath

import unified_path as up
from mergedeep import Strategy as MergeStrategy
from mergedeep import merge
from tqdm import tqdm
from unified_path import UnifiedPath, UnifiedPathEncoder


class Batch:
    """The Batch class.
    """
    USE_INPUT_EXT = ':input_ext'

    def __init__(self, argv=[], enable_multiprocess=True, enable_progress=True):
        """Batch constructor.

        Args:
            argv: List of command line arguments passed to a Python script.
            enable_multiprocess: Wether to use a multiprocessing pool or a simple loop iterator, by default True.
            enable_progress: Wether to display a nice tqdm progress bar, by default True.
        """
        self.__argv = argv
        self.__enable_multiprocess = enable_multiprocess
        self.__enable_progress = enable_progress

        self.__input_extensions = None
        self.__input_help = None
        self.__output_extensions = None
        self.__output_help = None

        self.__default_args = None

        self.__args = None
        self.__unknown_args = None
        self.__serialized_inputs = None

        self.__serialization_file = None
        self.__last_serialized_tag = None

    def is_multiprocessing_enabled(self):
        """Verify if we are in a multiprocessing pool or a simple loop iterator.
        """
        return self.__enable_multiprocess

    def set_multiprocessing_enabled(self, enable_multiprocess: bool):
        """Set if we should use a multiprocessing pool or a simple loop iterator.

        Args:
            enable_multiprocess: Wether to use a multiprocessing pool or a simple loop iterator.
        """
        self.__enable_multiprocess = enable_multiprocess

    def get_args(self):
        """Returns parsed argument namespace.
        """
        return self.__args

    def get_output_dir(self):
        """Returns the batch output directory.

        This fonction is different from just using args.output because the output can be a file or a folder, and may
        contain variables like :dir.
        """
        if vars(self.__args).get('output', None) is None:
            return None

        if self.__args.output.suffix:
            # If the output path contains an extension, then we consider that it is a file, and we return the parent
            # folder.
            return self.__args.output.parent

        out_dir = self.__args.output

        # Strip variables so we keep the base output directory.
        for i, p in enumerate(out_dir.parts):
            if p.startswith(':'):
                out_dir = Path(*out_dir.parts[:i])
                break

        return out_dir

    def set_io_description(self,
                           input_extensions: list[str] = None,
                           input_help: str = None,
                           output_extensions: list[str] = None,
                           output_help: str = None):
        """Set description for inputs and outputs.

        Args:
            input_extensions: Allowed extensions for input files, by default None.
            input_help: Help string for input files, by default None.
            output_extensions: Allowed extensions for output files, by default None.
            output_help: Help string for output files, by default None.
        """
        self.__input_extensions = input_extensions
        self.__input_help = input_help
        self.__output_extensions = output_extensions
        self.__output_help = output_help

    def set_args(self, **kwargs):
        """Set default argument values. Will be overriden by command line arguments.
        """
        self.__default_args = kwargs

    def parse_args(self, parser: argparse.ArgumentParser):
        """Parse command line arguments.

        Args:
            parser: An existing parser with the user arguments. Will be populated with the batch arguments.
        Returns:
            The populated namespace.
        """
        self.__populate_argument_parser(parser)
        self.__try_deserialize_args(parser)

        self.__args = parser.parse_args(self.__argv)
        self.__resolve_args()

        return self.__args

    def parse_known_args(self, parser: argparse.ArgumentParser):
        """Parse command line arguments.

        Like parse_args(), except that it does not produce an error when extra
        arguments are present. Instead, it returns a two item tuple containing the populated namespace and the list of
        remaining argument strings.

        Args:
            parser: An existing parser with the user arguments. Will be populated with the batch arguments.
        Returns:
            The populated namespace and the list of extra arguments.
        """
        self.__populate_argument_parser(parser)
        self.__try_deserialize_args(parser)

        self.__args, unknown_args = parser.parse_known_args(self.__argv)
        self.__unknown_args += unknown_args
        self.__resolve_args()

        return self.__args, self.__unknown_args

    def collect(self, tag: str = None):
        """Collect the inputs related to a given tag, or all if not given.

        Args:
            tag: The tag to collect, by default None.
        Returns:
            The collected input list.
            A list that contains for each input a tuple with the corresponding parsed arguments.
        """
        assert self.__args, 'Arguments must have been set before running the batch'

        if tag:
            logging.info('Collecting inputs with tag %s...' % tag)
        else:
            logging.info('Collecting inputs...')

        # -- INPUTS --

        tags = [tag] if tag else self.__args.tags
        collect_start = time.time()

        if self.__serialized_inputs:
            # We consider the overwritten inputs list to be already serialized: no need to parse it one more time
            inputs = self.__collect_tags(self.__serialized_inputs, tags)
        else:
            inputs = []
            if 'inputs' in vars(self.__args) and self.__args.inputs:
                input_list = self.__collect_tags(self.__args.inputs, tags)

                is_grouped_inputs_v = [isinstance(input, list) for input in input_list]
                is_grouped_inputs = all(is_grouped_inputs_v)
                assert is_grouped_inputs or not any(is_grouped_inputs_v), "mixing grouped inputs is not allowed"

                if is_grouped_inputs:
                    inputs = list(map(self.__collect_and_filter_inputs, input_list))
                    inputs = [inp for inp in inputs if inp]  # removes empty groups
                    inputs.sort()  # sort groups
                else:
                    inputs = self.__collect_and_filter_inputs(input_list)

                    if 'group_inputs' in vars(self.__args) and self.__args.group_inputs:
                        re_group = re.compile(self.__args.group_inputs)
                        inputs = self.__group_inputs(inputs, re_group)

        assert len(inputs) > 0, 'Cannot parse any input data'
        logging.info('Collected %d input(s) in %f second.' % (len(inputs), time.time() - collect_start))

        # -- SERIALIZATION --

        self.__serialize_args(inputs, tag)

        # -- OTHER ARGUMENTS --

        parsed_args = list(map(self.__parse_variables, inputs))

        return inputs, parsed_args

    def run(self, runnable, *user_args, tag: str = None, disable_output=False, output_ext: str = None):
        """Batch a function over a set of inputs.

        Args:
            runnable:
                A function that is parallelized for each input file. This will be called for each input with the
                following arguments:
                    - the current input
                    - the current output
                    - the parsed arguments corresponding to the input
                    - and any of the extra arguments passed to run().
            tag: The tag to collect, by default None.
            disable_output: Set to True if the run function does not produce any output file, by default False.
            output_ext: If given, will append this extension to the output filenames.
                Can be Batch.USE_INPUT_EXT to re-use input extension.
        Returns:
            A list containing the return values of the runnable.
        """
        inputs, parsed_args = self.collect(tag)

        # -- OUTPUTS --

        if 'output' not in vars(self.__args):
            disable_output = True

        if disable_output:
            iterable = zip(inputs, parsed_args, *(itertools.repeat(user_arg) for user_arg in user_args))

        else:
            cwd = Path.cwd()

            def get_output_name(input, args):
                if isinstance(input, list):
                    # Take first element in case of grouped inputs
                    input = input[0]

                output = input.stem

                if args.output_prefix:
                    sep = '_' if not args.output_prefix.endswith('_') else ''
                    output = args.output_prefix + sep + output
                if args.output_suffix:
                    sep = '_' if not args.output_suffix.startswith('_') else ''
                    output = output + sep + args.output_suffix

                if output_ext == self.USE_INPUT_EXT:
                    output += input.suffix
                elif output_ext:
                    sep = '.' if not output_ext.startswith('.') else ''
                    output = output + sep + output_ext

                args.output.mkdir(parents=True, exist_ok=True)

                return up.absolute(args.output, cwd, norm=True) / output

            if self.__args.output is not None:
                outputs = list(itertools.starmap(get_output_name, zip(inputs, parsed_args)))
            else:
                outputs = [None] * len(inputs)

            iterable = zip(inputs, outputs, parsed_args, *(itertools.repeat(user_arg) for user_arg in user_args))

        # -- MULTIPROCESS POOL --

        if self.__enable_progress:
            iterable = tqdm(iterable, total=len(inputs))

        if not self.__enable_multiprocess or self.__args.num_processes <= 1:
            return list(itertools.starmap(runnable, iterable))

        else:
            with multiprocessing.Pool(processes=self.__args.num_processes) as pool:
                return list(pool.imap(_multiprocess_imap_helper, map(lambda item: (runnable, item), iterable)))

    def __populate_argument_parser(self, parser: argparse.ArgumentParser):
        """Populate the given parser with the batch arguments.
        """
        group = parser.add_argument_group(title='batch inputs arguments')
        group.add_argument('-i',
                           '--inputs',
                           type=up.make_path_validator(extensions=self.__input_extensions),
                           required='inputs' not in parser._defaults,
                           nargs='+',
                           help=self.__input_help or 'list of input files, optionally with regex',
                           action=_InputParserAction)
        group.add_argument('--root-dir', type=UnifiedPath, help='root folder of input files')
        group.add_argument('--only', nargs='+', help='list of regex to only include some input pattern')
        group.add_argument('--exclude', nargs='+', help='list of regex to exclude some input pattern')
        group.add_argument('--tags',
                           nargs='+',
                           help='tag to select a subset of inputs when the later is in dictionary form')
        group.add_argument(
            '--group-inputs',
            help='group inputs according to the given regex, that identificates a common group key in the filenames')

        if self.__output_help != argparse.SUPPRESS:
            group = parser.add_argument_group('batch output arguments')
            group.add_argument('-o',
                               '--output',
                               type=up.make_path_validator(extensions=self.__output_extensions),
                               required='output' not in parser._defaults,
                               help=self.__output_help or 'output path')
            group.add_argument('--prefix', dest='output_prefix', help='output prefix')
            group.add_argument('--suffix', dest='output_suffix', help='output suffix')

        if self.__enable_multiprocess:
            group = parser.add_argument_group('batch multiprocessing arguments')
            group.add_argument('-j',
                               '--num-processes',
                               type=int,
                               default=max(min(multiprocessing.cpu_count() - 2, 12), 1),
                               help='number of parallel processes')

        group = parser.add_argument_group('batch logging arguments')
        group.add_argument('--log-file',
                           type=UnifiedPath,
                           help='redirect logs to specified file instead of standard output')
        group.add_argument('--log-level',
                           choices=['debug', 'info', 'warning', 'error'],
                           default='info',
                           help='log level')

        group = parser.add_argument_group('batch serialization arguments')
        self.__populate_serialization_arguments(group)

        # Argparse only applies type conversion on default value if it is a string, otherwise it uses the value as-is.
        # We override this behavior by recursively applies the type to the default object. This is useful for example
        # to automatically convert a list of strings to a list of paths.
        for action in parser._actions:
            if action.type is not None:
                default = parser.get_default(action.dest)
                if default is not None:
                    parser.set_defaults(**{action.dest: self.__recurse(action.type, default)})

        # If the user requested the help, parse and stop now.
        if '-h' in self.__argv or '--help' in self.__argv:
            parser.parse_args(self.__argv)

    @classmethod
    def __populate_serialization_arguments(cls, parser: argparse.ArgumentParser):
        """Populate the given parser with the batch arguments related to serialization.
        """
        parser.add_argument('-a',
                            '--arg-files',
                            type=up.make_path_validator(extensions=['.json']),
                            nargs='+',
                            help='list of serialized json files')
        parser.add_argument('--regenerate-inputs',
                            action='store_true',
                            help='ignore serialized input list and generate a new one')

    def __try_deserialize_args(self, parser: argparse.ArgumentParser):
        """Read from the command line the list of JSON files containing the serialized argument.

        The parser will be modified with the deserialized data.
        """
        merged_data = {}

        if not self.__argv:
            # No argument given on the command line, take the batch defaults
            if self.__default_args is not None:
                merged_data = self.__parse_arg_data({"args": self.__default_args}, parser)
                if 'arg_files' in self.__default_args and 'regenerate_inputs' not in self.__default_args:
                    # When batch default_args has arg_files, batch will parse arg_files.
                    parsed_args_from_file = self.__merge_args(self.__default_args['arg_files'], parser)
                    if parsed_args_from_file.get('inputs', None):
                        # When args_files constains inputs, put them to serialized inputs
                        self.__serialized_inputs = self.__recurse(UnifiedPath, parsed_args_from_file['inputs'])

        else:
            # Create a temporary parser that only parses the serialization arguments
            serialization_parser = argparse.ArgumentParser(add_help=False)
            self.__populate_serialization_arguments(serialization_parser)

            # Parse serialization arguments
            args, _ = serialization_parser.parse_known_args(self.__argv)

            # Parse and merge into one dict the serialized argument list.
            if args.arg_files is not None:
                merged_data = self.__merge_args(args.arg_files, parser)

            # Set serialized input list if present, so we will not have to parse inputs again.
            if not args.regenerate_inputs and merged_data.get('inputs', None):
                self.__serialized_inputs = self.__recurse(UnifiedPath, merged_data['inputs'])

        # Backup extra arguments for later use.
        self.__unknown_args = merged_data.get('unknownArgs', [])

        # For each deserialized argument, modify the corresponding parser action to set the deserialized value as the
        # action default. That's way:
        # - If the argument is given on the command line, it will override the deserialized value.
        # - If the argument is not given, it will take the default value, that is to say the deserialized value. In
        # this case we do not want to raise an error if the argument was required, so explictely set required to False.
        for k, v in merged_data.get('args', {}).items():
            action = next((action for action in parser._actions if action.dest == k), None)
            if action is None:
                continue

            action.default = v
            action.required = False

            # Also set required to False on needed mutually exclusive groups
            for group in parser._mutually_exclusive_groups:
                action = next((action for action in group._actions if action.dest == k), None)
                if action is None:
                    continue

                group.required = False

    @classmethod
    def __merge_args(cls, arg_files: list[str], parser: argparse.ArgumentParser, has_root_dir=False) -> dict:
        """Deserialize the given serialized files and merge all the data.
        """
        merged_data = {}

        for arg_file in list(map(UnifiedPath, arg_files)):
            with open(arg_file, 'r') as f:
                data = json.load(f)

            # Parse the data
            data = cls.__parse_arg_data(data, parser, root_dir=up.absolute(arg_file).parent, has_root_dir=has_root_dir)

            # Do the actual merge
            merged_data = merge(merged_data, data, strategy=MergeStrategy.TYPESAFE_ADDITIVE)
            has_root_dir = has_root_dir or bool(merged_data.get('args', {}).get('root_dir', None))

        return merged_data

    @classmethod
    def __parse_arg_data(cls,
                         data: dict,
                         parser: argparse.ArgumentParser,
                         root_dir=Path.cwd(),
                         has_root_dir=False) -> dict:
        """Parse arg data as deserialized from JSON files.

        Relative paths in serialized files are not relative to the current working directory, but relative to the
        serialized file directory. As we will lose the origin after the merge, we need to transform here all the
        relative paths to absolute paths, by taking into account correct root directory.
        """
        data = deepcopy(data)

        if data.get('args', None):
            expand_path = cls.__make_path_expander(root_dir)
            has_root_dir = has_root_dir or bool(data['args'].get('root_dir', None))

            for k, v in data['args'].items():
                if v is None:
                    continue

                # Use the parser to infer the data type.
                # This is needed because we lost all the type information during the JSON (de)serialization.
                actions = parser._actions if parser else []
                action = next((action for action in actions if action.dest == k), None)
                if action is None:
                    continue

                if action.type is not None:
                    v = cls.__recurse(action.type, v)

                # When root is given, inputs must be relative to this root.
                # We must not transform them to absolute paths here.
                if not has_root_dir or k != 'inputs':
                    v = cls.__recurse(expand_path, v)

                data['args'][k] = v

        if data.get('unknownArgs', None):
            # For extra arguments we do not have any type information, because they are not handled by the parser.
            # Thus we need to guess if it is a path by searching the path separator.
            expand_path = cls.__make_path_expander_guess(root_dir)
            data['unknownArgs'] = cls.__recurse(expand_path, data['unknownArgs'])

        # If the current serialized file itself contains other serialized files, then recursively merge them.
        if data.get('args', None) and data['args'].get('arg_files', None):
            arg_file_children = list(map(UnifiedPath, data['args']['arg_files']))
            data['args'].pop('arg_files')

            data = merge(cls.__merge_args(arg_file_children, parser, has_root_dir=has_root_dir),
                         data,
                         strategy=MergeStrategy.TYPESAFE_ADDITIVE)

        return data

    @classmethod
    def __make_path_expander(cls, root_dir=Path.cwd()):
        """Generate a function (to be used for example with __recurse())
        that will expand all relative paths to absolute.

        Assume that the path are typed as Path objects.

        Args:
            root_dir: The root directory, by default Path.cwd().
        """

        def expand_path(obj):
            if isinstance(obj, PurePath):
                return up.absolute(obj, root_dir)

            return obj

        return expand_path

    @classmethod
    def __make_path_expander_guess(cls, root_dir=Path.cwd()):
        """Generate a function (to be used for example with __recurse())
        that will expand all relative paths to absolute.

        Assume that the path are NOT typed as Path objects, then we must guess by searching a separator.

        Args:
            root_dir: The root directory, by default Path.cwd().
        """

        def expand_path(obj):
            if isinstance(obj, str) and (os.path.sep in obj or (os.path.altsep and os.path.altsep in obj)):
                return up.absolute(UnifiedPath(obj), root_dir)

            return obj

        return expand_path

    @classmethod
    def __recurse(cls, func, obj):
        """Helper to recursively apply a function inside a list or a dictionary.
        """
        if obj is None:
            return None

        if isinstance(obj, list):
            return [cls.__recurse(func, v) for v in obj]

        if isinstance(obj, dict):
            return {k: cls.__recurse(func, v) for k, v in obj.items()}

        return func(obj)

    def __resolve_args(self):
        """Common batch initialization after argument parsing.
        """
        self.__root_dir = up.absolute(self.__args.root_dir, norm=True) if self.__args.root_dir is not None else None

        log_level = {'error': logging.ERROR, 'warning': logging.WARNING, 'info': logging.INFO, 'debug': logging.DEBUG}
        self.__log_level = log_level[self.__args.log_level]

        self.__logger_init(logging.getLogger())

    def __logger_init(self, logger: logging.Logger):
        """Logger initialization.
        """
        if self.__args.log_file is not None:
            handler = logging.FileHandler(self.__args.log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        else:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')

        handler.setLevel(self.__log_level)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(self.__log_level)

    @classmethod
    def __collect_tags(cls, inputs, tags):
        """ Extract input list corresponding to given tags, if given.
        """
        if isinstance(inputs, list):
            assert not tags, ('Input is a list, not a dictionary hence cannot '
                              'extract data for the requested tag(s) {}'.format(tags))
            return inputs

        if isinstance(inputs, dict):
            tags = tags or list(inputs.keys())
            input_list = []
            for tag in tags:
                assert tag in inputs, 'Given tag {} is not in the input list'.format(tag)
                input_list += inputs[tag]

            return input_list

        raise AssertionError('Input must be a list or a dict')

    def __collect_and_filter_inputs(self, input_list: list[Path]):
        """Glob the inputs matching the user-provided list, then apply the regex filtering.
        """
        cwd = Path.cwd()

        def collect_inputs(input: Path):
            if self.__root_dir:
                if input.is_absolute():
                    assert up.is_relative_to(
                        input, self.__root_dir), '%s is not relative to root: %s' % (input, self.__root_dir)
                else:
                    input = self.__root_dir / input
            else:
                input = up.absolute(input, cwd, norm=True)

            # Ideally we should use input.is_file() here, but due to a bug in pathlib this raises an error 123 on
            # Windows when the path contains a wildcard. Thus we fallback on os.path.isfile.
            # See https://bugs.python.org/issue35306
            if os.path.isfile(input):
                return [input]

            inputs = glob.glob(os.fspath(input), recursive=True)
            assert inputs, '%s does not match any file' % input

            inputs = map(UnifiedPath, inputs)

            return inputs

        inputs = []
        for input in input_list:
            inputs += collect_inputs(input)

        inputs = list(set(inputs))  # removes duplicates
        inputs.sort()  # sort

        if 'only' in vars(self.__args) and self.__args.only:
            inputs = [input for input in inputs if any([input.match(only) for only in self.__args.only])]

        if 'exclude' in vars(self.__args) and self.__args.exclude:
            inputs = [input for input in inputs if all([not input.match(exclude) for exclude in self.__args.exclude])]

        return inputs

    @classmethod
    def __group_inputs(cls, inputs, re_group):
        """ Group inputs sharing a common key, obtained with the given regex.
        """

        def search_key(input):
            # Take the unified path representation to ensure that the path separators are the same across platforms
            input = input.as_unified()

            res = re.search(re_group, input)
            assert res and len(
                res.groups()) > 0, 'Could not match input %s with group regex %s' % (input, str(re_group))

            return '|'.join(res.groups())

        keys = [(i, search_key(input)) for i, input in enumerate(inputs)]
        keys = sorted(keys, key=lambda x: x[1])
        groups = [list(group) for _, group in itertools.groupby(keys, key=lambda x: x[1])]

        return [[inputs[i] for i, _ in group] for group in groups]

    def __parse_variables(self, input):
        """Parse the variables contained in the arguments, relative to a given input.

        Currently supported variables are:
            - :dir represents the part of the path between the root dir and the file.
                For example, if the root dir is B:/Images, and the input is B:/Images/Dir1/Dir2/image.tif,
                then :dir = "Dir1/Dir2".
        """

        def parse_arg(arg):
            # If there is no root dir, nothing to parse
            if not self.__root_dir:
                return arg

            # only str (no inherited type) or PurePath (and inherited type)
            if type(arg) is str or isinstance(arg, PurePath):
                return self.__parse_arg_string(arg, input)

            return arg

        parsed_args = argparse.Namespace()

        for arg in vars(self.__args):
            if arg == 'inputs':
                continue
            setattr(parsed_args, arg, self.__recurse(parse_arg, getattr(self.__args, arg)))

        setattr(parsed_args, 'unknown_args', self.__recurse(parse_arg, self.__unknown_args))

        return parsed_args

    def __parse_arg_string(self, arg, input):
        """Parse the string variables contained in the argument, relative to a given input.
        """
        if isinstance(input, list):
            # Take first element in case of grouped inputs
            input = input[0]

        rel_path = input.relative_to(self.__root_dir).parent

        is_path = isinstance(arg, PurePath)
        db_dir = str(rel_path) if is_path else str(rel_path).replace(os.path.sep, '_')

        arg = str(arg)
        arg = arg.replace(':dir', db_dir)

        return up.normalized(Path(arg)) if is_path else arg

    def __serialize_args(self, inputs, tag):
        """Serialize and save current batch state to a JSON file.
        """
        if vars(self.__args).get('output', None) is None:
            # No output, no serialization.
            return

        # Save the current git branch and commit in repository.
        script_path = up.absolute(UnifiedPath(sys.argv[0]))

        try:
            git_args = ['git', 'rev-parse', '--show-toplevel']
            git_root = subprocess.check_output(git_args, cwd=script_path.parent).strip().decode('utf-8')

            git_args = [
                'git', '--git-dir',
                os.path.join(git_root, '.git'), '--work-tree', git_root, 'describe', '--long', '--broken', '--all'
            ]
            git_describe = subprocess.check_output(git_args).strip().decode('utf-8')
        except subprocess.CalledProcessError:
            git_describe = None

        # Transform all relative paths to absolute paths, so the serialization file does not need to be at a specific
        # place.
        args = self.__recurse(self.__make_path_expander(), vars(self.__args))
        unknown_args = self.__recurse(self.__make_path_expander_guess(), self.__unknown_args)

        # When root is given, inputs must be relative to this root.
        # We must not transform them to absolute paths here.
        if self.__args.root_dir:
            args['inputs'] = self.__args.inputs

        # Save the batch state
        data = {
            'script': script_path.as_unified(),
            'git_describe': git_describe,
            'args': args,
            'unknownArgs': unknown_args,
            'inputs': inputs if tag is None else {
                tag: inputs
            }
        }

        if self.__serialization_file is None or (tag is None) != (self.__last_serialized_tag is None):
            today = datetime.datetime.today()
            basename = PurePath(sys.argv[0]).stem

            out_dir = self.get_output_dir() / '_batch'
            out_dir.mkdir(parents=True, exist_ok=True)

            self.__serialization_file = Path(
                out_dir, '%s_%04d%02d%02d_%02d%02d%02d.json' %
                (basename, today.year, today.month, today.day, today.hour, today.minute, today.second))

            # Protect against overwritting an existing file
            idx = 1
            serialization_file_orig = Path(self.__serialization_file)
            while self.__serialization_file.is_file():
                self.__serialization_file = serialization_file_orig.with_name(
                    '%s_%d%s' % (serialization_file_orig.stem, idx, serialization_file_orig.suffix))
                idx += 1

        # If serialization file already exists, it means that we have called multiple times collect(tag) with different
        # tags. In this case, we must update the existing file in order to append the new collected inputs.
        elif tag is not None:
            with open(self.__serialization_file, 'r') as f:
                previous_data = json.load(f)

            assert 'inputs' in previous_data and isinstance(previous_data['inputs'], dict)

            data['inputs'].update({k: v for k, v in previous_data['inputs'].items() if k != tag})

        with open(self.__serialization_file, 'w') as f:
            json.dump(data, f, cls=UnifiedPathEncoder, indent=4, sort_keys=True)

        self.__last_serialized_tag = tag


class _InputParserAction(argparse.Action):
    """Custom argparse action for input argument (-i).

    When there is (are) triggers (the letter +) in input argument, we parse it like a dictionary where the word right
    after the trigger is considered as the key and the next words are considered as the values.

    For example:
    -i +tag1 value_1 value_2 +tag2 value_3 value_4 value_5

    The input argument will be the dictionary:
    {
        "tag1": [value_1, value_2],
        "tag2": [value_3, value_4, value_5]
    }

    If there is no trigger, the default behavior is used.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        trigger_indexes = [i for i in range(len(values)) if '+' == str(values[i])[0]]
        if len(trigger_indexes) > 0:
            setattr(namespace, self.dest, dict())
            for i in range(len(trigger_indexes)):
                start_index = trigger_indexes[i] + 1
                stop_index = len(values) if i + 1 == len(trigger_indexes) else trigger_indexes[i + 1]
                tag = str(values[trigger_indexes[i]])[1:]
                getattr(namespace, self.dest)[tag] = [values[x] for x in range(start_index, stop_index)]
        else:  # no trigger '+' presents, so use the default behavior
            setattr(namespace, self.dest, values)


def _multiprocess_imap_helper(args):
    """Helper function for imap.

    This is needed since built-in multiprocessing library does not have `istarmap` function.
    If packed arguments are passed, it unpacks the arguments and pass through the function.
    Otherwise, it just pass the argument through the given function.

    Args:
        args: User-defined function and arguments to pass through.
    """
    assert len(args) == 2
    func = args[0]

    # Arguments are packed as a list or a tuple, so pass *args.
    if isinstance(args[1], list) or isinstance(args[1], tuple):
        return func(*(args[1]))

    # Otherwise, just pass the argument.
    return func(args[1])
