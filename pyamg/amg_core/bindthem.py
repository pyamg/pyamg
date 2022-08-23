#! /usr/bin/env python3
import yaml
import os

PYBINDHEADER = """\
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "{}"

namespace py = pybind11;
"""


def find_comments(fname, ch):
    """
    Find the comments for a function.

    fname: filename
    ch: CppHeaderParser parse tree

    The function must look like
    /*
     * comments
     * comments
     */
     template<class I, ...>
     void somefunc(...){

     -or-

    /*
     * comments
     * comments
     */
     void somefunc(...){

     -or-

    with // style comments

    Then, take off the first three spaces
    """
    with open(fname) as inf:
        fdata = inf.readlines()

    comments = {}
    for f in ch.functions:
        lineno = f['line_number'] - 1  # zero based indexing

        # set starting position
        lineptr = lineno - 1
        if f['template']:
            lineptr -= 1
        start = lineptr

        # find the top of the comment block
        while fdata[lineptr].startswith('//') or\
            fdata[lineptr].startswith('/*') or\
                fdata[lineptr].startswith(' *'):
            lineptr -= 1
        lineptr += 1
        comment = fdata[lineptr:(start + 1)]
        comment = [c[3:].rstrip() for c in comment]
        comments[f['name']] = '\n'.join(comment).strip()

    return comments


def build_function(func):
    """
    Build a function from a templated function.  The function must look like
    template<class I, class T, ...>
    void func(const p[], p_size, ...)

    rules:
        - a pointer or array p is followed by int p_size
        - all arrays are templated
        - non arrays are basic types: int, double, complex, etc
        - all functions are straight up c++
    """

    indent = '    '

    # temlpate and function name

    if func['template']:
        fdef = func['template'] + '\n'
    else:
        fdef = ''

    newcall = func['returns'] + ' _' + func['name'] + '('
    fdef += newcall + '\n'

    # function parameters

    # for each parameter
    # if it's an array
    #   - replace with py::array_t
    #   - skip the next _size argument
    #   - save in a list of arrays
    # else replicate

    i = 0
    arraylist = []
    needsize = False
    while i < len(func['parameters']):
        p = func['parameters'][i]
        i += 1

        # check if pointer/array
        if p['pointer'] or p['array']:
            paramtype = p['raw_type']
            const = ''
            if p['constant']:
                const = 'const '

            param = f'py::array_t<{paramtype}> &' + ' ' + p['name']
            arraylist.append((const, paramtype, p['name']))
            needsize = True
        elif '_size' not in p['name'] and needsize:
            # not a size, but needed one
            raise ValueError(
                'Expecting a _size parameter for {}'.format(
                    p['name']))
        elif '_size' in p['name']:
            # just size, skip it
            needsize = False
            continue
        else:
            # if not a pointer, just copy it
            param = p['type'] + ' ' + p['name']

        fdef += f'{param:>25}'    # set width to 25
        fdef += ',\n'
    fdef = fdef[:-2]  # remove trailing comma and newline
    fdef += '\n' + ' ' * len(newcall) + ')'
    fdef += '\n{\n'

    # make a list of python objects
    for a in arraylist:
        if 'const' in a[0]:
            unchecked = '.unchecked();\n'
        else:
            unchecked = '.mutable_unchecked();\n'

        fdef += indent
        fdef += "auto py_" + a[2] + ' = ' + a[2] + unchecked

    # make a list of pointers to the arrays
    for a in arraylist:
        if 'const' in a[0]:
            data = '.data();\n'
        else:
            data = '.mutable_data();\n'
        fdef += indent
        fdef += a[0] + a[1] + ' *_' + a[2] + ' = py_' + a[2] + data

    # get the template signature
    if len(arraylist) > 0:
        fdef += '\n'
    if func['template']:
        template = func['template']
        template = template.replace('template', '').replace(
            'class ', '')   # template <class T> ----> <T>
    else:
        template = ''
    newcall = '    return ' + func['name'] + template + '('
    fdef += newcall + '\n'

    # function parameters
    for i, p in enumerate(func['parameters']):
        if '_size' in p['name']:
            fdef = fdef.strip()
            name, s = p['name'].split('_size')
            if s == '':
                s = '0'
            fdef += f" {name}.shape({s})"
        else:
            if p['pointer'] or p['array']:
                name = '_' + p['name']
            else:
                name = p['name']
            fdef += f'{name:>25}'
        if i < len(func['parameters'])-1:
            fdef += ',\n'
    fdef += '\n' + ' ' * len(newcall) + ');\n}'
    return fdef


def build_plugin(headerfile, ch, comments, inst, remaps):
    """
    Take a header file (headerfile) and a parse tree (ch)
    and build the pybind11 plugin

    headerfile: somefile.h

    ch: parse tree from CppHeaderParser

    comments: a dictionary of comments

    inst: files to instantiate

    remaps: list of remaps
    """
    headerfilename = os.path.splitext(headerfile)[0]

    indent = '    '
    plugin = ''

    # plugin += '#define NC py::arg().noconvert()\n'
    # plugin += '#define YC py::arg()\n'
    plugin += f'PYBIND11_MODULE({headerfilename}, m) {{\n'
    plugin += indent + 'm.doc() = R"pbdoc(\n'
    plugin += indent + f'Pybind11 bindings for {headerfile}\n\n'
    plugin += indent + 'Methods\n'
    plugin += indent + '-------\n'
    for f in ch.functions:
        templated = bool(f['template'])
        if not templated:
            plugin += indent + f['name'] + '\n'
        if templated:
            for func in inst:
                if f['name'] in func['functions']:
                    plugin += indent + f['name'] + '\n'
    plugin += indent + ')pbdoc";\n\n'

    plugin += indent + 'py::options options;\n'
    plugin += indent + 'options.disable_function_signatures();\n\n'

    unbound = []
    bound = []
    for f in ch.functions:
        # for each function:
        #   1 determine if the function is templated
        #   2 if templated:
        #       - find the entry in the instantiation list
        #   3 note any array parameters to the function
        #   4 if templated:
        #       - for each type, instantiate and bind
        #   5 if not templated:
        #       - bind

        # 1
        templated = bool(f['template'])

        # 2
        found = False
        if templated:
            for func in inst:
                if f['name'] in func['functions']:
                    found = True
                    types = func['types']

            if not found:
                # print('Could not find an instantiation for {}'.format(f['name']))
                unbound.append(f['name'])
                continue
            else:
                bound.append(f['name'])
        else:
            unbound.append(f['name'])
            types = [None]
            continue

        # 3
        # find all parameter names and mark if array
        argnames = []
        for p in f['parameters']:

            array = False
            if p['pointer'] or p['array']:
                array = True

            # skip "_size" parameters
            if '_size' in p['name']:
                continue
            else:
                argnames.append((p['name'], array))

        ntypes = len(types)
        for i, t in enumerate(types):

            # add the function call with each template
            instname = f['name']

            # check the remaps
            for remap in remaps:
                if f['name'] in remap:
                    instname = remap[f['name']]

            if t is not None:
                # templated function
                typestr = '<' + ', '.join(t) + '>'
            else:
                # not a templated function
                typestr = ''

            plugin += indent + \
                'm.def("{}", &_{}{}'.format(instname, f['name'], typestr)

            # name the arguments
            pyargnames = []
            for p, array in argnames:
                convert = ''
                if array:
                    convert = '.noconvert()'
                pyargnames.append(f'py::arg("{p}"){convert}')

            argstring = indent + indent + ', '.join(pyargnames)
            if len(pyargnames) > 0:
                argstring = ',\n' + argstring
            plugin += argstring

            # add the docstring to the last
            if i == ntypes - 1:
                plugin += ',\nR"pbdoc(\n{})pbdoc");\n'.format(
                    comments[f['name']])
            else:
                plugin += ');\n'
        plugin += '\n'

    plugin += '}\n'

    return plugin, bound, unbound


def main():
    import argparse
    import CppHeaderParser

    parser = argparse.ArgumentParser(
        description='Wrap a C++ header with Pybind11')

    parser.add_argument(
        "-o",
        "--output-file",
        metavar="FILE",
        help="(default output name for header.h is header_bind.cpp)")

    parser.add_argument("input_file", metavar="FILE")

    args = parser.parse_args()

    #
    # Parse the header file
    #
    print(f'[Generating binding for {args.input_file}]')
    ch = CppHeaderParser.CppHeader(args.input_file)
    comments = find_comments(args.input_file, ch)

    #
    # load the instantiate file
    #
    if args.input_file == 'bind_examples.h':
        data = yaml.safe_load(open('instantiate-test.yml'))
    else:
        try:
            data = yaml.safe_load(open('instantiate.yml'))
        except:
            data = {'instantiate': None}
    inst = data['instantiate']

    #
    # remap functions
    #
    if 'remaps' in data:
        remaps = data['remaps']
    else:
        remaps = []

    #
    # build the plugin
    #
    plugin, bound, unbound = build_plugin(
        args.input_file, ch, comments, inst, remaps)

    #
    # build each function
    #
    print('\t[unbound functions: {}]'.format(' '.join(unbound)))
    flist = []
    for f in ch.functions:
        if f['name'] in bound:
            print('\t[building {}]'.format(f['name']))
            fdef = build_function(f)
            flist.append(fdef)

    #
    # write to _bind.cpp
    #
    if args.output_file is not None:
        outf = args.output_file
    else:
        basename = os.path.splitext(args.input_file)[0]
        outf = basename + '_bind.cpp'

    with open(outf, 'wt') as outf:

        print('// DO NOT EDIT: this file is generated\n', file=outf)
        print(PYBINDHEADER.format(args.input_file), file=outf)

        for f in flist:
            print(f, '\n', file=outf, sep="")

        print(plugin, file=outf)


if __name__ == '__main__':
    main()
