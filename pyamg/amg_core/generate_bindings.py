#! /usr/bin/env python3
import re
import yaml

PYBINDHEADER = """\
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "%s"

namespace py = pybind11;
"""


def find_comments(fname):
    """
    Find the comments for a templated function.  The function must look like
    /*
     * comments
     * comments
     */
     template<class I, ...>
     void somefunc(...){

    or with // style comments
    """
    with open(fname, 'r') as inf:
        f = inf.read()

    lines = f.split('\n')

    comments = {}
    startcomment = 0
    endcomment = 0
    for i, line in enumerate(lines):
        if 'template' in line:
            # remove spaces from template argument
            line = line.replace(' ', '')
            line = line.lstrip()
        if line.startswith('template<'):
            endcomment = i - 1
            startcomment = endcomment + 1
            for j in range(endcomment, 0, -1):
                if lines[j].startswith('//') or\
                   lines[j].startswith('/*') or\
                   lines[j].startswith(' *'):
                    startcomment = j
                else:
                    break
            comment = lines[startcomment:endcomment + 1]
            for s in ['/*', ' */', ' *', '//']:
                comment = [c.lstrip(s) for c in comment]
            comment = [c.lstrip() for c in comment]
            comment = '\n'.join(comment)

            # grab function name
            name = lines[i + 1]
            # if it's a function...
            if '(' in name:
                name = re.match('.*?(\w*)\(', name).group(1)
            # otherwise just take the last term
            else:
                name = name.split()[-1]
            comments[name] = comment

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
    fdef = func['template'] + '\n'
    fdef += func['returns'] + ' '
    fdef += '_' + func['name'] + '(\n'

    arraylist = []

    # find all parameters
    for p in func['parameters']:

        # skip "_size" parameters
        if '_size' in p['name']:
            continue

        const = '      '
        if p['constant']:
            const = 'const '

        paramtype = p['raw_type']
        if p['pointer'] or p['array']:
            param = 'py::array_t<%s> &' % paramtype
            param += ' ' + p['name']
            arraylist.append((const, paramtype, p['name']))
        else:
            param = paramtype
            param += ' ' + p['name']

        fdef += '     ' + param + ',\n'

    fdef = fdef.strip()[:-1] + ')'
    fdef += '\n{\n'

    # make a list of python objects
    for a in arraylist:
        if 'const' in a[0]:
            unchecked = '.unchecked();\n'
        else:
            unchecked = '.mutable_unchecked();\n'

        fdef += "auto py_" + a[2] + ' = ' + a[2] + unchecked

    # make a list of pointers to the arrays
    fdef += '\n'
    for a in arraylist:
        if 'const' in a[0]:
            data = '.data();\n'
        else:
            data = '.mutable_data();\n'
        fdef += a[0] + a[1] + ' *_' + a[2] + ' = py_' + a[2] + data

    # get the template signature
    template = func['template']
    template = template.replace('template', '').replace('class ', '')
    fdef += '\n return ' + func['name'] + template + '(\n'

    for p in func['parameters']:
        if '_size' in p['name']:
            fdef = fdef.strip()
            size = p['name'].replace('_size', '.size()')
            fdef += ' ' + size
        else:
            if p['pointer'] or p['array']:
                name = '_' + p['name']
            else:
                name = p['name']
            fdef += '     ' + name
        fdef += ',\n'
    fdef = fdef.strip()[:-1]
    fdef += ');\n}\n'
    return fdef


def build_plugin(headerfile, ch, comments, inst, remaps):
    """
    Take a header file (headerfile) and a parse tree (ch)
    and build the pybind11 plugin
    """
    headerfilename = headerfile.replace('.h', '')

    indent = '    '
    plugin = ''

    # plugin += '#define NC py::arg().noconvert()\n'
    # plugin += '#define YC py::arg()\n'
    plugin += 'PYBIND11_PLUGIN(%s) {\n' % headerfilename
    plugin += indent + 'py::module m("%s", R"pbdoc(\n' % headerfilename
    plugin += indent + 'pybind11 bindings for %s\n\n' % headerfile
    plugin += indent + 'Methods\n'
    plugin += indent + '-------\n'
    for f in ch.functions:
        for func in inst:
            if f['name'] in func['functions']:
                plugin += indent + f['name'] + '\n'
    plugin += indent + ')pbdoc");\n\n'

    # plugin += indent + 'py::options options;\n'
    # plugin += indent + 'options.disable_function_signatures();\n\n'

    for f in ch.functions:
        found = False
        for func in inst:
            if f['name'] in func['functions']:
                found = True
        if not found:
            continue

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

        types = []
        for func in inst:
            if f['name'] in func['functions']:
                types = func['types']
        if len(types) == 0:
            print('Could not find %s' % f['name'])

        ntypes = len(types)
        for i, t in enumerate(types):
            typestr = ', '.join(t)

            # get the instantiating function name

            # add the function call with each template
            instname = f['name']
            for remap in remaps:
                if f['name'] in remap:
                    instname = remap[f['name']]
            plugin += indent + 'm.def("%s", &_%s<%s>,\n' %\
                (instname, f['name'], typestr)

            # name the arguments
            pyargnames = []
            for p, array in argnames:
                convert = ''
                if array:
                    convert = '.noconvert()'
                pyargnames.append('py::arg("%s")%s' % (p, convert))

            argstring = indent + ', '.join(pyargnames)
            plugin += indent + argstring

            # add the docstring to the last
            if i == ntypes - 1:
                plugin += ',\nR"pbdoc(\n%s\n)pbdoc");\n' % comments[f['name']]
            else:
                plugin += ');\n'
        plugin += '\n'

    plugin += indent + 'return m.ptr();\n'
    plugin += '}\n'
    # plugin += '#undef NC\n'
    # plugin += '#undef YC\n'
    return plugin


def main():
    import argparse
    import CppHeaderParser

    parser = argparse.ArgumentParser(description='Wrap a C++ header with Pybind11')

    parser.add_argument("-o", "--output-file", metavar="FILE",
                        help="(default output name for header.h is header_bind.cpp)")

    parser.add_argument("input_file", metavar="FILE")

    args = parser.parse_args()

    print('[Generating %s from %s]' % (args.input_file.replace('.h', '_bind.cpp'), args.input_file))
    ch = CppHeaderParser.CppHeader(args.input_file)
    comments = find_comments(args.input_file)

    data = yaml.load(open('instantiate.yml', 'r'))
    inst = data['instantiate']
    remaps = data['remaps']
    plugin = build_plugin(args.input_file, ch, comments, inst, remaps)

    flist = []
    for f in ch.functions:

        # check to see if we should instantiate
        for func in inst:
            if f['name'] in func['functions']:
                print('\t[building {}]'.format(f['name']))
                fdef = build_function(f)
                flist.append(fdef)

    if args.output_file is not None:
        outf = args.output_file
    else:
        outf = args.input_file.replace('.h', '_bind.cpp')

    with open(outf, 'wt') as outf:

        print('// DO NOT EDIT: this file is generated\n', file=outf)
        print(PYBINDHEADER % args.input_file, file=outf)

        for f in flist:
            print(f, '\n\n\n', file=outf, sep="")

        print(plugin, file=outf)


if __name__ == '__main__':
    main()
