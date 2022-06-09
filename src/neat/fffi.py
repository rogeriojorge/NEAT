#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:32:32 2017

@author: Christopher Albert <albert@alumni.tugraz.at>

Copied from https://github.com/pyccel/fffi
"""
import importlib
import os
import re
import sys
import subprocess
from pathlib import Path
from os.path import join, dirname
import numpy as np
import inspect

from cffi import FFI


if 'linux' in sys.platform:
    libexts = ['.so']
elif 'darwin' in sys.platform:
    libexts = ['.so', '.dylib']
elif 'win' in sys.platform:
    libexts = ['.pyd', '.dll', '.so']

LOG_WARN = True
LOG_DEBUG = False

def warn(output):
    caller_frame = inspect.currentframe().f_back
    (filename, line_number,
     function_name, _, _) = inspect.getframeinfo(caller_frame)
    filename = os.path.split(filename)[-1]
    print('')
    print('WARNING {}:{} {}():'.format(filename, line_number, function_name))
    print(output)


def debug(output):
    if not LOG_DEBUG:
        return
    caller_frame = inspect.currentframe().f_back
    (filename, line_number,
     function_name, _, _) = inspect.getframeinfo(caller_frame)
    filename = os.path.split(filename)[-1]
    print('')
    print('DEBUG {}:{} {}():'.format(filename, line_number, function_name))
    print(output)

class FortranLibrary:
    def __init__(self, name, maxdim=7, path=None, compiler=None):
        self.name = name
        self.maxdim = maxdim  # maximum dimension of arrays
        self.csource = ''
        self.loaded = False
        self.compiler = compiler
        self.methods = set()

        if isinstance(path, Path):
            self.path = path.__str__()
        else:
            self.path = path

        if self.path:
            self.libpath = self.path
        else:
            self.libpath = '.'

        libfile = None
        for libext in libexts:
            libfile = os.path.join(self.libpath, 'lib'+name+libext)
            print(libfile)
            if os.path.exists(libfile):
                break
            else:
                libfile = None
            
        if libfile is None:
            raise RuntimeError(
                f'Cannot find library {name} with extensions {libexts}')

        if self.compiler is None:
            libstrings = subprocess.check_output(
                ['strings', libfile]
            )
            libstrings = libstrings.decode('utf-8').split('\n')
            for line in libstrings:
                if line.startswith('GCC:') or line.startswith('@GCC:'):
                    debug(line)
                    major = int(line.split(')')[-1].split('.')[0])
                    self.compiler = {'name': 'gfortran', 'version': major}
                    debug(self.compiler)
                    break

        if self.compiler is None:  # fallback to recent gfortran
            self.compiler = {'name': 'gfortran', 'version': 9}

        # Manual path specification is required for tests via `setup.py test`
        # which would not find the extension module otherwise
        if self.path not in sys.path:
            sys.path.append(self.path)

    def __dir__(self):
        return sorted(self.methods)

    def __getattr__(self, attr):
        if ('methods' in self.__dict__) and (attr in self.methods):
            def method(*args):
                return call_fortran(
                    self._ffi, self._lib, attr, self.compiler, None, *args)
            return method
        raise AttributeError('''Fortran library \'{}\' has no routine
                                \'{}\'.'''.format(self.name, attr))

    def compile(self, tmpdir='.', verbose=0, debugflag=None,
                skiplib=False, extra_objects=None):
        """
        Compiles a Python extension as an interface for the Fortran module
        """
        ffi = FFI()

        extraargs = []
        if not verbose:
            extraargs.append('-Wno-implicit-function-declaration')

        extralinkargs = []
        if self.compiler['name'] in ('gfortran', 'ifort'):
            extralinkargs.append('-Wl,-rpath,'+self.libpath)
        if self.compiler['name'] == 'gfortran' and 'darwin' not in sys.platform:
            extralinkargs.append('-lgfortran')

        if self.path:
            target = os.path.join(self.path, '_'+self.name+libexts[0])
        else:
            target = './_'+self.name+libexts[0]

        structdef = arraydims(self.compiler)
        descr = arraydescr(self.compiler)
        for kdim in range(1, self.maxdim+1):
            structdef += descr.format(kdim)

        ffi.cdef(structdef+self.csource)

        if skiplib:
            if extra_objects is None:
                raise RuntimeError('Need extra_objects if skiplib=True')
            ffi.set_source('_'+self.name,
                           structdef+self.csource,
                           extra_compile_args=extraargs,
                           extra_link_args=extralinkargs,
                           extra_objects=extra_objects)
        else:
            ffi.set_source('_'+self.name,
                           structdef+self.csource,
                           libraries=[self.name],
                           library_dirs=['.', self.libpath],
                           extra_compile_args=extraargs,
                           extra_link_args=extralinkargs,
                           extra_objects=extra_objects)

        debug('Compilation starting')
        ffi.compile(tmpdir, verbose, target, debugflag)

    def load(self):
        """
        Loads the Fortran module using the generated Python extension.
        Attention: module cannot be re-/unloaded unless Python is restarted.
        """
        if self.loaded:
            # TODO: add a check if the extension module itself is loaded.
            # Otherwise a new instance of a FortranModule makes you think
            # you can reload the extension module without warning.
            warn('Library cannot be re-/unloaded unless Python is restarted.')

        self._mod = importlib.import_module('_'+self.name)
        self._ffi = self._mod.ffi
        self._lib = self._mod.lib

        self.methods = set()
        ext_methods = dir(self._lib)
        for m in ext_methods:
            if not m.endswith('_'):
                continue
            mname = m.strip('_')
            attr = getattr(self._lib, m)
            debug('Name: {}, Type: {}, Callable: {}'.format(
                mname, type(attr), callable(attr)))
            if callable(attr):  # subroutine or function
                self.methods.add(mname)

        self.loaded = True

    def cdef(self, csource):
        """
        Specifies C source with suffix template replacements
        """
        self.csource += csource
        debug('C signatures are\n' + self.csource)

    def fdef(self, fsource):
        ast = parse(fsource)
        csource = ccodegen(ast, module=False)
        self.cdef(csource)

    def new(self, typename, value=None):
        typelow = typename.lower()  # Case-insensitive Fortran

        # Basic types
        if typelow in ['integer', 'int', 'int32']:
            return self._ffi.new('int32_t*', value)
        if typelow in ['integer(8)', 'int64']:
            return self._ffi.new('int64_t*', value)
        if typelow in ['real', 'real(4)', 'float']:
            return self._ffi.new('float*', value)
        if typelow in ['real(8)', 'double']:
            return self._ffi.new('double*', value)

        # User-defined types
        if value is None:
            return self._ffi.new('struct {} *'.format(typename))
        raise NotImplementedError(
            'Cannot assign value to type {}'.format(typename))


class FortranModule:
    def __init__(self, library, name, maxdim=7, path=None, compiler=None):
        if isinstance(library, str):
            self.lib = FortranLibrary(library, maxdim, path, compiler)
        else:
            self.lib = library
        self.name = name
        self.methods = set()
        self.variables = set()
        self.csource = ''
        self.loaded = False

    def __dir__(self):
        return sorted(self.methods | self.variables)

    def __getattr__(self, attr):
        if ('methods' in self.__dict__) and (attr in self.methods):
            def method(*args):
                return call_fortran(self.lib._ffi, self.lib._lib,
                                    attr, self.lib.compiler, self.name, *args)
            return method
        if ('variables' in self.__dict__) and (attr in self.variables):
            return self.__get_var_fortran(attr)
        raise AttributeError('''Fortran module \'{}\' has no attribute
                                \'{}\'.'''.format(self.name, attr))

    def __setattr__(self, attr, value):
        if ('variables' in self.__dict__) and (attr in self.variables):
            if self.lib.compiler['name'] == 'gfortran':
                varname = '__'+self.name+'_MOD_'+attr
            elif self.lib.compiler['name'] == 'ifort':
                varname = self.name+'_mp_'+attr+'_'
            else:
                raise NotImplementedError(
                    '''Compiler {} not supported. Use gfortran or ifort
                    '''.format(self.compiler))
            setattr(self.lib._lib, varname, value)
        else:
            super(FortranModule, self).__setattr__(attr, value)

    def __get_var_fortran(self, var):
        """
        Returns a Fortran variable based on its name
        """
        if self.lib.compiler['name'] == 'gfortran':
            varname = '__'+self.name+'_MOD_'+var
        elif self.lib.compiler['name'] == 'ifort':
            varname = self.name+'_mp_'+var+'_'
        else:
            raise NotImplementedError(
                '''Compiler {} not supported. Use gfortran or ifort
                '''.format(self.lib.compiler))
        var = getattr(self.lib._lib, varname)

        if isinstance(var, self.lib._ffi.CData):  # array
            return fortran2numpy(self.lib._ffi, var)

        return var

    def cdef(self, csource):
        """
        Specifies C source with some template replacements:
        {mod} -> compiler module prefix, e.g. for self.name == testmod for GCC:
          void {mod}_func() -> void __testmod_MOD_func()
        """
        # GNU specific
        if self.lib.compiler['name'] == 'gfortran':
            self.csource += csource.format(mod='__'+self.name+'_MOD',
                                           suffix='')
        elif self.lib.compiler['name'] == 'ifort':
            self.csource += csource.format(mod=self.name+'_mp',
                                           suffix='_')
        else:
            raise NotImplementedError(
                '''Compiler {} not supported. Use gfortran or ifort
                '''.format(self.lib.compiler))
        debug('C signatures are\n' + self.csource)
        self.lib.csource = self.lib.csource + self.csource

    def fdef(self, fsource):
        ast = parse(fsource)
        csource = ccodegen(ast, module=True)
        self.cdef(csource)

    def load(self):
        if not self.lib.loaded:
            self.lib.load()
        self.methods = set()
        ext_methods = dir(self.lib._lib)
        for m in ext_methods:
            if self.lib.compiler['name'] == 'gfortran':
                mod_sym = '__{}_MOD_'.format(self.name)
            elif self.lib.compiler['name'] == 'ifort':
                mod_sym = '{}_mp_'.format(self.name)
            else:
                raise NotImplementedError(
                    '''Compiler {} not supported. Use gfortran or ifort
                    '''.format(self.compiler))
            if not mod_sym in m:
                continue
            mname = re.sub(mod_sym, '', m)
            if self.lib.compiler['name'] == 'ifort':
                mname = mname.strip('_')
            attr = getattr(self.lib._lib, m)
            debug('Name: {}, Type: {}, Callable: {}'.format(
                mname, type(attr), callable(attr)))
            if isinstance(attr, self.lib._ffi.CData):  # array variable
                self.variables.add(mname)
            elif callable(attr):  # subroutine or function
                self.methods.add(mname)
            else:  # scalar variable
                self.variables.add(mname)

    def compile(self, tmpdir='.', verbose=0, debugflag=None):
        self.lib.compile(tmpdir, verbose, debugflag)

    def new(self, typename):
        return self.lib.new(typename)

def arraydims(compiler):
    if compiler['name'] == 'gfortran':
        if compiler['version'] >= 8:  # TODO: check versions
            return """
              typedef struct array_dims array_dims;
              struct array_dims {
                ptrdiff_t stride;
                ptrdiff_t lower_bound;
                ptrdiff_t extent;
              };

              typedef struct datatype datatype;
              struct datatype {
                size_t len;
                int ver;
                signed char rank;
                signed char type;
                signed short attribute;
              };
            """
        # gfortran version < 8
        return """
            typedef struct array_dims array_dims;
            struct array_dims {
            ptrdiff_t stride;
            ptrdiff_t lower_bound;
            ptrdiff_t upper_bound;
            };
        """
    if compiler['name'] == 'ifort':
        return """
              typedef struct array_dims array_dims;
              struct array_dims {
                uintptr_t extent;
                uintptr_t distance;
                uintptr_t lower_bound;
              };
            """
    else:
        raise NotImplementedError(
            "Compiler {} not supported. Use gfortran or ifort".format(compiler))


def arraydescr(compiler):
    if compiler['name'] == 'gfortran':
        if compiler['version'] >= 8:
            return """
              typedef struct array_{0}d array_{0}d;
              struct array_{0}d {{
                void *base_addr;
                size_t offset;
                datatype dtype;
                ptrdiff_t span;
                struct array_dims dim[{0}];
              }};
            """
        return """
            typedef struct array_{0}d array_{0}d;
            struct array_{0}d {{
            void *base_addr;
            size_t offset;
            ptrdiff_t dtype;
            struct array_dims dim[{0}];
            }};
        """
    if compiler['name'] == 'ifort':
        return """
              typedef struct array_{0}d array_{0}d;
              struct array_{0}d {{
                void *base_addr;
                size_t elem_size;
                uintptr_t reserved;
                uintptr_t info;
                uintptr_t rank;
                uintptr_t reserved2;
                struct array_dims dim[{0}];
              }};
            """
    raise NotImplementedError(
        "Compiler {} not supported. Use gfortran or ifort".format(compiler))


ctypemap = {
    ('int', None): 'int32_t',
    ('real', None): 'float',
    ('complex', None): 'float _Complex',
    ('logical', None): '_Bool',
    ('str', None): 'char',
    ('int', 1): 'int8_t',
    ('int', 2): 'int16_t',
    ('int', 4): 'int32_t',
    ('int', 8): 'int64_t',
    ('real', 4): 'float',
    ('real', 8): 'double',
    ('complex', 4): 'float _Complex',
    ('complex', 8): 'double _Complex',
    ('logical', 4): '_Bool'
}

# Map for GCC datatypes
dtypemap = {
    1: 'integer',
    2: 'logical',
    3: 'real',
    4: 'complex'
}

def ccodegen(ast, module):
    """Generates C signature for Fortran subprogram.

    Parameters:
        ast: AST containing Fortran types, variables, subroutines and/or functions.
        module (bool): True if code is part of a module, False otherwise.

    Returns:
        str: C code for the according signature for type definitions and
             declaration of variables and functions.

    """
    csource = ''
    for typename, typedef in ast.types.items():  # types
        debug('Adding type {}'.format(typename))
        csource += 'struct {} {{{{\n'.format(typename)
        for decl in typedef.declarations:
            for var in decl.namespace.values():
                ctype, cdecl = c_declaration(var)
                debug('{} {}'.format(ctype, cdecl))
                csource += '{} {};\n'.format(ctype, cdecl)
        csource += '}};\n'

    for subname, subp in ast.subprograms.items():  # subprograms
        debug('Adding subprogram {}({})'.format(
            subname, ', '.join(subp.args)))
        csource += ccodegen_sub(subp, module)

    if module:  # module variables
        for var in ast.namespace.values():
            ctype, cdecl = c_declaration(var)
            csource += 'extern {} {{mod}}_{}{{suffix}};\n'.format(ctype, cdecl)

    return csource


def ccodegen_sub(subprogram, module=True):
    """Generates C function signature for Fortran subprogram.

    Parameters:
        subprogram: AST branch representing a Fortran subroutine or function.
        module (bool): True if subprogram is part of a module, False otherwise.

    Returns:
        str: C code for the according signature to call the subprogram.

    """
    cargs = []
    nstr = 0  # number of string arguments, for adding hidden length arguments
    for arg in subprogram.args:
        # TODO: add handling of more than 1D fixed size array arguments
        attrs = subprogram.namespace[arg]
        dtype = attrs.dtype
        rank = attrs.rank
        shape = attrs.shape
        precision = attrs.precision
        debug('{} rank={} bytes={}'.format(dtype, rank, precision))

        if dtype == 'str':
            nstr = nstr+1

        if dtype.startswith('type'):
            typename = dtype.split(' ')[1]
            ctypename = 'struct {}'.format(typename)
        elif rank == 0:
            ctypename = ctypemap[(dtype, precision)]
        else:
            if (shape is None
                or shape[0] is None
                    or shape[0][1] is None):  # Assumed size array
                ctypename = 'array_{}d'.format(rank)
            else:  # Fixed size array
                ctypename = ctypemap[(dtype, precision)]

        if ctypename is None:
            raise NotImplementedError('{} rank={}'.format(dtype, rank))

        cargs.append('{} *{}'.format(ctypename, arg))

    for kstr in range(nstr):
        cargs.append('size_t strlen{}'.format(kstr))

    if module:  # subroutine in module
        csource = 'extern void {{mod}}_{}{{suffix}}({});\n'.format(
            subprogram.name, ', '.join(cargs))
    else:  # global subroutine
        csource = 'extern void {}_({});\n'.format(
            subprogram.name, ', '.join(cargs))

    return csource


def c_declaration(var):
    # TODO: add support for derived types also here
    ctype = ctypemap[(var.dtype, var.precision)]

    # Scalars
    if var.rank == 0:
        debug('Adding scalar {} ({})'.format(var.name, ctype))
        return ctype, var.name.lower()

    # Assumed size arrays

    if var.shape is None or var.shape[0] is None or var.shape[0][1] is None:
        return 'array_{}d'.format(var.rank), var.name.lower()

    # Fixed size arrays

    if var.rank == 1:
        debug('Adding rank {} array {} ({})'.format(var.rank, var.name, ctype))
        length = var.shape[0][1]
        return ctype, '{}[{}]'.format(var.name.lower(), length)
    if var.rank > 1:
        raise NotImplementedError('''
           Fixed size arrays with rank > 1 not yet supported
           as module variables''')



def fortran2numpy(ffi, var):
    # See https://gist.github.com/yig/77667e676163bbfc6c44af02657618a6
    # TODO: add support for more types than real(8) and also assumed size

    vartype = ffi.typeof(var)

    # Fixed size array
    if vartype.kind == 'array':  # fixed size
        ctype = vartype.item.cname
        ptr = ffi.addressof(var)
        size = ffi.sizeof(var)
    elif vartype.kind == 'struct':
        ptr = var.base_addr
        dtype = dtypemap[var.dtype.type]
        if var.dtype.type == 4:  # complex has twice the bytes
            ctype = ctypemap[(dtype, var.dtype.len/2)]
        else:
            ctype = ctypemap[(dtype, var.dtype.len)]
        
        size = var.dtype.len*var.dim[0].extent # TODO: support >1D
        dtype = var.dtype.type
    else:
        raise NotImplementedError(f'''
        Array of kind {vartype.kind} not supported.
        ''')
    
    if ctype == 'double':
        return np.frombuffer(ffi.buffer(ptr, size), 'f8')
    elif ctype == 'double _Complex':
        return np.frombuffer(ffi.buffer(ptr, size), 'c16')
    raise NotImplementedError(f'''
        Array of type {ctype} not supported.
        ''')


def numpy2fortran(ffi, arr, compiler):
    """
    Converts Fortran-contiguous NumPy array arr into an array descriptor
    compatible with gfortran to be passed to library routines via cffi.
    """
    if not arr.flags.f_contiguous:
        raise TypeError('needs Fortran order in NumPy arrays')

    ndims = len(arr.shape)
    arrdata = ffi.new('array_{}d*'.format(ndims))
    arrdata.base_addr = ffi.cast('void*', arr.ctypes.data)
    if compiler['name'] == 'gfortran':
        arrdata.offset = 0
        if compiler['version'] >= 8:
            arrdata.span = np.size(arr)*arr.dtype.itemsize
            arrdata.dtype.len = arr.dtype.itemsize
            arrdata.dtype.ver = 0
            arrdata.dtype.rank = ndims
            arrdata.dtype.type = 3  # "3" for float, TODO:others
            arrdata.dtype.attribute = 0
        else:
            arrdata.dtype = ndims  # rank of the array
            arrdata.dtype = arrdata.dtype | (3 << 3)  # float: "3" TODO:others
            arrdata.dtype = arrdata.dtype | (arr.dtype.itemsize << 6)

        stride = 1
        for kd in range(ndims):
            arrdata.dim[kd].stride = stride
            arrdata.dim[kd].lower_bound = 1
            if compiler['version'] >= 8:
                arrdata.dim[kd].extent = arr.shape[kd]
            else:
                arrdata.dim[kd].upper_bound = arr.shape[kd]
            stride = stride*arr.shape[kd]
    elif compiler['name'] == 'ifort':
        arrdata.elem_size = arr.dtype.itemsize
        arrdata.reserved = 0
        arrdata.info = int('100001110', 2)
        arrdata.rank = ndims
        arrdata.reserved2 = 0
        distance = arr.dtype.itemsize
        for kd in range(ndims):
            arrdata.dim[kd].distance = distance
            arrdata.dim[kd].lower_bound = 1
            arrdata.dim[kd].extent = arr.shape[kd]
            distance = distance*arr.shape[kd]

    return arrdata


def call_fortran(ffi, lib, function, compiler, module, *args):
    """
    Calls a Fortran routine based on its name
    """
    # TODO: scalars should be able to be either mutable 0d numpy arrays
    # for in/out, or immutable Python types for pure input
    # TODO: should be able to cast variables e.g. int/float if needed
    cargs = []
    cextraargs = []
    for arg in args:
        if isinstance(arg, str):
            cargs.append(ffi.new("char[]", arg.encode('ascii')))
            cextraargs.append(len(arg))
        elif isinstance(arg, int):
            cargs.append(ffi.new('int32_t*', arg))
        elif isinstance(arg, float):
            cargs.append(ffi.new('double*', arg))
        elif isinstance(arg, np.ndarray):
            cargs.append(numpy2fortran(ffi, arg, compiler))
        else:  # TODO: add more basic types
            cargs.append(arg)

    if module is None:
        funcname = function + '_'
    else:
        if compiler['name'] == 'gfortran':
            funcname = '__'+module+'_MOD_'+function
        elif compiler['name'] == 'ifort':
            funcname = module+'_mp_'+function+'_'
        else:
            raise NotImplementedError(
                '''Compiler {} not supported. Use gfortran or ifort
                '''.format(compiler))
    func = getattr(lib, funcname)
    debug('Calling {}({})'.format(funcname, cargs))
    func(*(cargs + cextraargs))


class FortranWrapper:
    def __init__(self, name, csource, extra_objects):
        self.name = name
        self.target = './_' + self.name + libexts[0]
        self.ffi = FFI()
        self.ffi.set_source(name, csource, extra_objects=extra_objects)

    def compile(self, tmpdir='.', verbose=0, debugflag=None):
        self.ffi.compile(tmpdir, verbose, self.target, debugflag)

    def load(self):
        pass


from textx.metamodel import metamodel_from_file

# ==============================================================================
# TODO: integrate again with pyccel
#from pyccel.ast import Variable
#from pyccel.ast.datatypes import dtype_and_precsision_registry as dtype_registry

# Here are replacements to be compatible with pyccel


class Variable():
    def __init__(self,
                 dtype,
                 name,
                 rank=0,
                 allocatable=False,
                 is_stack_array=False,
                 is_pointer=False,
                 is_target=False,
                 is_polymorphic=None,
                 is_optional=None,
                 shape=None,
                 cls_base=None,
                 cls_parameters=None,
                 order='C',
                 precision=0):

        self.dtype = dtype
        self.name = name
        self.rank = rank
        self.allocatable = allocatable
        self.is_pointer = is_pointer
        self.is_target = is_target
        self.shape = shape
        self.precision = precision


# TODO: integrate again with pyccel
dtype_registry = {'real': ('real', 8),
                  'double': ('real', 8),
                  'double precision': ('real', 8),
                  'float': ('real', 8),
                  'float32': ('real', 4),
                  'float64': ('real', 8),
                  'complex': ('complex', 8),
                  'complex64': ('complex', 4),
                  'complex128': ('complex', 8),
                  'int8': ('int', 1),
                  'int16': ('int', 2),
                  'int32': ('int', 4),
                  'int64': ('int', 8),
                  'int': ('int', 4),
                  'integer': ('int', 4),
                  'logical': ('logical', 4),
                  'character': ('str', None)}

# ==============================================================================


class Fortran(object):
    """Class for Fortran syntax."""

    def __init__(self, *args, **kwargs):
        self.statements = kwargs.pop('statements', [])
        self.modules = {}
        self.subprograms = {}
        self.namespace = {}
        self.types = {}

        for stmt in self.statements:
            if isinstance(stmt, Module):
                self.modules[stmt.name] = stmt
                self.subprograms.update({sub.name:sub for sub in stmt.subprograms})
                self.namespace.update(stmt.namespace)
            elif isinstance(stmt, InternalSubprogram):
                self.subprograms[stmt.name] = stmt
            elif isinstance(stmt, Declaration):
                self.namespace = {**self.namespace, **stmt.namespace}
            elif isinstance(stmt, DerivedTypeDefinition):
                self.types[stmt.name] = stmt
            else:
                raise NotImplementedError('TODO {}'.format(type(stmt)))


class Module(object):
    """Class representing a Fortran module."""

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name')
        self.declarations = kwargs.pop('declarations', [])  # optional
        self.subprograms = kwargs.pop('subprograms', [])  # optional

        self.namespace = {}

        for decl in self.declarations:
            self.namespace = {**self.namespace, **decl.namespace}


class InternalSubprogram(object):
    """Class representing a Fortran internal subprogram."""

    def __init__(self, **kwargs):
        self.heading = kwargs.pop('heading')
        self.statements = kwargs.pop('statements', [])  # optional
        self.declarations = [stmt for stmt in self.statements if isinstance(stmt, Declaration)]
        self.name = self.heading.name
        self.args = self.heading.arguments
        self.namespace = {}

        for decl in self.declarations:
            self.namespace = {**self.namespace, **decl.namespace}


class SubprogramHeading(object):
    """Class representing a Fortran internal subprogram."""

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.arguments = kwargs.pop('arguments', [])  # optional


class SubprogramEnding(object):
    """Class representing a Fortran internal subprogram."""

    def __init__(self, **kwargs):
        self.end = kwargs.pop('name')


class InternalSubprogramHeading(object):
    """Class representing a Fortran internal subprogram."""

    def __init__(self, **kwargs):
        self.heading = kwargs.pop('heading')
        self.ending = kwargs.pop('ending')

class Stmt(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

class Declaration(object):
    """Class representing a Fortran declaration."""

    def __init__(self, **kwargs):
        self.dtype = kwargs.pop('dtype')
        self.attributes = kwargs.pop('attributes', [])  # this is optional
        self.entities = kwargs.pop('entities')
        self._build_namespace()

    def _build_namespace(self):
        self.namespace = {}
        # ...
        dtype = self.dtype.type

        if isinstance(dtype, DerivedType):
            dtype_type = 'type ' + dtype.name
            precision = None
        elif hasattr(dtype, 'kind') and dtype.kind:
            precision = dtype.kind
            dtype_type = dtype_registry[dtype.name.lower()][0]
        else:
            dtype_type, precision = dtype_registry[dtype.name.lower()]

        rank = 0
        shape = None
        attributes = []
        for i in self.attributes:
            key = i.key.lower()
            value = i.value

            if key == 'dimension':
                d_infos = value.expr
                shape = d_infos['shape']
                rank = len(shape)
            elif key == 'parameter':
                #we don't add a parameter to the namespace
                return
            attributes.append(key)

        is_allocatable = ('allocatable' in attributes)
        is_pointer = ('pointer' in attributes)
        is_target = ('target' in attributes)

        for ent in self.entities:
            localrank = rank
            arrayspec = ent.arrayspec
            if rank == 0 and (arrayspec is not None):
                localrank = len(arrayspec.expr['shape'])

            var = Variable(dtype_type,
                           ent.name,
                           rank=localrank,
                           allocatable=is_allocatable,
                           #                                is_stack_array = ,
                           is_pointer=is_pointer,
                           is_target=is_target,
                           #                                is_polymorphic=,
                           #                                is_optional=,
                           shape=shape,
                           #                                cls_base=None,
                           #                                cls_parameters=None,
                           #                                order=,
                           precision=precision
                           )
            self.namespace[ent.name] = var


class DeclarationAttribute(object):
    """Class representing a Fortran declaration attribute."""

    def __init__(self, **kwargs):
        self.key = kwargs.pop('key')
        self.value = kwargs.pop('value', None)  # this is optional


class DeclarationEntityObject(object):
    """Class representing an entity object ."""

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.arrayspec = kwargs.pop('arrayspec')


class DeclarationEntityFunction(object):
    """Class representing an entity function."""

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.charlen = kwargs.pop('charlen', None)
        raise NotImplementedError('')


class ArraySpec(object):
    """Class representing array spec."""

    def __init__(self, **kwargs):
        self.value = kwargs.pop('value')

    @property
    def expr(self):
        d_infos = {}

        shapes = []
        if isinstance(self.value, (list, tuple)):
            for i in self.value:
                if isinstance(i, ArrayExplicitShapeSpec):
                    lower = i.lower_bound
                    upper = i.upper_bound
                    shapes.append([lower, upper])

        else:
            raise NotImplementedError('')

        d_infos['shape'] = shapes
        return d_infos


class ArrayExplicitShapeSpec(object):
    """Class representing explicit array shape."""

    def __init__(self, **kwargs):
        self.upper_bound = kwargs.pop('upper_bound', None)
        self.lower_bound = kwargs.pop('lower_bound', 1)


class DerivedType(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')


class DerivedTypeDefinition(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.declarations = kwargs.pop('declarations', [])

# ==============================================================================

# ==============================================================================


def get_by_name(ast, name):
    """
    Returns an object from the AST by giving its name.
    """
    for token in ast.declarations:
        if token.name == name:
            return token
    return None

# ==============================================================================


def ast_to_dict(ast):
    """
    Returns an object from the AST by giving its name.
    """
    tokens = {}
    for token in ast.declarations:
        tokens[token.name] = token
    return tokens

# ==============================================================================


def parse(inputs, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, 'grammar.tx')

    classes = [Fortran,
               Module,
               InternalSubprogram,
            #    SubprogramHeading,
               Declaration,
               Stmt,
               DeclarationAttribute,
               DeclarationEntityObject,
               DeclarationEntityFunction,
               ArraySpec,
               ArrayExplicitShapeSpec,
               DerivedType,
               DerivedTypeDefinition]

    meta = metamodel_from_file(
        grammar, debug=debug, classes=classes, ignore_case=True)

    # Instantiate model
    if os.path.isfile(inputs):
        ast = meta.model_from_file(inputs)

    else:
        ast = meta.model_from_str(inputs)

    return ast
