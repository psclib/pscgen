from distutils.core import setup, Extension
import glob

USE_OpenMP = '${USE_OpenMP}'

c_sources = glob.glob('../c/pscgen/*.c') 
c_sources += ['../c/pscgen/linalg/linalg.c']
c_sources += ['pscgen_c.c']

if USE_OpenMP == 'ON':
    extra_link_args = ['-lblas', '-llapack', '-lgomp']
    extra_compile_args = ['-fopenmp']
else:
    extra_link_args = ['-lblas', '-llapack']
    extra_compile_args = []

pscgen_c = Extension('pscgen_c', sources=c_sources,
                     include_dirs=['../c/pscgen', '../c/pscgen/linalg'],
                     extra_compile_args=extra_compile_args,
                     extra_link_args=extra_link_args)

setup(name='pscgen',
      version='0.1',
      description='Python wrapper for pscgen.',
      ext_modules=[pscgen_c],
      py_modules=['pscgen'])
