from distutils.core import setup, Extension
import glob

c_sources = glob.glob('../c/pscgen/*.c') 
c_sources += ['../c/pscgen/linalg/linalg.c']
c_sources += ['pscgen_c.c']

pscgen_c = Extension('pscgen_c', sources=c_sources,
                   include_dirs=['../c/pscgen', '../c/pscgen/linalg'],
                   extra_link_args=['-lblas', '-llapack'])

setup(name='pscgen',
      version='0.1',
      description='Python wrapper for pscgen.',
      ext_modules=[pscgen_c],
      py_modules=['pscgen'])

# setup(name='pscgen',
#       version='0.1',
#       description='Nicer user interface for pscgen.',
#       py_modules=['pscgen'])
