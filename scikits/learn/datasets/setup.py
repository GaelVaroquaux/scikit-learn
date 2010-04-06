#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('datasets',parent_package,top_path)
    config.add_subpackage('samples_generator')
    config.add_data_dir('data')
    config.add_data_dir('descr')
    config.make_config_py() # installs __config__.py
    return config

if __name__ == '__main__':
    print 'This is the wrong setup.py file to run'
