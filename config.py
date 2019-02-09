import os
from configparser import ConfigParser
from utils import wrap_namespace
from ast import literal_eval as Eval


class Configuration(object):
    """
    Class to handle the config files for the project

    """
    def __init__(self,config_file_path):

        self.parser = ConfigParser()
        self.parser.optionxform = str
        try:
            with open(config_file_path) as f:
                self.parser.read_file(f)
        except IOError:
            raise IOError(f'Failed to open/find {config_file_path} , please check your path.')
    
    def get_config_options(self,section_names=None):
        """
        Get all the options of the specified sections. The dict is returned as 
        a nested namespace for its implementation, please see `utils.py` 
        
        This function also parses the values and converts them to their expected 
        types. This will work on most of the types, but can throw an error if the
        string is malformed. 
        Args:
            section_names (list): A list of strings that includes all the sections that need to be included
                
                default = None
            
            NOTE: If the section_names is None, all the options found in the
            config file will be used.
    
        Returns:
            namespace : containing all the options and their values
        """   
        section_options = dict()
        if section_names is None:
            sections = self.parser.sections()
        else:
            sections = section_names
        
        for section in sections:
            options = dict(self.parser.items(section))
            for k,v in options.items():
                options[k] = Eval(v)

            section_options[section] = options
        return wrap_namespace(section_options)
    
    def display(self):
        for section in self.parser.sections():
            print(f'\n{section}')
            print('==========')
            for k,v in self.parser.items(section):
                print (f'{k} : {v}')
            