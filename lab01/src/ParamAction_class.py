import argparse  

class ParamAction(argparse.Action):  
    def __init__(self, option_strings, dest, **kwargs):  
        super(ParamAction, self).__init__(option_strings, dest, **kwargs)  
        self.was_provided = False  
  
    def __call__(self, parser, namespace, values, option_string=None):  
        setattr(namespace, self.dest, values)  
        self.was_provided = True 
        
    def was_provided(self):  
        return self.was_provided