def get_disp_network(name):
    if name == 'sfml':
        from .sfmlDispNet import DispResNet 
        return DispResNet()
    elif name == 'diffnet':
        from .diffDispNet import DispNet 
        return DispNet() 
    else:
        raise 'Unknown network name'