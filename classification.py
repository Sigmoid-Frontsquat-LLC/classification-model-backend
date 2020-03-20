import sys # this is for extracting command line arguments.

def parse_activator(flag, value):
    if flag[1] == 'a':
        return (True, value)
    else:
        return (False,None)
    pass

def parse_optimizer(flag, value):
    if flag[1] == 'o':
        return (True, value)
    else:
        return (False,None)
    pass
def parse_source(flag, value):
    if flag[1] == 's':
        return (True, value)
    else:
        return (False,None)
    pass

activator = ''
optimizer = ''
source = ''

if len(sys.argv) == 1 or (len(sys.argv) - 1) % 2 != 0:
    raise ValueError("Usage: -source [-a activator] [-o optimizer]")
else:
    # could this be done better?
    # sure, but this works for now...
    for i in range(1, len(sys.argv) - 1):
        flag = sys.argv[i]
        value = sys.argv[i + 1]

        isActivator, act = parse_activator(flag, value)

        if isActivator:
            activator = act
            continue

        isOptimizer, opt = parse_optimizer(flag, value)

        if isOptimizer:
            optimizer = opt
            continue
        isSource, so = parse_source(flag, value)

        if isSource:
            source = so
            continue
        pass
    pass

# naive check to ensure no argument is left unfilled
if len(activator) == 0 or len(optimizer) == 0 or len(source) or 0:
    raise ValueError("Usage: -source [-a activator] [-o optimizer]")

print("Hello, World!")
## each argument has been passed in, however
## we need to make sure they are valid.

# validate the 'activator'
pass
# validate the 'optimizer'
pass

# Load weights based on activator and optimizer

# Preprocess the image information

# Get the classification

# Print out the classification