import numpy as np

def getFactors(number):
    """
    Get all factors of a number.
    """
    factors = []
    for factor in range(1, number + 1):
        if number % factor == 0:
            factors.append(factor)
            
    return(np.array(factors))

def blockAveraging(data, dt):
    """
    Performs block averaging on a data set.
    Returns standard deviation and error for each block time value.
    """
    std = []
    
    # Make sure there are even number of data points
    if len(data) % 2 != 0:
        data = data[1:]
        
    N = len(data)
    totalTime = N * dt
    blocks = getFactors(N)
    for i in range(len(blocks)):
        data_reshaped = data.reshape(int(N/blocks[i]), blocks[i])
        data_blocked = np.mean(data_reshaped, axis=0)
        assert len(data_blocked) == blocks[i], 'Number of blocked data points has to be equal to the number of blocks'
        std.append(np.std(data_blocked, axis=0))
    
    std = np.array(std)
    
    err = std / np.sqrt(blocks)
    blockTime = totalTime / blocks
        
    return((blockTime, blocks, std, err))