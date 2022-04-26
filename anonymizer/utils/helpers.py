

def get_default_session_config(memory_fraction=None, **extra_config_kwargs):
    """ Returns default session configuration

    :param memory_fraction: percentage of the memory which should be kept free (growing is allowed).
    :return: tensorflow session configuration object
    """
    import tensorflow as tf
    
    gpu_options = tf.GPUOptions(allow_growth=True, allocator_type = 'BFC')
    
    if memory_fraction is not None:
        gpu_options.per_process_gpu_memory_fraction=memory_fraction
    
    options = {
        **dict(gpu_options=gpu_options,
               allow_soft_placement = True, 
               device_count={'GPU': 1}), 
        **extra_config_kwargs
    }
    
    return tf.ConfigProto(
        **options
    )
