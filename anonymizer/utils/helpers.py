

def get_default_session_config(memory_fraction=None, use_cpu=False, **extra_config_kwargs):
    """ Returns default session configuration

    :param memory_fraction: percentage of the memory which should be kept free (growing is allowed).
    :return: tensorflow session configuration object
    """
    import tensorflow as tf
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    
    if memory_fraction is not None:
        gpu_options.per_process_gpu_memory_fraction=memory_fraction
    
    if use_cpu:
        gpu_options.allocator_type = 'BFC'
    
    
    options = {
        **dict(gpu_options=gpu_options,
               allow_soft_placement = use_cpu, 
               device_count={'GPU': 0 if use_cpu else 1}), 
        **extra_config_kwargs
    }
    
    return tf.ConfigProto(
        **options
    )
