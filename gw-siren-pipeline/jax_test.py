import jax
import jax.numpy as jnp

# It's good practice to enable float64 if your pipeline needs it,
# but test Metal functionality with float32 first as it's generally more robust.
# If float32 works, then try enabling float64.
# jax.config.update("jax_enable_x64", True)

print(f"JAX version: {jax.__version__}") # You mentioned this showed 0.5.0, please double check.
                                      # If it's truly 0.5.0, that's an unusual version.
                                      # Typical recent versions before your initial 0.6.1 were in the 0.4.x series.

print(f"JAX default backend: {jax.default_backend()}")
devices = jax.devices()
print(f"JAX devices: {devices}")

metal_devices = [d for d in devices if d.platform.lower() == 'metal']

if metal_devices:
    print(f"Successfully found Metal GPU device(s): {metal_devices}")
    
    # Use the first available Metal device
    target_device = metal_devices[0]
    print(f"Will attempt to use: {target_device}")

    try:
        key = jax.random.PRNGKey(0)
        
        # Create an array. JAX will attempt to place it on the default device (Metal).
        x = jax.random.normal(key, (500, 500), dtype=jnp.float32) 
        
        # How to check device of x:
        # The .device attribute should give you the device object directly.
        # If x.device is an object, printing it or its attributes is fine.
        # Do NOT call x.device() unless it's documented as a method for that JAX version.
        if hasattr(x, 'device'):
             print(f"x is on device: {x.device}") # Access as an attribute
        elif hasattr(x, 'device_buffer'): # Older JAX versions
             print(f"x is on device: {x.device_buffer.device()}") # Call method on device_buffer
        else:
             print("Could not determine device for x using common attributes.")

        # Perform a computation.
        y = jnp.dot(x, x) 
        y.block_until_ready() # Ensure computation finishes
        
        if hasattr(y, 'device'):
            print(f"y is on device: {y.device}")
        elif hasattr(y, 'device_buffer'):
            print(f"y is on device: {y.device_buffer.device()}")
        else:
            print("Could not determine device for y using common attributes.")
            
        print("Simple JAX computation (dot product) on Metal GPU successful.")
        print("Output sample (sum):", jnp.sum(y))

    except Exception as e:
        print(f"Error during test computation on Metal GPU: {e}")
        
else:
    print("Metal GPU device NOT found.")