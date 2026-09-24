"""Query Vulkan requirements observed in the actual env_airsim_16 failures.

Read-only capability admission, not a rendering correctness or flight test.
Uses the process-selected ICD and the system Vulkan loader; no driver patching.
"""
import ctypes as C


class InstanceInfo(C.Structure):
    _fields_=[("sType",C.c_uint32),("pNext",C.c_void_p),("flags",C.c_uint32),
              ("application",C.c_void_p),("layer_count",C.c_uint32),("layers",C.c_void_p),
              ("extension_count",C.c_uint32),("extensions",C.c_void_p)]


class FormatProperties(C.Structure):
    _fields_=[("linear",C.c_uint32),("optimal",C.c_uint32),("buffer",C.c_uint32)]


def probe():
    library=C.CDLL("libvulkan.so.1")
    library.vkCreateInstance.argtypes=[C.POINTER(InstanceInfo),C.c_void_p,C.POINTER(C.c_void_p)]
    library.vkCreateInstance.restype=C.c_int32
    library.vkDestroyInstance.argtypes=[C.c_void_p,C.c_void_p]
    library.vkDestroyInstance.restype=None
    library.vkEnumeratePhysicalDevices.argtypes=[C.c_void_p,C.POINTER(C.c_uint32),C.POINTER(C.c_void_p)]
    library.vkEnumeratePhysicalDevices.restype=C.c_int32
    library.vkGetPhysicalDeviceFeatures.argtypes=[C.c_void_p,C.c_void_p]
    library.vkGetPhysicalDeviceFeatures.restype=None
    library.vkGetPhysicalDeviceFormatProperties.argtypes=[C.c_void_p,C.c_uint32,C.POINTER(FormatProperties)]
    library.vkGetPhysicalDeviceFormatProperties.restype=None
    instance=C.c_void_p()
    code=library.vkCreateInstance(C.byref(InstanceInfo(sType=1)),None,C.byref(instance))
    if code:
        raise RuntimeError(f"vkCreateInstance failed: {code}")
    try:
        count=C.c_uint32()
        code=library.vkEnumeratePhysicalDevices(instance,C.byref(count),None)
        if code or not count.value:
            raise RuntimeError(f"No usable Vulkan device: {code}")
        handles=(C.c_void_p*count.value)()
        code=library.vkEnumeratePhysicalDevices(instance,C.byref(count),handles)
        if code:
            raise RuntimeError(f"Vulkan device enumeration failed: {code}")
        devices=[]
        for handle in handles:
            # Vulkan 1.0 VkPhysicalDeviceFeatures is 55 VkBool32 fields.
            features=(C.c_uint32*55)()
            library.vkGetPhysicalDeviceFeatures(handle,features)
            properties=FormatProperties()
            library.vkGetPhysicalDeviceFormatProperties(handle,15,C.byref(properties)) # VK_FORMAT_R8_SRGB
            missing=[]
            if not properties.optimal & 1: # VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
                missing.append("VK_FORMAT_R8_SRGB optimal sampled image")
            if not properties.optimal & 0x1000: # SAMPLED_IMAGE_FILTER_LINEAR_BIT
                missing.append("VK_FORMAT_R8_SRGB linear filtering")
            if not features[4]:
                missing.append("geometryShader")
            devices.append(dict(index=len(devices),geometry_shader=bool(features[4]),
                                r8_srgb_optimal_features=properties.optimal,missing=missing))
        return dict(status="capabilities_available" if all(not x["missing"] for x in devices) else "blocked_graphics_capabilities",
                    scope="Required features observed in selected-scene validation traces; not full renderer validation",
                    devices=devices)
    finally:
        library.vkDestroyInstance(instance,None)


if __name__=="__main__":
    import json
    print(json.dumps(probe(),indent=2))
