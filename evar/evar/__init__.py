# EVAR
def import_module(module_name):
    try:
        module = __import__(module_name)
        return module
    except ModuleNotFoundError as e:
        print(f"Error: Module '{module_name}' not found.\n", e)
    except ImportError as e:
        print(f"Error: Unable to import module '{module_name}'.\n", e)
    return None


def import_object_from_module(module_name, object_name):
    try:
        module = __import__(module_name, fromlist=[object_name])
        obj = getattr(module, object_name)
        return obj
    except ModuleNotFoundError as e:
        print(f"Error: Module '{module_name}' not found..\n", e)
    except ImportError as e:
        print(f"Error: Unable to import object '{object_name}' from module '{module_name}'.\n", e)
    except AttributeError as e:
        print(f"Error: Object '{object_name}' not found in module '{module_name}'.\n", e)
    return None

# ar_stems = import_module('.ar_stems')
# ARStems = import_object_from_module('.ar_stems', 'ARStems')
try:
    from .ar_stems import ARStems
except ImportError as e:
    print(f"Error: Unable to import AR stems.\n", e)
try:
    from .ar_clap import ARCLAP
except ImportError as e:
    print(f"Error: Unable to import AR stems.\n", e)

try:
    from .ar_audiomae import ARAudioMAE
except (ModuleNotFoundError, ImportError) as e:
    print(e)
