import os
import importlib

# Dynamically load all handler modules
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py":
        module_name = f"handlers.{file[:-3]}"
        importlib.import_module(module_name)
