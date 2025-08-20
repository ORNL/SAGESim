import ast
import inspect
from typing import Callable, List, Set, Dict, Union, Tuple
from pathlib import Path


class StepFuncWrapper:
    """Wrapper function to handle double buffering for step functions."""
    
    def __init__(self, step_func: Callable):
        self.step_func = step_func
        self.original_signature = inspect.signature(step_func)
        self.write_property_indices = self._analyze_assignments()
    
    def _analyze_assignments(self) -> Set[int]:
        """
        Analyze the step function source code to find assignment operations
        to tensor properties that need write buffers.
        
        In the GPU function, properties are passed as a0, a1, a2, etc.
        We need to identify which property indices need write buffers.
        
        Returns:
            Set of property indices (integers) that need write buffers
        """
        write_property_indices = set()
        
        try:
            # Get the source code of the function
            source = inspect.getsource(self.step_func)
            
            # Parse the source code into an AST
            tree = ast.parse(source)
            
            # Walk through the AST to find assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Look for calls to set_this_agent_data_from_tensor
                    if (isinstance(node.func, ast.Name) and 
                        node.func.id == 'set_this_agent_data_from_tensor'):
                        
                        # The tensor parameter is the second argument (index 1)
                        if len(node.args) >= 2:
                            tensor_arg = node.args[1]
                            if isinstance(tensor_arg, ast.Name):
                                tensor_name = tensor_arg.id
                                # Extract property index from tensor name
                                # Tensor names in GPU func are like state_tensor, locations, etc.
                                # We need to map them to property indices
                                # For now, we'll use a heuristic based on parameter position
                                param_names = list(self.original_signature.parameters.keys())
                                if tensor_name in param_names:
                                    # Find the position of this tensor in parameters
                                    # Skip first few standard params (tick, agent_index, globals, agent_ids)
                                    standard_params = {'tick', 'agent_index', 'globals', 'agent_ids', 'breeds', 'locations'}
                                    property_params = [p for p in param_names if p not in standard_params]
                                    if tensor_name in property_params:
                                        property_index = property_params.index(tensor_name)
                                        write_property_indices.add(property_index)
        
        except (OSError, TypeError):
            # If we can't get the source, fall back to empty set
            pass
        
        return write_property_indices
    
    def get_write_property_indices(self) -> Set[int]:
        """
        Get the set of property indices that need write buffers.
        
        Returns:
            Set of property indices that need write buffers
        """
        return self.write_property_indices


def analyze_step_functions_for_write_buffers(
    breed_idx_2_step_func_by_priority: List[Dict[int, Tuple[Callable, str]]]
) -> Set[int]:
    """
    Analyze all step functions to determine which property indices need write buffers.
    
    Args:
        breed_idx_2_step_func_by_priority: List of dictionaries mapping breed index 
            to (step_func, module_path) tuples
    
    Returns:
        Set of property indices that need write buffers across all step functions
    """
    all_write_property_indices = set()
    
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
            breed_step_func_impl, module_fpath = breed_step_func_info
            wrapper = StepFuncWrapper(breed_step_func_impl)
            all_write_property_indices.update(wrapper.get_write_property_indices())
    return all_write_property_indices


def generate_modified_step_func_code(step_func: Callable) -> str:
    """
    Generate modified step function code that uses write buffers for assignments.
    
    This function:
    1. Analyzes the original step function to identify parameters needing write buffers
    2. Adds write buffer parameters to the signature
    3. Transforms assignment operations to use write buffers instead of read buffers
    
    Args:
        step_func: The original step function
        
    Returns:
        String containing the modified step function code
    """
    import inspect
    
    try:
        # Get the source code of the function
        source = inspect.getsource(step_func)
        lines = source.split('\n')
        
        # Parse the source code into an AST for analysis
        tree = ast.parse(source)
        
        # Get parameter names and identify which need write buffers
        wrapper = StepFuncWrapper(step_func)
        write_property_indices = wrapper.get_write_property_indices()
        
        # Get parameter names from the function signature
        param_names = list(wrapper.original_signature.parameters.keys())
        standard_params = {'tick', 'agent_index', 'globals', 'agent_ids', 'breeds', 'locations'}
        property_params = [p for p in param_names if p not in standard_params]
        
        # Create mapping from property parameter names to their indices
        param_to_write_param = {}
        for i, param_name in enumerate(property_params):
            if i in write_property_indices:
                param_to_write_param[param_name] = f"write_{param_name}"
        
        # Modify the source code
        modified_lines = []
        in_function_def = False
        signature_complete = False
        in_set_call = False
        set_call_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Handle function signature
            if 'def ' in line and step_func.__name__ in line:
                in_function_def = True
                modified_lines.append(line)
                continue
                
            if in_function_def and not signature_complete:
                # Handle signature modification
                if line.strip().endswith('):'):
                    # Check if the previous line ended with a comma (multi-line signature)
                    if modified_lines and modified_lines[-1].strip().endswith(','):
                        # Previous line has comma, we can add write parameters before closing
                        if param_to_write_param:
                            indent = len(line) - len(line.lstrip())
                            # Add write parameters
                            for param_name, write_param_name in param_to_write_param.items():
                                modified_lines.append(" " * indent + write_param_name + ",")
                            # Add the closing line
                            modified_lines.append(line)
                        else:
                            modified_lines.append(line)
                    else:
                        # Single line or last line without comma
                        if param_to_write_param:
                            # Need to add comma and write parameters
                            indent = len(line) - len(line.lstrip())
                            # Remove ): and add comma
                            base_line = line.rstrip()[:-2] + ","
                            modified_lines.append(base_line)
                            # Add write parameters  
                            for param_name, write_param_name in param_to_write_param.items():
                                modified_lines.append(" " * indent + write_param_name + ",")
                            # Add closing
                            modified_lines.append(" " * indent + "):")
                        else:
                            modified_lines.append(line)
                    
                    signature_complete = True
                    continue
                else:
                    # Check if this line is the last parameter (ends with comma)
                    if line.strip().endswith(',') and param_to_write_param:
                        # This is a parameter line, add it normally
                        modified_lines.append(line)
                    else:
                        modified_lines.append(line)
                    continue
            
            # Handle multi-line set_this_agent_data_from_tensor calls
            if 'set_this_agent_data_from_tensor(' in line:
                in_set_call = True
                set_call_lines = [line]
                continue
            elif in_set_call:
                set_call_lines.append(line)
                if ')' in line:
                    # End of set call, process all lines
                    in_set_call = False
                    full_call = '\n'.join(set_call_lines)
                    
                    # Transform the call
                    for param_name, write_param_name in param_to_write_param.items():
                        if f', {param_name},' in full_call:
                            full_call = full_call.replace(f', {param_name},', f', {write_param_name},')
                        elif f'({param_name},' in full_call:
                            full_call = full_call.replace(f'({param_name},', f'({write_param_name},')
                        elif f', {param_name} ' in full_call:
                            full_call = full_call.replace(f', {param_name} ', f', {write_param_name} ')
                    
                    # Add the modified call lines
                    modified_lines.extend(full_call.split('\n'))
                    set_call_lines = []
                continue
            
            modified_lines.append(line)
        
        return '\n'.join(modified_lines)
        
    except Exception as e:
        # If we can't modify, return original source with a comment
        try:
            original_source = inspect.getsource(step_func)
            return f"# Error modifying step function: {e}\n{original_source}"
        except:
            return f"# Could not get source for step function: {step_func.__name__}"


def extract_imports_from_file(file_path: str) -> List[str]:
    """
    Extract import statements from a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of import statement strings
    """
    imports = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the file to extract imports
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle: import module, import module as alias
                for alias in node.names:
                    if alias.asname:
                        imports.append(f"import {alias.name} as {alias.asname}")
                    else:
                        imports.append(f"import {alias.name}")
                        
            elif isinstance(node, ast.ImportFrom):
                # Handle: from module import name, from module import name as alias
                module = node.module or ''
                names = []
                for alias in node.names:
                    if alias.asname:
                        names.append(f"{alias.name} as {alias.asname}")
                    else:
                        names.append(alias.name)
                
                if len(names) == 1:
                    imports.append(f"from {module} import {names[0]}")
                else:
                    # Single line format for simplicity in generated code
                    imports.append(f"from {module} import {', '.join(names)}")
        
    except Exception as e:
        # If we can't parse imports, add common ones
        imports = [
            "import cupy as cp",
            "import random",
            "from sagesim.utils import (",
            "    get_this_agent_data_from_tensor,",
            "    set_this_agent_data_from_tensor,",
            "    get_neighbor_data_from_tensor,",
            ")"
        ]
    
    return imports


def create_double_buffered_step_func_file(step_func: Callable, original_module_path: str) -> str:
    """
    Create a separate file with the double buffered version of the step function.
    
    Args:
        step_func: The original step function
        original_module_path: Path to the original module containing the step function
        
    Returns:
        Path to the created double buffered step function file
    """
    # Generate modified step function code
    modified_code = generate_modified_step_func_code(step_func)
    
    # Create the new file path
    original_path = Path(original_module_path)
    new_file_path = original_path.parent / f"{original_path.stem}_double_buffered.py"
    
    # Create imports from the original file
    imports = [
        "import cupy as cp",
        "import random",
        "from cupyx import jit", 
        "from sagesim.utils import (",
        "    get_this_agent_data_from_tensor,",
        "    set_this_agent_data_from_tensor,", 
        "    get_neighbor_data_from_tensor,",
        ")",
        "",
        "# Modified step function with double buffering",
    ]
    
    # Write the file
    with open(new_file_path, 'w') as f:
        f.write('\n'.join(imports))
        f.write('\n\n')
        f.write(modified_code)
    
    return str(new_file_path)


def generate_gpu_func_with_double_buffer(
    n_properties: int,
    breed_idx_2_step_func_by_priority: List[Dict[int, Tuple[Callable, str]]],
) -> str:
    """
    Generate GPU function string with double buffering support.
    
    This is a modified version of the original generate_gpu_func that adds
    write buffer parameters for properties that have assignment operations.
    
    Args:
        n_properties: Total number of agent properties
        breed_idx_2_step_func_by_priority: List of dictionaries mapping breed index 
            to (step_func, module_path) tuples, ordered by execution priority
    
    Returns:
        String representation of GPU stepfunc with double buffering
    """
    # Analyze which properties need write buffers
    write_property_indices = analyze_step_functions_for_write_buffers(
        breed_idx_2_step_func_by_priority
    )
    
    # Generate read arguments (original properties)
    read_args = [f"a{i}" for i in range(n_properties)]
    
    # Generate write arguments for properties that need write buffers
    write_args = [f"write_a{i}" for i in sorted(write_property_indices)]
    
    # Combine all arguments
    all_args = read_args + write_args
    
    sim_loop = []
    modified_step_functions = []
    all_imports = set()
    processed_files = set()
    
    # First pass: collect all imports and generate modified step functions
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
            breed_step_func_impl, module_fpath = breed_step_func_info
            step_func_name = getattr(breed_step_func_impl, "__name__", repr(callable))
            modified_step_func_name = f"{step_func_name}_double_buffer"
            module_fpath = Path(module_fpath).absolute()
            
            # Extract imports from the original file
            if str(module_fpath) not in processed_files:
                file_imports = extract_imports_from_file(str(module_fpath))
                all_imports.update(file_imports)
                processed_files.add(str(module_fpath))
            
            # Generate the modified step function code
            modified_step_func_code = generate_modified_step_func_code(breed_step_func_impl)
            # Replace the function name in the modified code
            modified_step_func_code = modified_step_func_code.replace(
                f"def {step_func_name}(",
                f"def {modified_step_func_name}("
            )
            modified_step_functions.append(modified_step_func_code)
            
            # Generate the step function call with both read and write arguments
            sim_loop += [
                f"if breed_id == {breedidx}:",
                f"\t{modified_step_func_name}(",
                "\t\tthread_local_tick,",
                "\t\tagent_index,",
                "\t\tdevice_global_data_vector,",
                "\t\tagent_ids,",
                f"\t\t{','.join(all_args)},",
                "\t)",
            ]
    
    # Create import section - sort and clean imports
    import_lines = []
    basic_imports = []
    from_imports = []
    
    # Remove duplicates while preserving order
    seen = set()
    for imp in sorted(all_imports):
        if imp not in seen:
            seen.add(imp)
            if imp.startswith('import '):
                basic_imports.append(imp)
            elif imp.startswith('from '):
                from_imports.append(imp)
    
    import_lines = basic_imports + from_imports
    
    step_sources = "\n".join(import_lines)
    
    # Add modified step functions to the code
    all_modified_step_functions = "\n\n".join(modified_step_functions)
    
    # Preprocess parts that would break in f-strings
    joined_sim_loop = "\n\t\t\t".join(sim_loop)
    joined_args = ",".join(all_args)
    
    func = [
        "from cupyx import jit",
        step_sources,
        "",
        "# Modified step functions with double buffering",
        all_modified_step_functions,
        "",
        "@jit.rawkernel(device='cuda')",
        "def stepfunc(",
        "global_tick,",
        "device_global_data_vector,",
        joined_args + ",",
        "sync_workers_every_n_ticks,",
        "num_rank_local_agents,",
        "agent_ids,",
        "):",
        "\tthread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x",
        "\tagent_index = thread_id",
        "\tif agent_index < num_rank_local_agents:",
        "\t\tbreed_id = a0[agent_index]",
        "\t\tfor tick in range(sync_workers_every_n_ticks):",
        f"\n\t\t\tthread_local_tick = int(global_tick) + tick",
        f"\n\t\t\t{joined_sim_loop}",
    ]
    
    func = "\n".join(func)
    return func