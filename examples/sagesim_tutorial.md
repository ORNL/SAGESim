# **`sagesim` Implementation**

**SAGESim** (Scalable Agent-Based GPU-Enabled Simulator) is the first scalable, pure-Python, general-purpose agent-based modeling framework that supports both distributed computing and GPU acceleration. It is designed to run efficiently on modern high-performance computing (HPC) systems.

In this tutorial, we begin by explaining how data is stored and transferred across multiple GPUs in our distributed setup. We then walk through how to use **`sagesim`** to build and run large-scale agent-based simulations tailored to your specific problem domain. The core idea centers on subclassing the `Model` class provided by the `sagesim` library to define your custom model class.


## Distributed Computing and GPU Acceleration


For distributed execution, `sagesim` uses the Python-based Message Passing Interface library, **`mpi4py`**, to manage inter-process communication. In an MPI application, each process is assigned a unique *rank* and runs the same simulation code concurrently. However, each rank has its own local memory, meaning data (e.g., variables or objects) created in one process is inaccessible to others. As a result, communication and coordination between ranks must be explicitly handled through message passing.

Each rank is responsible for offloading its portion of the agent-based simulation to a dedicated GPU. Ideally, each rank independently simulates a subset of the global agent population using its assigned GPU. Details on running `sagesim` on an HPC system—including GPU assignment and job scheduling—are provided in a later section.

To simulate the behavior of its assigned agents, each rank requires the complete state information for two groups of agents: (1) the \$n\$ agents allocated to the rank itself, and (2) a set of \$m\$ additional agents that are neighbors of the \$n\$ agents but reside on other ranks. Although some neighbors may also be part of the same rank, we do not need to identify them separately since they are already included in the \$n\$ agents. Thus, each rank must have access to the state information of \$n + m\$ unique agents to correctly perform the simulation.

An **agent data tensor (ADT)** refers to a list containing a particular piece of information for a group of agents. In a single-process setup, each ADT covers the entire agent population; in a distributed multi-rank setting, it includes the local agents and their neighbors (of length $n + m$, as described earlier).

Each ADT holds a single info of agents (e.g., agent IDs, breeds, or locations), and is represented as a list of primitive types (e.g., `int`, `float`) or structured types (e.g., `list`, `nested lists`).  

In sagesim, we have ADTs include:

- `agent_ids`: `List[int]`  
  A list of unique agent IDs.

- `breeds`: `List[int]`  
  A list specifying the breed or type of each agent.

- `locations`: `List[List[int]]`  
  A list of neighbors for each agent, where each inner list contains the IDs of neighboring agents. Padding (e.g., `nan`) may be used to ensure uniform length.

- User-defined properties: `List[Any]`  
  Custom fields defined by the user. Each such property is its own ADT and can contain integers, floats, lists, or nested lists.


> **Note:** CuPy requires that all nested sublists have uniform length. This applies to fields like `locations`, where each entry is a list of neighboring agent IDs. To ensure consistency, shorter lists must be padded with `nan` values to match the longest list. For instance, if the maximum number of neighbors is 5, then an agent with only 3 neighbors will have a list like `[2, 4, 5, nan, nan]`. The same rule applies to any user-defined property structured as a list of lists per agent -- padding must be applied at each nesting level to maintain consistent dimensions. This padding is handled automatically during the `sagesim` setup process, so users generally do not need to manage it manually. However, it is important to be aware of this behavior: a single agent with a long sublist can substantially increase the memory footprint of the entire ADT. Additionally, since ADTs are passed into breed-specific `step` functions, users should understand the padding format when writing logic that accesses these values.



## Defining a Custom Model Class

Builiding a custom model class that subclasses the base `Model` class provided by `sagesim`, is the core part of using `sagesim`, as this enables access to the built-in `simulate()` method to execute your simulations. 

This model class is repsonsible for 
- **Define and Register Breeds**: Register the breed inside the model’s `__init__()` method using `register_breed()`.
- **Define and Register the Reduce Function**: If a reduce function is needed, register it in the model’s `__init__()` method using the `register_reduce_function()` method.
- **Register global properties**: If you have any global properties, they should be registered in the model class's `__init__()` method using `register_global_property()`.
- **Create class methods to create and connect Agents**
    - Specify which breed each agent belongs to and assign initial values for the agent's attributes using `create_agent_of_breed()`, which takes the breed object along with user-defined breed properties. It returns the unique ID of the newly created agent.
    - Connect two agents, that is, to create neighborship between two agents, using `connect_agents()`.

The example `Custom_Model` below demonstrates a model class with:

- Two breeds, each having two user-defined properties
- One global property
- A registered reduce function

````python

from sagesim.model import Model
from sagesim.space import NetworkSpace

class Custom_Model(Model):

    def __init__(self, global_property_value, **kwargs) -> None:
        space = NetworkSpace()
        super().__init__(space)
        # register your breeds
        self.Custom_Bread_1 = Custom_Bread_1
        self.Custom_Bread_2 = Custom_Bread_2
        self.register_breed(breed=self.Custom_Bread_1)
        self.register_breed(breed=self.Custom_Bread_2)
        # register your global property
        self.register_global_property("global_property_1", global_property_value)
        # register reduce function if needed
        self.register_reduce_function(reduce_function)

    def create_agent_1(self, breed_1_property_1, breed_1_property_2):
        agent_id = self.create_agent_of_breed(
            self.Custom_Bread_1, breed_1_property_1, breed_1_property_2
        )
        self.get_space().add_agent(agent_id)
        return agent_id
        
    def create_agent_2(self, breed_2_property_1, breed_2_property_2):
        agent_id = self.create_agent_of_breed(
            self.Custom_Bread_2, breed_2_property_1, breed_2_property_2
        )
        self.get_space().add_agent(agent_id)
        return agent_id
    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)
````


## Defining a Custom Model Class

Builiding a custom model class that subclasses the base `Model` class provided by `sagesim`, is the core part of using `sagesim`, as this enables access to the built-in `simulate()` method to execute your simulations. 

This model class is repsonsible for 
- **Define and Register Breeds**: Register the breed inside the model’s `__init__()` method using `register_breed()`.
- **Define and Register the Reduce Function**: If a reduce function is needed, register it in the model’s `__init__()` method using the `register_reduce_function()` method.
- **Register global properties**: If you have any global properties, they should be registered in the model class's `__init__()` method using `register_global_property()`.
- **Create class methods to create and connect Agents**
    - Specify which breed each agent belongs to and assign initial values for the agent's attributes using `create_agent_of_breed()`, which takes the breed object along with user-defined breed properties. It returns the unique ID of the newly created agent.
    - Connect two agents, that is, to create neighborship between two agents, using `connect_agents()`.

The example `Custom_Model` below demonstrates a model class with:

- Two breeds, each having two user-defined properties
- One global property
- A registered reduce function

````python

from sagesim.model import Model
from sagesim.space import NetworkSpace

class Custom_Model(Model):

    def __init__(self, global_property_value, **kwargs) -> None:
        space = NetworkSpace()
        super().__init__(space)
        # register your breeds
        self.Custom_Bread_1 = Custom_Bread_1
        self.Custom_Bread_2 = Custom_Bread_2
        self.register_breed(breed=self.Custom_Bread_1)
        self.register_breed(breed=self.Custom_Bread_2)
        # register your global property
        self.register_global_property("global_property_1", global_property_value)
        # register reduce function if needed
        self.register_reduce_function(reduce_function)

    def create_agent_1(self, breed_1_property_1, breed_1_property_2):
        agent_id = self.create_agent_of_breed(
            self.Custom_Bread_1, breed_1_property_1, breed_1_property_2
        )
        self.get_space().add_agent(agent_id)
        return agent_id
        
    def create_agent_2(self, breed_2_property_1, breed_2_property_2):
        agent_id = self.create_agent_of_breed(
            self.Custom_Bread_2, breed_2_property_1, breed_2_property_2
        )
        self.get_space().add_agent(agent_id)
        return agent_id
    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)
````


### 2. **The Reduce Function**

Whether a reduce function is required depends entirely on your model design.

- If your model ensures that, during each simulation step, agents **only update their own properties**—even if they read information from their neighbors—then a reduce function is **not needed**. This is a common design choice. In such cases, each rank owns a disjoint subset of agents (the $n$ agents on that rank), and no conflicting updates occur. To reconstruct the full state of the system, one simply collects the $n$ agents' data from all ranks.

- However, if your model allows an agent to **modify the properties of its neighbors**, and those neighbors may belong to **other ranks**, conflicts can arise. For example, if an agent updates properties of $m$ agents outside its own rank, then multiple ranks may end up with different versions (copies) of the same agent data. In such cases, a reduce function must be used to reconcile these conflicts. The reduction process takes place on the rank that owns each affected agent and ensures all updates are merged into a consistent version. This function is essential for maintaining **logical correctness** in distributed simulations where agents can modify the state of other agents across ranks.

````python
def reduce_agent_data_tensors_(adts_A, adts_B):
    """
    This function takes two agent data tensors (adts_A and adts_B) and reduces them into a single tensor.
    Parameters:
    ----------
    adts_A : list
        The first agent data tensor.
    adts_B : list
        The second agent data tensor.
    Returns:    
    -------
    list
        The reduced agent data tensor.
    """
````

### 3. CuPy Implementation: What It Means to You

`sagesim` uses a **CuPy** implementation to support both NVIDIA CUDA and AMD ROCm GPUs. However, there are important constraints when using **`cupyx.jit.rawkernel`**. Kernel code must be written using low-level Python functions, as many advanced Python features and abstractions are not supported. 

As a result, when implementing your own `step functions` and `reduce functions`, you must adhere to these limitations. Key restrictions include (but are not limited to):

- NaN checks must be done via inequality to self (e.g., `x != x`). This is an unfortunate limitation of `cupyx`.
- Dictionaries and custom Python objects are not supported.
- `*args` and `**kwargs` are unsupported.
- Nested function definitions are not allowed.
- Use **CuPy** data types and array routines instead of NumPy: [https://docs.cupy.dev/en/stable/reference/routines.html](https://docs.cupy.dev/en/stable/reference/routines.html)
- `for` loops must use the `range` iterator only — no `for-each` style loops.
- `return` statements do not behave reliably.
- `break` and `continue` statements are unsupported.
- Variables cannot be reassigned within `if` or `for` blocks. Declare and assign them at the top level or within new subscopes.
- Negative indexing (e.g., `array[-1]`) may not work as expected; it can access memory outside the logical bounds of the array. Use `len(array) - 1` instead.


