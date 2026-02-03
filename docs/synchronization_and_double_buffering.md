# SAGESim Synchronization and Double Buffering

## Overview

SAGESim uses a multi-level synchronization system and double buffering to ensure correct parallel execution across distributed workers (MPI ranks) and GPU threads.

---

## The Problem: Race Conditions in Parallel GPU Execution

### What Is a Race Condition?

A race condition occurs when multiple threads read and write the same memory location, and the final result depends on the unpredictable order of execution. GPUs execute thousands of threads simultaneously with no guaranteed ordering, making race conditions a critical concern.

### The Scenario in Agent-Based Simulation

Consider an SIR (Susceptible-Infected-Recovered) model where agents check their neighbors' infection status:

```
GPU Memory (single buffer scenario):
┌─────────────────────────────────────────┐
│ Agent 0: health = SUSCEPTIBLE           │
│ Agent 1: health = INFECTED              │
│ Agent 2: health = SUSCEPTIBLE           │
│ Agent 3: health = SUSCEPTIBLE           │
└─────────────────────────────────────────┘
```

**Step function logic:**
```python
def step(model, agent, health, neighbors):
    if health[agent] == SUSCEPTIBLE:
        for neighbor in neighbors[agent]:
            if health[neighbor] == INFECTED:
                health[agent] = INFECTED  # Write to shared memory!
                break
```

### The Race Condition (Without Double Buffering)

GPU launches thousands of threads simultaneously. Consider this execution:

```
Time    Thread 0 (Agent 0)              Thread 2 (Agent 2)
─────   ─────────────────────────────   ─────────────────────────────
T1      Read health[0] = SUSCEPTIBLE    Read health[2] = SUSCEPTIBLE
T2      Read health[1] = INFECTED       Read health[0] = ???
        (neighbor check)                (neighbor check)
T3      Write health[0] = INFECTED      What does Thread 2 see?
```

**The problem at T2:** When Thread 2 reads `health[0]`, what value does it see?

- If Thread 0 hasn't written yet → `SUSCEPTIBLE` (original value)
- If Thread 0 already wrote → `INFECTED` (new value)

This is **non-deterministic**. The simulation result depends on thread scheduling, which varies between runs.

### Why This Matters

```
Scenario A (Thread 0 writes first):
  Agent 0: SUSCEPTIBLE → INFECTED (from Agent 1)
  Agent 2: Sees Agent 0 as INFECTED → Agent 2 becomes INFECTED
  Result: 3 infected agents

Scenario B (Thread 2 reads first):
  Agent 0: SUSCEPTIBLE → INFECTED (from Agent 1)
  Agent 2: Sees Agent 0 as SUSCEPTIBLE → Agent 2 stays SUSCEPTIBLE
  Result: 2 infected agents
```

**Same initial state, same logic, different results.** This makes simulations irreproducible and scientifically invalid.

---

## The Solution: Double Buffering

### Concept

Separate memory into two buffers:
1. **Read Buffer**: The state at the START of the tick (frozen, never modified during tick)
2. **Write Buffer**: Where threads write their updates

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOUBLE BUFFERING                             │
├─────────────────────────────────┬───────────────────────────────────┤
│     READ BUFFER (frozen)        │      WRITE BUFFER (updates)       │
├─────────────────────────────────┼───────────────────────────────────┤
│ Agent 0: health = SUSCEPTIBLE   │ Agent 0: health = ?               │
│ Agent 1: health = INFECTED      │ Agent 1: health = ?               │
│ Agent 2: health = SUSCEPTIBLE   │ Agent 2: health = ?               │
│ Agent 3: health = SUSCEPTIBLE   │ Agent 3: health = ?               │
└─────────────────────────────────┴───────────────────────────────────┘
         ↑                                    ↑
    All threads READ here              All threads WRITE here
```

### Execution Flow

```
TICK START:
  Read Buffer  = [SUSCEPTIBLE, INFECTED, SUSCEPTIBLE, SUSCEPTIBLE]
  Write Buffer = [copy of read buffer]

PARALLEL EXECUTION (all threads run simultaneously):
  Thread 0: Reads neighbors from READ buffer
            Sees Agent 1 = INFECTED
            Writes INFECTED to WRITE buffer[0]

  Thread 2: Reads neighbors from READ buffer
            Sees Agent 0 = SUSCEPTIBLE  ← Always sees original value!
            Stays SUSCEPTIBLE
            Writes SUSCEPTIBLE to WRITE buffer[2]

TICK END:
  Copy Write Buffer → Read Buffer

NEXT TICK:
  Read Buffer  = [INFECTED, INFECTED, SUSCEPTIBLE, SUSCEPTIBLE]
  Now Agent 2 can see Agent 0's updated state
```

### Why This Works

Every thread sees a **consistent snapshot** of the simulation state at tick start:

| Thread | Reads From | Writes To | Sees Consistent State? |
|--------|------------|-----------|------------------------|
| 0 | Read Buffer | Write Buffer | Yes (tick-start snapshot) |
| 1 | Read Buffer | Write Buffer | Yes (tick-start snapshot) |
| 2 | Read Buffer | Write Buffer | Yes (tick-start snapshot) |
| N | Read Buffer | Write Buffer | Yes (tick-start snapshot) |

No thread can see another thread's in-progress writes because **writes go to a different buffer**.

---

## Visualization: Single Buffer vs Double Buffer

### Single Buffer (WRONG - Race Condition)

```
Time ──────────────────────────────────────────────────────────────►

Thread 0:  [Read A0=S]──[Read A1=I]──[Write A0=I]
                                          │
Thread 2:  [Read A2=S]──[Read A0=?]───────┼──────[Write A2=?]
                              │           │
                              └───────────┘
                              Race! May see S or I
```

### Double Buffer (CORRECT - Deterministic)

```
Time ──────────────────────────────────────────────────────────────►

           READ BUFFER                    WRITE BUFFER
           (frozen)                       (updates)
           ┌─────────┐                    ┌─────────┐
Thread 0:  │ A0=S    │──read──►compute──► │ A0=I    │
           │ A1=I    │                    │         │
           │ A2=S    │                    │         │
           └─────────┘                    └─────────┘

           ┌─────────┐                    ┌─────────┐
Thread 2:  │ A0=S    │──read──►compute──► │ A2=S    │
           │ A1=I    │    ↑               │         │
           │ A2=S    │    │               │         │
           └─────────┘    │               └─────────┘
                          │
                    Always sees S (original value)
                    because reads from frozen buffer
```

---

## Implementation in SAGESim

### Buffer Creation (`model.py:701-707`)

```python
# Create write buffers for properties that need them
write_buffers = []
for prop_idx in self._write_property_indices:
    # Create a copy of the tensor for writing
    write_buffer = cp.array(rank_local_agent_and_neighbor_adts[prop_idx])
    write_buffers.append(write_buffer)
```

Key points:
- Only properties that are **written to** get write buffers (detected via AST analysis)
- Write buffer starts as a **copy** of read buffer
- Separate GPU memory allocation

### Kernel Arguments (`model.py:709-717`)

```python
# Prepare all arguments: read tensors + write tensors
# Use converted locations for kernel, but keep original in rank_local_agent_and_neighbor_adts
all_args = []
for i, tensor in enumerate(rank_local_agent_and_neighbor_adts):
    if i == 1 and locations_for_kernel is not None:
        all_args.append(locations_for_kernel)  # Use converted indices for kernel
    else:
        all_args.append(tensor)  # Use original
all_args = all_args + write_buffers
```

The generated kernel receives both:
```python
def stepfunc(tick, global_data,
             a0, a1, a2, a3,           # Read tensors (properties 0-3)
             write_a0, write_a2):       # Write tensors (only written properties)
```

### Generated Kernel Code Transformation

The step function is transformed to use separate read/write arrays:

**User writes:**
```python
def step(model, agent, health, locations):
    if health[agent] == SUSCEPTIBLE:
        for neighbor in locations[agent]:
            if health[neighbor] == INFECTED:
                health[agent] = INFECTED
```

**Generated GPU kernel:**
```python
# Read from read buffer, write to write buffer
if read_health[agent_idx] == SUSCEPTIBLE:
    for n in range(max_neighbors):
        neighbor_idx = read_locations[agent_idx, n]
        if neighbor_idx < 0: break
        if read_health[neighbor_idx] == INFECTED:    # READ from read buffer
            write_health[agent_idx] = INFECTED        # WRITE to write buffer
            break
```

### Buffer Copy After Tick (`model.py:745-748`)

```python
# Copy write buffers back to read buffers after ALL priority groups complete
# This ensures all priorities execute with the same read buffer (state at start of tick)
for i, prop_idx in enumerate(self._write_property_indices):
    rank_local_agent_and_neighbor_adts[prop_idx][:len(self.__rank_local_agent_ids)] = \
        write_buffers[i][:len(self.__rank_local_agent_ids)]
```

Key points:
- Happens **after all threads complete** (after `synchronize()`)
- Only copies **local agents** (indices 0 to N-1), not neighbors
- Neighbors are read-only (owned by other workers)

---

## Why Copy Only Local Agents?

The GPU array contains both local and neighbor agents:

```
┌────────────────────────────────────────────────────────┐
│ Index 0-99:    Local agents (this worker owns these)  │
│ Index 100-150: Neighbor agents (other workers own)    │
└────────────────────────────────────────────────────────┘
```

Only local agents are copied because:

1. **Ownership**: Each worker is authoritative for its local agents only
2. **Read-only neighbors**: Neighbor data came from MPI; writing to it is meaningless
3. **Next tick sync**: Updated neighbor data will arrive via MPI next tick

```python
# Only copy local agents, not neighbors
read_buffer[:num_local_agents] = write_buffer[:num_local_agents]
#           ^^^^^^^^^^^^^^^^^
#           Indices 0 to N-1 only
```

---

## Priority System for Step Functions

### Overview

SAGESim allows **step functions** to be registered with different priorities. This enables:
1. **Ordered execution within a breed**: A single breed can have multiple step functions that run in sequence
2. **Ordered execution across breeds**: Different breeds can coordinate their execution order
3. **Grouped execution**: Step functions with the same priority run in parallel

### API: Registering Step Functions with Priorities

**Location:** `sagesim/breed.py:66-94`

```python
def register_step_func(
    self,
    step_func: Callable,
    module_fpath: str,
    priority: int = 0,
    no_double_buffer: Optional[List[str]] = None,
):
    """
    What the agent is supposed to do during a simulation step.

    :param step_func: The step function to execute
    :param module_fpath: Path to the module containing the step function
    :param priority: Execution priority (lower values execute first)
    :param no_double_buffer: List of property names that should NOT use double
        buffering in this step function. These properties will write directly
        to the read buffer, making changes visible to subsequent priorities
        within the same tick.
    """
    self._step_funcs[priority] = (step_func, module_fpath)

    # Accumulate no_double_buffer properties from all step functions
    if no_double_buffer:
        self._no_double_buffer_props.update(no_double_buffer)
```

### Example: Single Breed with Multiple Priorities

A breed can have multiple step functions at different priorities:

```python
class Neuron(Breed):
    def __init__(self):
        super().__init__("Neuron")
        self.register_property("voltage", default=0.0)
        self.register_property("fired", default=False)

        # Priority 0: Receive input from neighbors
        self.register_step_func(receive_input, __file__, priority=0)

        # Priority 1: Update internal state based on received input
        self.register_step_func(update_voltage, __file__, priority=1)

        # Priority 2: Fire if threshold reached
        self.register_step_func(check_fire, __file__, priority=2)
```

**Execution order within each tick:**
```
Priority 0: All neurons receive input (in parallel)
   ↓ GPU sync
Priority 1: All neurons update voltage (in parallel)
   ↓ GPU sync
Priority 2: All neurons check firing (in parallel)
   ↓ GPU sync
Buffer copy: write → read
```

### Example: Multiple Breeds with Step Functions at Various Priorities

Step function priorities are independent of breeds. A single breed can have multiple step functions at different priorities, and different breeds can share the same priority:

```python
# Prey moves first
prey_breed = Breed("Prey")
prey_breed.register_step_func(prey_move, __file__, priority=0)

# Predator hunts after prey moves
predator_breed = Breed("Predator")
predator_breed.register_step_func(predator_hunt, __file__, priority=1)

# Both breeds update energy at the same priority (run in parallel)
prey_breed.register_step_func(prey_update_energy, __file__, priority=2)
predator_breed.register_step_func(predator_update_energy, __file__, priority=2)
```

**Execution order:**
```
Priority 0: All Prey agents move (in parallel)
   ↓ GPU sync
Priority 1: All Predator agents hunt (in parallel)
   ↓ GPU sync
Priority 2: All Prey AND Predator update energy (in parallel, same priority group)
   ↓ GPU sync
Buffer copy: write → read
```

### Implementation: Priority Group Organization

**Location:** `sagesim/model.py:388-408`

During model setup, step functions are organized into priority groups using a heap:

```python
# Create record of agent step functions by breed and priority
self._breed_idx_2_step_func_by_priority: List[Dict[int, Callable]] = []
heap_priority_breedidx_func = []

# Collect all (priority, breed_idx, func) tuples
for breed in self._agent_factory.breeds:
    for priority, func in breed.step_funcs.items():
        heap_priority_breedidx_func.append((priority, (breed._breedidx, func)))

heapq.heapify(heap_priority_breedidx_func)

# Group by priority
last_priority = None
while heap_priority_breedidx_func:
    priority, breed_idx_func = heapq.heappop(heap_priority_breedidx_func)
    if last_priority == priority:
        # Same priority: add to current group
        self._breed_idx_2_step_func_by_priority[-1].update(
            {breed_idx_func[0]: breed_idx_func[1]}
        )
    else:
        # New priority: create new group
        self._breed_idx_2_step_func_by_priority.append(
            {breed_idx_func[0]: breed_idx_func[1]}
        )
        last_priority = priority
```

**Result:** A list of dictionaries, where each dictionary maps breed index to the step function for that priority. Step functions from different breeds (or multiple from the same breed) can share a priority group:

```
_breed_idx_2_step_func_by_priority = [
    {0: prey_move},                           # Priority 0
    {1: predator_hunt},                       # Priority 1
    {0: prey_update_energy, 1: predator_update_energy}  # Priority 2 (both breeds)
]
```

---

## Priority Execution and Double Buffering Interaction

### The Problem: Cross-Priority Data Visibility

When step functions at different priorities modify data, what should later priorities see?

```
Tick N:
  Priority 0 (Prey): Move to new location
  Priority 1 (Predator): Check prey location → Which location?
```

**Option A:** Predator sees prey's NEW location (mid-tick update)
**Option B:** Predator sees prey's OLD location (tick-start snapshot)

### SAGESim's Design Choice: Tick-Start Snapshot

All priorities within a tick see the **same state** (the state at tick start). This is achieved by:

1. All priorities **read from the read buffer** (tick-start state)
2. All priorities **write to the write buffer**
3. Buffer copy happens **after ALL priorities complete**

### Execution Flow with Priorities

**Location:** `sagesim/model.py:725-756`

```python
for tick_offset in range(sync_workers_every_n_ticks):
    current_tick = self.tick + tick_offset

    # Execute each priority group separately with synchronization
    for priority_idx, priority_group in enumerate(self._breed_idx_2_step_func_by_priority):
        # Execute step functions for this priority group
        self._step_func[blockspergrid, threadsperblock](
            current_tick,
            self._global_data_vector,
            *all_args,
            1,  # Process exactly 1 tick
            cp.float32(len(self.__rank_local_agent_ids)),
            rank_local_agent_and_non_local_neighbor_ids,
            priority_idx,  # Which priority group to execute
        )

        # Synchronization barrier: Wait for all blocks to complete this breed phase
        cp.cuda.Stream.null.synchronize()

    # Copy write buffers back to read buffers after ALL priority groups complete
    # This ensures all priorities execute with the same read buffer (state at start of tick)
    for i, prop_idx in enumerate(self._write_property_indices):
        rank_local_agent_and_neighbor_adts[prop_idx][:len(self.__rank_local_agent_ids)] = \
            write_buffers[i][:len(self.__rank_local_agent_ids)]

    # Final synchronization before next tick
    cp.cuda.Stream.null.synchronize()
```

### Visualization: Priority Execution Timeline

```
TICK N
│
├─► Priority 0 Launch ─────────────────────────────────►│
│   (All prey move, read from READ buffer)              │ GPU Sync
│                                                       │
├─► Priority 1 Launch ─────────────────────────────────►│
│   (All predators hunt, STILL read from READ buffer)   │ GPU Sync
│   (Sees prey's OLD location!)                         │
│                                                       │
├─► Priority 2 Launch ─────────────────────────────────►│
│   (All agents update energy, read from READ buffer)   │ GPU Sync
│                                                       │
├─► Buffer Copy ───────────────────────────────────────►│
│   (write_buffer → read_buffer)                        │ GPU Sync
│                                                       │
▼
TICK N+1
│
├─► Priority 0 Launch ─────────────────────────────────►
    (Now sees all changes from Tick N)
```

### Why GPU Sync Between Priorities?

The `cp.cuda.Stream.null.synchronize()` after each priority ensures:

1. **All threads complete** before the next priority starts
2. **Write buffer is consistent** - no partial writes from in-flight threads
3. **Generated kernel conditional** works correctly - it checks `priority_idx` to decide which step function to run

**Generated kernel code:**

```python
# In GPU kernel (simplified)
if current_priority_index == 0:
    if breed_id == 0:  # Prey
        prey_move(...)
elif current_priority_index == 1:
    if breed_id == 1:  # Predator
        predator_hunt(...)
elif current_priority_index == 2:
    if breed_id == 0:  # Prey
        prey_update_energy(...)
    elif breed_id == 1:  # Predator
        predator_update_energy(...)
```

Without GPU sync, a predator thread might execute `predator_hunt` while a prey thread is still executing `prey_move`, causing race conditions.

### Critical Design Decision: When to Copy Buffers

**Wrong approach (copy after each priority):**
```
Priority 0 (Prey): Move → Write new location to write_buffer
Buffer copy: write_buffer → read_buffer  ← WRONG!
Priority 1 (Predator): Sees prey's NEW location (inconsistent mid-tick state)
```

**Correct approach (copy after ALL priorities):**
```
Priority 0 (Prey): Move → Write new location to write_buffer
Priority 1 (Predator): Still reads from read_buffer (tick-start state)
Priority 2: Both breeds update energy (tick-start state)
Buffer copy: write_buffer → read_buffer  ← Only once per tick
```

**Why correct approach?**
- All agents within a tick see a **consistent snapshot** (tick-start state)
- No ordering dependencies within a single tick
- Deterministic: result doesn't depend on priority execution order for data visibility
- Priorities only control **when** step functions run, not **what data** they see

### Summary: Priority System Guarantees

| Guarantee | Description |
|-----------|-------------|
| **Execution Order** | Lower priority numbers execute first |
| **Parallel Within Priority** | All agents in a priority group run in parallel |
| **GPU Sync Between Priorities** | All threads complete before next priority starts |
| **Consistent Read State** | All priorities read from tick-start snapshot |
| **Single Buffer Copy** | Write → Read copy happens once per tick, after all priorities |

---

## Disabling Double Buffering: `no_double_buffer`

### When to Disable Double Buffering

By default, all written properties use double buffering for safety. However, there are cases where you want a later priority to see changes made by an earlier priority **within the same tick**:

```
Priority 1: Gap writes litter_accum
Priority 2: Site reads litter_accum → Needs to see the value written at P1!
```

With double buffering, Site would see the **start-of-tick** value, not the updated value. The `no_double_buffer` parameter solves this.

### API: Disabling Double Buffering

**Location:** `sagesim/breed.py:66-94`

```python
def register_step_func(
    self,
    step_func: Callable,
    module_fpath: str,
    priority: int = 0,
    no_double_buffer: Optional[List[str]] = None,
):
    """
    :param no_double_buffer: List of property names that should NOT use double
        buffering in this step function. These properties will write directly
        to the read buffer, making changes visible to subsequent priorities
        within the same tick.
    """
```

### Usage Example

```python
# Gap breed writes states at P1, Site reads at P2
gap_breed.register_step_func(
    gap_aggregate_step, __file__, priority=1,
    no_double_buffer=["params", "states"],  # Site needs to see these changes
)

# Site reads Gap.states at P2 - sees the updated value from P1!
site_breed.register_step_func(site_soil_step, __file__, priority=2)
```

### Implementation

**Location:** `sagesim/model.py:411-443`

During setup, SAGESim collects all `no_double_buffer` properties from all breeds and excludes them from write buffer creation:

```python
# Collect all no_double_buffer property names from all breeds
no_double_buffer_prop_names = set()
for breed in self._agent_factory.breeds:
    no_double_buffer_prop_names.update(breed.no_double_buffer_props)

# Convert property names to indices
no_double_buffer_indices = set()
for prop_name in no_double_buffer_prop_names:
    if prop_name in self._agent_factory._property_name_2_index:
        no_double_buffer_indices.add(
            self._agent_factory._property_name_2_index[prop_name]
        )

# Exclude no_double_buffer properties from write buffer creation
self._write_property_indices = self._write_property_indices - no_double_buffer_indices
```

### When to Use `no_double_buffer`

| Scenario | Use `no_double_buffer`? | Why |
|----------|------------------------|-----|
| Same-priority parallel reads | **NO** | Need buffering to prevent race conditions |
| Later priority reads earlier write (same tick) | **YES** | Must see updated value |
| Earlier priority reads later write | No (doesn't matter) | Reader runs first anyway |
| Cross-tick visibility is acceptable | No | Double buffering works fine |

### Trade-offs

| With Double Buffering | Without Double Buffering (`no_double_buffer`) |
|----------------------|----------------------------------------------|
| Consistent tick-start snapshot | Mid-tick visibility |
| Safe for same-priority reads | **UNSAFE** for same-priority reads |
| Extra memory for write buffer | Direct write to read buffer |
| Later priorities see old values | Later priorities see updated values |

---

## Synchronization Levels

### Level 1: MPI Worker Synchronization (CPU)

**Location**: `model.py` `simulate()` and `worker_coroutine()`

**Before simulation:**
```python
comm.barrier()  # All workers start together
```

**Neighbor data exchange:**
```python
contextualize_agent_data_tensors(
    local_agent_data,
    local_agent_ids,
    neighbor_requests
)
# MPI isend/irecv: exchange neighbor data between ranks
```

**After tick completion:**
```python
comm.allreduce(global_data_vector, op=reduce_func)
# Merge global properties across all workers
```

**Independence**: Workers execute GPU kernels independently on their agent subset. No inter-worker synchronization during kernel execution.

---

### Level 2: GPU Thread Synchronization (Within Worker)

**Location**: `model.py` `worker_coroutine()` lines 725-756

```python
for tick_offset in range(sync_workers_every_n_ticks):
    current_tick = self.tick + tick_offset

    # Execute each priority group separately with synchronization
    for priority_idx, priority_group in enumerate(self._breed_idx_2_step_func_by_priority):
        # Execute step functions for this priority group
        self._step_func[blockspergrid, threadsperblock](
            current_tick,
            self._global_data_vector,
            *all_args,
            1,  # Process exactly 1 tick
            cp.float32(len(self.__rank_local_agent_ids)),
            rank_local_agent_and_non_local_neighbor_ids,
            priority_idx,  # Which priority group to execute
        )

        # SYNC POINT 1: Wait for all GPU threads to finish this priority
        cp.cuda.Stream.null.synchronize()

    # Copy write buffers → read buffers (once per tick, after all priorities)
    for i, prop_idx in enumerate(self._write_property_indices):
        rank_local_agent_and_neighbor_adts[prop_idx][:len(self.__rank_local_agent_ids)] = \
            write_buffers[i][:len(self.__rank_local_agent_ids)]

    # SYNC POINT 2: Ensure copy completes before next tick
    cp.cuda.Stream.null.synchronize()
```

**Synchronization Points:**
1. After each priority group - ensures all threads finish before next priority
2. After buffer copy - ensures writes complete before next tick begins

---

## Complete Execution Order

```
1. INITIALIZATION (once per simulation)
   └─ Analyze step functions → determine write_property_indices
   └─ Pre-allocate GPU buffers

2. PER SIMULATION CALL:
   └─ MPI barrier (all workers sync)
   └─ For each time chunk:
       │
       ├─ CONTEXTUALIZE (MPI)
       │   └─ Send local agent data to workers who need it
       │   └─ Receive neighbor agent data from other workers
       │
       ├─ DATA PREPARATION (CPU → GPU)
       │   └─ Combine local + neighbor data
       │   └─ Convert agent IDs → array indices
       │   └─ Pad ragged arrays → rectangular
       │   └─ Upload to GPU memory
       │   └─ Create write buffer copies
       │
       ├─ GPU EXECUTION (per tick)
       │   └─ For each priority group:
       │       ├─ Launch GPU kernel (threads read from read_buffer)
       │       ├─ Threads write to write_buffer
       │       └─ GPU sync (wait for all threads)
       │   └─ Copy write_buffer → read_buffer
       │   └─ GPU sync (wait for copy)
       │
       ├─ DATA COLLECTION (GPU → CPU)
       │   └─ Download local agent data
       │   └─ Convert array indices → agent IDs
       │
       └─ MPI SYNC
           └─ Barrier (all workers finished GPU)
           └─ Allreduce (merge global properties)
```

---

## Summary: Single Buffer vs Double Buffer

| Aspect | Single Buffer | Double Buffer |
|--------|---------------|---------------|
| Thread reads | Shared, changing state | Frozen tick-start snapshot |
| Thread writes | Same location threads read | Separate buffer |
| Result | Non-deterministic (race) | Deterministic |
| Memory cost | 1x | 2x for written properties |
| Performance | Faster (no copy) | Slightly slower (buffer copy) |
| Correctness | Wrong | Correct |

The small overhead of double buffering (extra memory + one GPU-to-GPU copy per tick) is essential for **correctness** in parallel agent-based simulation.

---

## Code References

| Component | File | Lines |
|-----------|------|-------|
| Write property detection | `model.py` | 225-267 (`analyze_step_function_for_writes()`) |
| Double buffer creation | `model.py` | 701-707 |
| Kernel argument setup | `model.py` | 709-717 |
| Priority loop + sync | `model.py` | 725-756 |
| Buffer copy | `model.py` | 745-748 |
| no_double_buffer handling | `model.py` | 411-443 |
| GPU kernel generation | `model.py` | 820-1130 (`generate_gpu_func()`) |
| Priority group organization | `model.py` | 388-408 |
| Step function registration | `breed.py` | 66-94 |
| MPI contextualization | `agent.py` | 493-767 (`contextualize_agent_data_tensors()`) |
