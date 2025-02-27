
    
from random import random
    
from cupyx import jit
    



@jit.rawkernel(device='cuda')
def step_func(id, id2index, globals, breeds, locations, states):
    #agent_index = step_func_helper_get_agent_index(id, id2index)
    # nan checked by inequality to self. Unfortunate limitation of cupyx
    if (id == id) and (id2index[int(id)] == id2index[int(id)]):
        agent_index = int(id2index[int(id)])
         
        neighbors = locations[agent_index]  # network location is defined by neighbors
        rand = random()#0.1#step_func_helper_get_random_float(rng_states, id)
        
        p_infection = globals[1]
        
        for i in range(len(neighbors)):
            neighbor_id = neighbors[i]                
            #neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
            if (neighbor_id == neighbor_id) and (id2index[int(neighbor_id)] == id2index[int(neighbor_id)]):
                    neighbor_index = int(id2index[int(neighbor_id)])
                    neighbor_state = states[neighbor_index]
                    if neighbor_state == 2 and rand < p_infection:
                        states[agent_index] = 2.0

    

@jit.rawkernel()
def stepfunc(
    device_global_data_vector,
    a0,a1,a2,
    sync_workers_every_n_ticks,
    agent_ids,
    agents_index_in_subcontext,
    ):
        thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        #g = cuda.cg.this_grid()
        agent_id = thread_id        
        if agent_id < agent_ids.shape[0]:
            breed_id = a0[agent_id]                
            for tick in range(sync_workers_every_n_ticks):
                
            
                if breed_id == 0:
                    step_func(
                        agent_id,
                        agents_index_in_subcontext,
                        device_global_data_vector,
                        a0,a1,a2,
                    )
            #cuda.syncthreads()

                            
                if thread_id == 0:
                    device_global_data_vector[0] += 1
    