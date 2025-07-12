# ecan_attention_allocator Module Flowchart

```mermaid
graph TD
    ecan_attention_allocator[ecan_attention_allocator]
    ecan_attention_allocator_ResourceType[ResourceType]
    ecan_attention_allocator --> ecan_attention_allocator_ResourceType
    ecan_attention_allocator_TaskPriority[TaskPriority]
    ecan_attention_allocator --> ecan_attention_allocator_TaskPriority
    ecan_attention_allocator_ResourceBid[ResourceBid]
    ecan_attention_allocator --> ecan_attention_allocator_ResourceBid
    ecan_attention_allocator_AttentionAtom[AttentionAtom]
    ecan_attention_allocator --> ecan_attention_allocator_AttentionAtom
    ecan_attention_allocator_CognitiveTask[CognitiveTask]
    ecan_attention_allocator --> ecan_attention_allocator_CognitiveTask
    ecan_attention_allocator_ECANAttentionAllocator[ECANAttentionAllocator]
    ecan_attention_allocator --> ecan_attention_allocator_ECANAttentionAllocator
    ecan_attention_allocator_ECANAttentionAllocator___init__[__init__()]
    ecan_attention_allocator_ECANAttentionAllocator --> ecan_attention_allocator_ECANAttentionAllocator___init__
    ecan_attention_allocator_ECANAttentionAllocator_create_attention_atom[create_attention_atom()]
    ecan_attention_allocator_ECANAttentionAllocator --> ecan_attention_allocator_ECANAttentionAllocator_create_attention_atom
    ecan_attention_allocator_ECANAttentionAllocator_update_attention_values[update_attention_values()]
    ecan_attention_allocator_ECANAttentionAllocator --> ecan_attention_allocator_ECANAttentionAllocator_update_attention_values
    ecan_attention_allocator_ECANAttentionAllocator_bid_for_resources[bid_for_resources()]
    ecan_attention_allocator_ECANAttentionAllocator --> ecan_attention_allocator_ECANAttentionAllocator_bid_for_resources
    ecan_attention_allocator_ECANAttentionAllocator_process_resource_allocation[process_resource_allocation()]
    ecan_attention_allocator_ECANAttentionAllocator --> ecan_attention_allocator_ECANAttentionAllocator_process_resource_allocation
    ecan_attention_allocator_create_reasoning_task[create_reasoning_task()]
    ecan_attention_allocator --> ecan_attention_allocator_create_reasoning_task
    ecan_attention_allocator_create_memory_task[create_memory_task()]
    ecan_attention_allocator --> ecan_attention_allocator_create_memory_task
    ecan_attention_allocator_create_attention_task[create_attention_task()]
    ecan_attention_allocator --> ecan_attention_allocator_create_attention_task
    ecan_attention_allocator_create_learning_task[create_learning_task()]
    ecan_attention_allocator --> ecan_attention_allocator_create_learning_task
```