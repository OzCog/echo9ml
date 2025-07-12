# deep_tree_echo Module Flowchart

```mermaid
graph TD
    deep_tree_echo[deep_tree_echo]
    deep_tree_echo_SpatialContext[SpatialContext]
    deep_tree_echo --> deep_tree_echo_SpatialContext
    deep_tree_echo_TreeNode[TreeNode]
    deep_tree_echo --> deep_tree_echo_TreeNode
    deep_tree_echo_TreeNode___post_init__[__post_init__()]
    deep_tree_echo_TreeNode --> deep_tree_echo_TreeNode___post_init__
    deep_tree_echo_DeepTreeEcho[DeepTreeEcho]
    deep_tree_echo --> deep_tree_echo_DeepTreeEcho
    deep_tree_echo_DeepTreeEcho___init__[__init__()]
    deep_tree_echo_DeepTreeEcho --> deep_tree_echo_DeepTreeEcho___init__
    deep_tree_echo_DeepTreeEcho_create_tree[create_tree()]
    deep_tree_echo_DeepTreeEcho --> deep_tree_echo_DeepTreeEcho_create_tree
    deep_tree_echo_DeepTreeEcho_add_child[add_child()]
    deep_tree_echo_DeepTreeEcho --> deep_tree_echo_DeepTreeEcho_add_child
    deep_tree_echo_DeepTreeEcho_add_child_with_spatial_context[add_child_with_spatial_context()]
    deep_tree_echo_DeepTreeEcho --> deep_tree_echo_DeepTreeEcho_add_child_with_spatial_context
    deep_tree_echo_DeepTreeEcho_calculate_echo_value[calculate_echo_value()]
    deep_tree_echo_DeepTreeEcho --> deep_tree_echo_DeepTreeEcho_calculate_echo_value
    style deep_tree_echo fill:#ff9999
```