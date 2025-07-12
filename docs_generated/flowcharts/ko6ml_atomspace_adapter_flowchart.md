# ko6ml_atomspace_adapter Module Flowchart

```mermaid
graph TD
    ko6ml_atomspace_adapter[ko6ml_atomspace_adapter]
    ko6ml_atomspace_adapter_Ko6mlPrimitiveType[Ko6mlPrimitiveType]
    ko6ml_atomspace_adapter --> ko6ml_atomspace_adapter_Ko6mlPrimitiveType
    ko6ml_atomspace_adapter_Ko6mlPrimitive[Ko6mlPrimitive]
    ko6ml_atomspace_adapter --> ko6ml_atomspace_adapter_Ko6mlPrimitive
    ko6ml_atomspace_adapter_Ko6mlPrimitive_to_scheme_expr[to_scheme_expr()]
    ko6ml_atomspace_adapter_Ko6mlPrimitive --> ko6ml_atomspace_adapter_Ko6mlPrimitive_to_scheme_expr
    ko6ml_atomspace_adapter_AtomSpaceFragment[AtomSpaceFragment]
    ko6ml_atomspace_adapter --> ko6ml_atomspace_adapter_AtomSpaceFragment
    ko6ml_atomspace_adapter_AtomSpaceFragment_to_dict[to_dict()]
    ko6ml_atomspace_adapter_AtomSpaceFragment --> ko6ml_atomspace_adapter_AtomSpaceFragment_to_dict
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter[Ko6mlAtomSpaceAdapter]
    ko6ml_atomspace_adapter --> ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter___init__[__init__()]
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter --> ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter___init__
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter__initialize_primitive_mappings[_initialize_primitive_mappings()]
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter --> ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter__initialize_primitive_mappings
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter__initialize_atomspace_mappings[_initialize_atomspace_mappings()]
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter --> ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter__initialize_atomspace_mappings
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter_ko6ml_to_atomspace[ko6ml_to_atomspace()]
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter --> ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter_ko6ml_to_atomspace
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter_atomspace_to_ko6ml[atomspace_to_ko6ml()]
    ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter --> ko6ml_atomspace_adapter_Ko6mlAtomSpaceAdapter_atomspace_to_ko6ml
    ko6ml_atomspace_adapter_create_ko6ml_adapter[create_ko6ml_adapter()]
    ko6ml_atomspace_adapter --> ko6ml_atomspace_adapter_create_ko6ml_adapter
    ko6ml_atomspace_adapter_create_test_primitives[create_test_primitives()]
    ko6ml_atomspace_adapter --> ko6ml_atomspace_adapter_create_test_primitives
    style ko6ml_atomspace_adapter fill:#99ccff
```