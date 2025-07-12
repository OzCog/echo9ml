# psystem_membrane_architecture Module Flowchart

```mermaid
graph TD
    psystem_membrane_architecture[psystem_membrane_architecture]
    psystem_membrane_architecture_MembraneType[MembraneType]
    psystem_membrane_architecture --> psystem_membrane_architecture_MembraneType
    psystem_membrane_architecture_ObjectType[ObjectType]
    psystem_membrane_architecture --> psystem_membrane_architecture_ObjectType
    psystem_membrane_architecture_PermeabilityType[PermeabilityType]
    psystem_membrane_architecture --> psystem_membrane_architecture_PermeabilityType
    psystem_membrane_architecture_MembraneObject[MembraneObject]
    psystem_membrane_architecture --> psystem_membrane_architecture_MembraneObject
    psystem_membrane_architecture_MembraneObject___post_init__[__post_init__()]
    psystem_membrane_architecture_MembraneObject --> psystem_membrane_architecture_MembraneObject___post_init__
    psystem_membrane_architecture_MembraneObject_to_dict[to_dict()]
    psystem_membrane_architecture_MembraneObject --> psystem_membrane_architecture_MembraneObject_to_dict
    psystem_membrane_architecture_MembraneRule[MembraneRule]
    psystem_membrane_architecture --> psystem_membrane_architecture_MembraneRule
    psystem_membrane_architecture_MembraneRule___post_init__[__post_init__()]
    psystem_membrane_architecture_MembraneRule --> psystem_membrane_architecture_MembraneRule___post_init__
    psystem_membrane_architecture_MembranePermeability[MembranePermeability]
    psystem_membrane_architecture --> psystem_membrane_architecture_MembranePermeability
    psystem_membrane_architecture_CognitiveMembrane[CognitiveMembrane]
    psystem_membrane_architecture --> psystem_membrane_architecture_CognitiveMembrane
    psystem_membrane_architecture_CognitiveMembrane___init__[__init__()]
    psystem_membrane_architecture_CognitiveMembrane --> psystem_membrane_architecture_CognitiveMembrane___init__
    psystem_membrane_architecture_CognitiveMembrane_add_object[add_object()]
    psystem_membrane_architecture_CognitiveMembrane --> psystem_membrane_architecture_CognitiveMembrane_add_object
    psystem_membrane_architecture_CognitiveMembrane_remove_object[remove_object()]
    psystem_membrane_architecture_CognitiveMembrane --> psystem_membrane_architecture_CognitiveMembrane_remove_object
    psystem_membrane_architecture_CognitiveMembrane_add_rule[add_rule()]
    psystem_membrane_architecture_CognitiveMembrane --> psystem_membrane_architecture_CognitiveMembrane_add_rule
    psystem_membrane_architecture_CognitiveMembrane_execute_rules[execute_rules()]
    psystem_membrane_architecture_CognitiveMembrane --> psystem_membrane_architecture_CognitiveMembrane_execute_rules
    psystem_membrane_architecture_PSystemMembraneArchitecture[PSystemMembraneArchitecture]
    psystem_membrane_architecture --> psystem_membrane_architecture_PSystemMembraneArchitecture
    psystem_membrane_architecture_PSystemMembraneArchitecture___init__[__init__()]
    psystem_membrane_architecture_PSystemMembraneArchitecture --> psystem_membrane_architecture_PSystemMembraneArchitecture___init__
    psystem_membrane_architecture_PSystemMembraneArchitecture_create_membrane[create_membrane()]
    psystem_membrane_architecture_PSystemMembraneArchitecture --> psystem_membrane_architecture_PSystemMembraneArchitecture_create_membrane
    psystem_membrane_architecture_PSystemMembraneArchitecture_dissolve_membrane[dissolve_membrane()]
    psystem_membrane_architecture_PSystemMembraneArchitecture --> psystem_membrane_architecture_PSystemMembraneArchitecture_dissolve_membrane
    psystem_membrane_architecture_PSystemMembraneArchitecture_add_object_to_membrane[add_object_to_membrane()]
    psystem_membrane_architecture_PSystemMembraneArchitecture --> psystem_membrane_architecture_PSystemMembraneArchitecture_add_object_to_membrane
    psystem_membrane_architecture_PSystemMembraneArchitecture_transfer_object[transfer_object()]
    psystem_membrane_architecture_PSystemMembraneArchitecture --> psystem_membrane_architecture_PSystemMembraneArchitecture_transfer_object
    style psystem_membrane_architecture fill:#99ccff
```