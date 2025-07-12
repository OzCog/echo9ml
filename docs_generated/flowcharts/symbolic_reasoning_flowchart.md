# symbolic_reasoning Module Flowchart

```mermaid
graph TD
    symbolic_reasoning[symbolic_reasoning]
    symbolic_reasoning_LogicalOperator[LogicalOperator]
    symbolic_reasoning --> symbolic_reasoning_LogicalOperator
    symbolic_reasoning_TruthValue[TruthValue]
    symbolic_reasoning --> symbolic_reasoning_TruthValue
    symbolic_reasoning_TruthValue___new__[__new__()]
    symbolic_reasoning_TruthValue --> symbolic_reasoning_TruthValue___new__
    symbolic_reasoning_TruthValue___str__[__str__()]
    symbolic_reasoning_TruthValue --> symbolic_reasoning_TruthValue___str__
    symbolic_reasoning_TruthValue_to_dict[to_dict()]
    symbolic_reasoning_TruthValue --> symbolic_reasoning_TruthValue_to_dict
    symbolic_reasoning_Atom[Atom]
    symbolic_reasoning --> symbolic_reasoning_Atom
    symbolic_reasoning_Atom___hash__[__hash__()]
    symbolic_reasoning_Atom --> symbolic_reasoning_Atom___hash__
    symbolic_reasoning_Atom___eq__[__eq__()]
    symbolic_reasoning_Atom --> symbolic_reasoning_Atom___eq__
    symbolic_reasoning_Atom_to_dict[to_dict()]
    symbolic_reasoning_Atom --> symbolic_reasoning_Atom_to_dict
    symbolic_reasoning_Atom_from_dict[from_dict()]
    symbolic_reasoning_Atom --> symbolic_reasoning_Atom_from_dict
    symbolic_reasoning_Link[Link]
    symbolic_reasoning --> symbolic_reasoning_Link
    symbolic_reasoning_Link___hash__[__hash__()]
    symbolic_reasoning_Link --> symbolic_reasoning_Link___hash__
    symbolic_reasoning_Link___eq__[__eq__()]
    symbolic_reasoning_Link --> symbolic_reasoning_Link___eq__
    symbolic_reasoning_Link_to_dict[to_dict()]
    symbolic_reasoning_Link --> symbolic_reasoning_Link_to_dict
    symbolic_reasoning_Link_from_dict[from_dict()]
    symbolic_reasoning_Link --> symbolic_reasoning_Link_from_dict
    symbolic_reasoning_Pattern[Pattern]
    symbolic_reasoning --> symbolic_reasoning_Pattern
    symbolic_reasoning_Pattern_matches[matches()]
    symbolic_reasoning_Pattern --> symbolic_reasoning_Pattern_matches
    symbolic_reasoning_Pattern_to_dict[to_dict()]
    symbolic_reasoning_Pattern --> symbolic_reasoning_Pattern_to_dict
    symbolic_reasoning_Rule[Rule]
    symbolic_reasoning --> symbolic_reasoning_Rule
    symbolic_reasoning_Rule_to_dict[to_dict()]
    symbolic_reasoning_Rule --> symbolic_reasoning_Rule_to_dict
    symbolic_reasoning_SymbolicAtomSpace[SymbolicAtomSpace]
    symbolic_reasoning --> symbolic_reasoning_SymbolicAtomSpace
    symbolic_reasoning_SymbolicAtomSpace___init__[__init__()]
    symbolic_reasoning_SymbolicAtomSpace --> symbolic_reasoning_SymbolicAtomSpace___init__
    symbolic_reasoning_SymbolicAtomSpace__initialize_basic_patterns[_initialize_basic_patterns()]
    symbolic_reasoning_SymbolicAtomSpace --> symbolic_reasoning_SymbolicAtomSpace__initialize_basic_patterns
    symbolic_reasoning_SymbolicAtomSpace__initialize_basic_rules[_initialize_basic_rules()]
    symbolic_reasoning_SymbolicAtomSpace --> symbolic_reasoning_SymbolicAtomSpace__initialize_basic_rules
    symbolic_reasoning_SymbolicAtomSpace_add_atom[add_atom()]
    symbolic_reasoning_SymbolicAtomSpace --> symbolic_reasoning_SymbolicAtomSpace_add_atom
    symbolic_reasoning_SymbolicAtomSpace_add_link[add_link()]
    symbolic_reasoning_SymbolicAtomSpace --> symbolic_reasoning_SymbolicAtomSpace_add_link
```