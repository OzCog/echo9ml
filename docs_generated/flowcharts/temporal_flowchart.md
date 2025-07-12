# temporal Module Flowchart

```mermaid
graph TD
    temporal[temporal]
    temporal_SubGear[SubGear]
    temporal --> temporal_SubGear
    temporal_SubGear___init__[__init__()]
    temporal_SubGear --> temporal_SubGear___init__
    temporal_CoreGear[CoreGear]
    temporal --> temporal_CoreGear
    temporal_CoreGear___init__[__init__()]
    temporal_CoreGear --> temporal_CoreGear___init__
    temporal_CelestialTaskFramework[CelestialTaskFramework]
    temporal --> temporal_CelestialTaskFramework
    temporal_CelestialTaskFramework___init__[__init__()]
    temporal_CelestialTaskFramework --> temporal_CelestialTaskFramework___init__
```