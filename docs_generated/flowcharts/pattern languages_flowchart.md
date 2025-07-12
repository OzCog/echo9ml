# pattern languages Module Flowchart

```mermaid
graph TD
    pattern languages[pattern languages]
    pattern languages_CityPattern[CityPattern]
    pattern languages --> pattern languages_CityPattern
    pattern languages_CityPattern___init__[__init__()]
    pattern languages_CityPattern --> pattern languages_CityPattern___init__
    pattern languages_CityPattern_display[display()]
    pattern languages_CityPattern --> pattern languages_CityPattern_display
    pattern languages_MainStreet[MainStreet]
    pattern languages --> pattern languages_MainStreet
    pattern languages_MainStreet___init__[__init__()]
    pattern languages_MainStreet --> pattern languages_MainStreet___init__
    pattern languages_MainStreet_add_district[add_district()]
    pattern languages_MainStreet --> pattern languages_MainStreet_add_district
    pattern languages_PublicSquare[PublicSquare]
    pattern languages --> pattern languages_PublicSquare
    pattern languages_PublicSquare___init__[__init__()]
    pattern languages_PublicSquare --> pattern languages_PublicSquare___init__
    pattern languages_PublicSquare_host_meeting[host_meeting()]
    pattern languages_PublicSquare --> pattern languages_PublicSquare_host_meeting
    pattern languages_Neighborhood[Neighborhood]
    pattern languages --> pattern languages_Neighborhood
    pattern languages_Neighborhood___init__[__init__()]
    pattern languages_Neighborhood --> pattern languages_Neighborhood___init__
    pattern languages_Neighborhood_add_building[add_building()]
    pattern languages_Neighborhood --> pattern languages_Neighborhood_add_building
    pattern languages_Building[Building]
    pattern languages --> pattern languages_Building
    pattern languages_Building___init__[__init__()]
    pattern languages_Building --> pattern languages_Building___init__
    pattern languages_EnterpriseCity[EnterpriseCity]
    pattern languages --> pattern languages_EnterpriseCity
    pattern languages_EnterpriseCity___init__[__init__()]
    pattern languages_EnterpriseCity --> pattern languages_EnterpriseCity___init__
    pattern languages_EnterpriseCity_add_neighborhood[add_neighborhood()]
    pattern languages_EnterpriseCity --> pattern languages_EnterpriseCity_add_neighborhood
    pattern languages_EnterpriseCity_display_city[display_city()]
    pattern languages_EnterpriseCity --> pattern languages_EnterpriseCity_display_city
    pattern languages_main[main()]
    pattern languages --> pattern languages_main
```