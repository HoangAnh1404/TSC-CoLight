from tshub.sumo_tools.osm_build import scenario_build

osm_file = "/home/hoanganh04/Projects/LLM-Assisted-Light/TSCScenario/map/map.osm"
output_directory = "/home/hoanganh04/Projects/LLM-Assisted-Light/TSCScenario/map/env/"
scenario_build(
    osm_file=osm_file,
    output_directory=output_directory
)